import torch
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import math
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
bert_score = evaluate.load("bertscore")
dialogue_datasets = load_dataset(
    "json",
    data_files={
        "train": "/teamspace/studios/this_studio/ai_things/data/processed/convs_dialogues_train.jsonl",
        "valid": "/teamspace/studios/this_studio/ai_things/data/processed/convs_dialogues_valid.jsonl",
        "test": "/teamspace/studios/this_studio/ai_things/data/processed/convs_dialogues_test.jsonl",
    },
)
tokenizer = AutoTokenizer("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token


def prepare_messages(messages: list):
    system_content = None
    new_messages = []
    for message in messages:
        if message["role"] == "system":
            system_content = message["content"]
        elif message["role"] == "user":
            if system_content is not None:
                message["content"] = f"{system_content}\n\n{message["content"]}"
                system_content = None
            new_messages.append(message)
        else:
            new_messages.append(message)
    return new_messages


def template_conv(sample):
    messages = prepare_messages(sample["message"])
    template_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    template_prompt += "<|endoftext|>"
    return {"text": template_prompt}


dialogue_datasets = dialogue_datasets.map(template_conv, remove_columns=["message"])


def tokenize(sample):
    return tokenizer(
        sample["text"], truncation=True, padding="max_length", mac_length=2048
    )


dialogue_datasets = dialogue_datasets.map(
    tokenize, batched=True, remove_columns=["text"]
)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_prtrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=quant_config,
    device_map="auto",
    torch_type=torch.float16,
).to(device)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters
training_args = TrainingArguments(
    output_dir="./result/mistral-7b-hotel-assistant",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    logging_dir="./logs",
)


def compute_metrics(eval_preds):
    eval_loss = (
        eval_preds.metrics["eval_loss"]
        if "eval_loss" in eval_preds.metrics
        else eval_preds[0]
    )
    preplexity = math.exp(eval_loss)
    logistc, labels = eval_preds
    pred_ids = np.argmax(logistc, axis=-1)
    labels = np.where(labels != 100, labels, tokenizer.pad_token_id)
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bert_results = bert_score.compute(
        predictions=pred_texts, references=label_texts, lang="en"
    )
    precison = np.mean(bert_results["precion"])
    recall = np.mean(bert_results["recall"])
    f1_score = np.mean(bert_results["f1"])
    return {
        "preplexity": preplexity,
        "precision": precison,
        "recall": recall,
        "f1-score": f1_score,
    }


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dialogue_datasets["train"],
    eval_dataset=dialogue_datasets["valid"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

result = [trainer.evaluate(eval_dataset=dialogue_datasets["test"])]
df_result = pd.DataFrame(result)
output_paths = {
    "result": "/teamspace/studios/this_studio/ai_things/train_models/chatbot/results/mtcs_perf_diag_clf.csv",
    "bert": "/teamspace/studios/this_studio/ai_things/train_models/chatbot/models/diag_model",
}
df_result.to_csv(output_paths["result"], index=False)
model.save_pretrained(output_paths["bert"])
tokenizer.save_pretrained(output_paths["bert"])
