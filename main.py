import json
import random
from uuid import uuid4
from datetime import datetime

# ====================== قوائم شاملة ======================
intents = [
    "suggest_hotels", "inquire_facilities", "predict_future_price", "inquire_current_price",
    "inquire_reviews", "compare_hotels", "ask_recommendation", "inquire_amenities",
    "inquire_location", "inquire_transport", "inquire_dining", "inquire_policy",
    "general_hotel_info"
]

dialogue_actions = [
    "show_options", "list_facilities", "provide_price_prediction", "ask_for_clarification",
    "describe_facility", "recommend_based_on_preferences", "show_price", "provide_comparison",
    "offer_alternatives", "confirm_understanding", "query_internal_db", "offer_more_info"
]

locations = ["Paris", "London", "New York", "Dubai", "Tokyo", "Rome", "Barcelona", "Bangkok", "Sydney", "Istanbul", "Cairo", "Madrid", "Berlin", "Amsterdam", "Singapore"]
hotels = ["Hilton", "Marriott", "Ritz-Carlton", "Four Seasons", "Sheraton", "InterContinental", "Hyatt", "Le Bristol", "Burj Al Arab", "Mandarin Oriental", "The Peninsula", "Waldorf Astoria"]
features = ["swimming pool", "gym", "spa", "free Wi-Fi", "beach access", "kids club", "sauna", "restaurant", "parking", "airport shuttle"]
room_types = ["single", "double", "suite", "deluxe"]
dates = ["next week", "in December", "this weekend", "in July", "next month", "during summer", "for Christmas"]

# ====================== مولد المحادثة ======================
def generate_conversation(conv_num):
    conversation_id = f"conv_{str(uuid4())[:8]}"
    num_turns = random.randint(4, 7)   # متوسط 5.5 turn
    turns = []
    
    # بداية المحادثة (Turn 1)
    location = random.choice(locations)
    feature = random.choice(features)
    hotel = random.choice(hotels) if random.random() > 0.4 else None
    
    main_intent = random.choice(intents)
    main_utterance = {
        "suggest_hotels": f"Suggest some good hotels in {location} with {feature}",
        "inquire_facilities": f"What facilities does {hotel or 'the best hotel in ' + location} have?",
        "predict_future_price": f"What is the predicted price for a double room in {location} {random.choice(dates)}?",
        "inquire_current_price": f"How much does a room cost in {hotel or location} right now?",
        "inquire_reviews": f"What are the recent reviews for {hotel or 'hotels in ' + location}?",
        "compare_hotels": f"Compare between Hilton and Marriott in {location}",
        "ask_recommendation": f"Which hotel would you recommend in {location} for a couple?",
        "inquire_amenities": f"What amenities are available in luxury hotels in {location}?",
        "inquire_location": f"Where is the best area to stay in {location}?",
        "inquire_transport": f"How can I get from the airport to hotels in {location}?",
        "inquire_dining": f"What are the best dining options near hotels in {location}?",
        "inquire_policy": f"What is the cancellation policy for most hotels in {location}?",
        "general_hotel_info": f"Tell me general information about hotels in {location}"
    }[main_intent]
    
    turns.append({
        "turn_id": 1,
        "guest_utterances": main_utterance,
        "intent": main_intent,
        "entities": {
            "location": location,
            "feature": feature if random.random() > 0.5 else None,
            "hotel_name": hotel,
            "room_type": random.choice(room_types) if random.random() > 0.6 else None,
            "date": random.choice(dates) if "price" in main_intent else None
        },
        "dialogue_action": "show_options" if main_intent == "suggest_hotels" else "list_facilities" if "facilities" in main_intent else "provide_price_prediction",
        "response": f"Here are some great options in {location}..." if main_intent == "suggest_hotels" else "The hotel offers gym, spa, pool..." 
    })
    
    # الـ Turns المتابعة (Multi-turn حقيقي)
    current_intent = main_intent
    for t in range(2, num_turns + 1):
        follow_intent = random.choice([i for i in intents if i != current_intent])
        current_intent = follow_intent
        
        utterance_templates = {
            "inquire_facilities": f"Can you tell me more about the {random.choice(features)} in one of the hotels you suggested?",
            "predict_future_price": f"What will be the price next week for the hotel you mentioned?",
            "inquire_reviews": "Are the reviews good for these hotels?",
            "compare_hotels": "Which one is better between the two?",
            "ask_recommendation": "Which one do you recommend for family?",
            "inquire_amenities": "Do they have free breakfast?"
        }
        
        guest_utt = utterance_templates.get(follow_intent, f"Tell me more about {follow_intent.replace('_', ' ')} in {location}")
        
        entity_dict = {"location": location}
        if hotel and random.random() > 0.5:
            entity_dict["hotel_name"] = hotel
        if "price" in follow_intent:
            entity_dict["date"] = random.choice(dates)
        
        turns.append({
            "turn_id": t,
            "guest_utterances": guest_utt,
            "intent": follow_intent,
            "entities": {k: v for k, v in entity_dict.items() if v is not None},
            "dialogue_action": random.choice(dialogue_actions),
            "response": "Based on your request, here is the detailed information..." if "price" in follow_intent else "Yes, it includes " + random.choice(features) + " and more."
        })
    
    return {"conversation_id": conversation_id, "turns": turns}

# ====================== إنشاء الملف ======================
print("جاري إنشاء 5,000 محادثة... (سيستغرق حوالي 20-40 ثانية)")

with open("conversational_dataset.jsonl", "w", encoding="utf-8") as f:
    for i in range(5000):
        conv = generate_conversation(i)
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        
        if (i + 1) % 500 == 0:
            print(f"تم إنشاء {i+1}/5000 محادثة...")

print("✅ تم إنشاء الملف بنجاح!")
print("الملف: conversational_dataset.jsonl")
print(f"عدد المحادثات: 5000")
print(f"متوسط عدد الـ turns: ~5.2")