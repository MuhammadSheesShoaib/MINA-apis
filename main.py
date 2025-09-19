from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
import re  # 🔍 For cleaning special tokens
from fastapi.middleware.cors import CORSMiddleware
import time
import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
app = FastAPI(title="Therapeutic Chat API", description="API for therapeutic chat with sentiment analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
SENTIMENT_API_URL = "https://b2vn59hbkrg6rwjq.us-east-1.aws.endpoints.huggingface.cloud"
SENTIMENT_HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/json"
}

SENTIMENT_LABELS = {
    "LABEL_0": "<normal>",
    "LABEL_1": "<depression>",
    "LABEL_2": "<suicidal>",
    "LABEL_3": "<anxiety>",
    "LABEL_4": "<stress>",
    "LABEL_5": "<bipolar>",
    "LABEL_6": "<personality_disorder>"
}

PREDEFINED_RESPONSES = {
    "hi": "Hi, how are you doing today?",
    "hello": "Hello there, I'm here to listen. What's on your mind?",
    "hey": "Hey! What would you like to talk about today?",
    "good morning": "Good morning. How are you feeling today?",
    "good afternoon": "Good afternoon. I'm here for you — how can I support you?",
    "good evening": "Good evening. What's been on your mind lately?",
    "how are you": "I'm here and ready to support you. How are you doing?",
    "thanks": "You're welcome. I'm here whenever you need to talk.",
    "thank you": "You're very welcome. I'm glad to be here for you.",
    "bye": "Take care of yourself. I'm always here if you need me.",
    "goodbye": "Goodbye for now. You're not alone.",
}

# Pydantic models
class ChatMessage(BaseModel):
    session_id: str = None
    user_message: str

class ChatResponse(BaseModel):
    therapist_reply: str
    sentiment: str

class ChatHistory(BaseModel):
    messages: List[Dict]

# Store chat histories
chat_histories = {}

def get_sentiment_token(text: str) -> str:
    try:
        start_time = time.time()  # ⏱️ Start timer for sentiment
        response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json={"inputs": text})
        elapsed = round(time.time() - start_time, 2)  # ⏱️ End timer
        print(f"⏱️ Sentiment analysis took {elapsed} seconds")

        output = response.json()
        if isinstance(output, list) and len(output) > 0:
            predictions = output[0]
            predicted_label_id = predictions["label"]
            return SENTIMENT_LABELS.get(predicted_label_id, "<unknown>")
        return "<normal>"
    except Exception:
        return "<normal>"


def build_prompt(chat_history: List[Dict]) -> str:
    prompt = "<s>"
    prompt += (
        "You are a highly trained, compassionate psychotherapist helping users navigate mental and emotional challenges. "
        "You specialize in offering empathetic, non-judgmental, trauma-informed care. "
        "Before every user message, you will see an emotion tag such as <depression>, <stress>, <anxiety>, etc. "
        "Use this tag to adjust the tone, sensitivity, and focus of your response, ensuring it is appropriate to the user's emotional state. "
        "Do not reference or repeat the tag in your response. Be gentle, validating, and supportive while guiding the user through their thoughts and feelings.\n\n"
    )
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    prompt += "</s>\nTherapist: "
    return prompt

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    start_time = time.time()
    try:
        if message.session_id not in chat_histories:
            chat_histories[message.session_id] = []

        text_lower = message.user_message.lower().strip()
        if text_lower in PREDEFINED_RESPONSES:
            predefined_reply = PREDEFINED_RESPONSES[text_lower]
            chat_histories[message.session_id].append({"from": "human", "value": f"<normal> {message.user_message}"})
            chat_histories[message.session_id].append({"from": "gpt", "value": predefined_reply})

            elapsed = round(time.time() - start_time, 2)  # ⏱️ End timer
            print(f"⏱️ /chat took {elapsed} seconds")
            return ChatResponse(therapist_reply=predefined_reply, sentiment="normal")

        sentiment_token = get_sentiment_token(message.user_message)
        combined_input = f"{sentiment_token} {message.user_message}"
        chat_histories[message.session_id].append({"from": "human", "value": combined_input})

        prompt_text = build_prompt(chat_histories[message.session_id])

        generation_response = requests.post(
            url="https://xrd6kt9cqs9nzmwd.us-east-1.aws.endpoints.huggingface.cloud/v1/completions",
            headers={
                "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "tgi",
                "prompt": prompt_text,
                "max_tokens": 80,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

        completion = generation_response.json()
        if "choices" in completion and len(completion["choices"]) > 0:
            raw_text = completion["choices"][0]["text"].strip()
            clean_text = re.sub(r"<[^>]+>", "", raw_text).strip().split("\n")[0].strip()
            generated_text = clean_text
        else:
            generated_text = "I'm here to listen and support you. Could you tell me more about how you're feeling?"

        chat_histories[message.session_id].append({"from": "gpt", "value": generated_text})

        sentiment = sentiment_token.strip("<>")

        elapsed = round(time.time() - start_time, 2)  # ⏱️ End timer
        print(f"⏱️ /chat took {elapsed} seconds")
        return ChatResponse(therapist_reply=generated_text, sentiment=sentiment)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str = "default"):
    if session_id not in chat_histories:
        return ChatHistory(messages=[])
    return ChatHistory(messages=chat_histories[session_id])

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return {"message": "Chat history cleared"}

# if __name__ == "_main_":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)