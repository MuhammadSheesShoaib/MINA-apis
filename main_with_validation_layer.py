from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
import re
from fastapi.middleware.cors import CORSMiddleware
import os
from logging import Logger
from dotenv import load_dotenv
from groq import Groq
import time

# Load environment variables
load_dotenv()

# FastAPI setup
app = FastAPI(title="Therapeutic Chat API", description="API for therapeutic chat with sentiment analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys & Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY is missing in environment variables")

groq_client = Groq(api_key=GROQ_API_KEY)

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


def transform_to_mind_science(therapist_response: str) -> str:
    """
    Transform therapeutic response to mind science terminology using Groq API
    """
    system_prompt = """You are a response transformer. Your task is to transform therapeutic/psychological responses into "mind science" terminology while maintaining the same supportive tone and meaning.

Key transformations:
1. Always refer to yourself as "MINA, your mind coach"
2. Replace "mental health" with "mind science"
3. Replace "therapy/therapeutic" with "mind science approach"
4. Replace "therapist/psychologist/psychiatrist" with "mind coach"
5. Replace "psychological" with "mind science-based"
6. Replace "mental" with "cognitive" or "mind-related"
7. Keep the supportive, caring tone intact
8. Maintain the same level of empathy and understanding

Transform the response but keep it natural and conversational. Don't mention that you're transforming anything."""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ You can swap model here if needed
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transform this response: {therapist_response}"}
        ],
        max_tokens=150,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def get_sentiment_token(text: str) -> str:
    try:
        response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json={"inputs": text})
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
    start_time = time.time()  # ⏱️ Start timer
    try:
        if message.session_id not in chat_histories:
            chat_histories[message.session_id] = []

        text_lower = message.user_message.lower().strip()
        if text_lower in PREDEFINED_RESPONSES:
            predefined_reply = PREDEFINED_RESPONSES[text_lower]
            transformed_reply = transform_to_mind_science(predefined_reply)
            chat_histories[message.session_id].append({"from": "human", "value": f"<normal> {message.user_message}"})
            chat_histories[message.session_id].append({"from": "gpt", "value": transformed_reply})

            elapsed = round(time.time() - start_time, 2)  # ⏱️ End timer
            print(f"⏱️ /chat took {elapsed} seconds")  # backend log
            return ChatResponse(therapist_reply=transformed_reply, sentiment="normal")

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

        final_response = transform_to_mind_science(generated_text)

        chat_histories[message.session_id].append({"from": "gpt", "value": final_response})

        sentiment = sentiment_token.strip("<>")

        elapsed = round(time.time() - start_time, 2)  # ⏱️ End timer
        print(f"⏱️ /chat took {elapsed} seconds")  # backend log
        return ChatResponse(therapist_reply=final_response, sentiment=sentiment)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chat-history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str = "default"):
    try:
        Logger.info(f"Retrieving chat history for session: {session_id}")
        if session_id not in chat_histories:
            Logger.info(f"No chat history found for session: {session_id}")
            return ChatHistory(messages=[])
        history = chat_histories[session_id]
        Logger.info(f"Retrieved {len(history)} messages for session: {session_id}")
        return ChatHistory(messages=history)
    except Exception as e:
        Logger.error(f"Error retrieving chat history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")


@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str = "default"):
    try:
        Logger.info(f"Clearing chat history for session: {session_id}")
        if session_id in chat_histories:
            chat_histories[session_id] = []
            Logger.info(f"Chat history cleared for session: {session_id}")
        else:
            Logger.info(f"No chat history found to clear for session: {session_id}")
        return {"message": "Chat history cleared"}
    except Exception as e:
        Logger.error(f"Error clearing chat history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")
