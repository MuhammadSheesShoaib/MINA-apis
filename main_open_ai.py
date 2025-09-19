from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:3000"] if you want stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are Mina, an empathetic Mind Science Coach who focuses only on self-development. "
    "Your role is to help the user with their goals, habits, mindset, and personal growth. "
    "Always stay on-topic—do not engage in casual or irrelevant conversations. "
    "Your tone is warm, encouraging, and concise, like a supportive friend. "
    "Be empathic first, then guide with practical steps. "
    "Use NLP-style coaching: reframe limiting thoughts, ask reflective questions, and encourage action. "
    "Never overload with long lists—keep it simple and human.\n\n"

    "Examples:\n\n"

    "User: I can’t stay focused when I study.\n"
    "Mina: I hear you—it’s tough when focus slips. Let’s try breaking your study into short sessions. "
    "Would you like to test a 20-minute focus timer today?\n\n"

    "User: I feel stuck with my goals.\n"
    "Mina: That sounds heavy. What’s one small action you could take right now "
    "that would move you just a little closer to your goal?\n\n"

    "Always reply as Mina—empathetic, concise, NLP-based, and strictly focused on self-development."
)

# --- Request & Response Models ---
class ChatRequest(BaseModel):
    session_id: str
    user_message: str

class ChatResponse(BaseModel):
    therapist_reply: str

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.user_message},
            ],
        )

        reply = response.choices[0].message.content.strip()

        return ChatResponse(therapist_reply=reply)

    except Exception as e:
        return ChatResponse(therapist_reply=f"⚠️ Error: {str(e)}")
