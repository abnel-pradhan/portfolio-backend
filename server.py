from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import google.generativeai as genai

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── MongoDB Setup (With Safety Net) ──────────────────────────────────────────
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME', 'portfolio')

if mongo_url:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Error: {e}")
        db = None
else:
    logger.warning("🚨 MONGO_URL not found! Server will run, but chat history won't be saved.")
    db = None

# ── Gemini Setup ─────────────────────────────────────────────────────────────
# This pulls the key from your Render Environment Variables
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logger.error("❌ GEMINI_API_KEY is missing from environment variables!")

genai.configure(api_key=api_key)

app = FastAPI()
api_router = APIRouter(prefix="/api")

ABNEL_SYSTEM_PROMPT = """You are Abnel's AI Assistant — a sharp, concise, and friendly co-pilot embedded in Abnel Pradhan's 3D portfolio website.

ABOUT ABNEL:
- Name: Abnel Pradhan
- Role: BCA (Bachelor of Computer Applications) student and full-stack developer
- Location: Kathmandu, Nepal
- Passions: coding, video editing, and creative AI tools
- Personality: curious, hands-on, loves shipping products
- Status: Open for work

CORE SKILLS:
- React, Next.js, Node.js, MongoDB, Tailwind CSS
- JavaScript (ES6+), Three.js / React Three Fiber
- Git, DevOps basics, Video Editing

PROJECTS:
1. NewarPrime — Abnel's flagship learn & earn ecosystem.
2. Real-time Chat App — WebSocket-based chat with rooms.
3. Data Visualization Dashboard — Interactive D3.js charts.

HOW TO RESPOND:
- Keep responses short (1-3 sentences).
- Speak about Abnel in the third person.
- If asked something outside Abnel's portfolio, politely steer back.
- For contact, point them to the Contact form below.
"""

# Gemini Configuration
generation_config = {"temperature": 0.7, "max_output_tokens": 1024}
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash", # Use models/ prefix for stability
    generation_config=generation_config,
    system_instruction=ABNEL_SYSTEM_PROMPT
)

# ── Models ───────────────────────────────────────────────────────────────────
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str

class ContactResponse(BaseModel):
    id: str
    ok: bool = True

# ── Routes ───────────────────────────────────────────────────────────────────
@api_router.get("/")
async def root():
    return {"message": "Abnel Portfolio API is Live!"}

@api_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or f"abnel-{uuid.uuid4()}"

    # 1. Load history from MongoDB (if connected)
    gemini_history = []
    if db is not None:
        try:
            history_docs = await db.chat_messages.find(
                {"session_id": session_id}, {"_id": 0}
            ).sort("ts", 1).to_list(20)

            for h in history_docs:
                role = "model" if h.get("role") == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [h["content"]]})
        except Exception as e:
            logger.error(f"Error loading history: {e}")

    # 2. Get Gemini Response
    try:
        # Standard synchronous send_message is often more stable in Render's Uvicorn loop
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(req.message) 
        reply = response.text
    except Exception as e:
        # This print statement will show up in your Render Logs!
        print(f"DEBUG: Detailed Gemini Error: {type(e).__name__} - {e}")
        logger.exception("Gemini API error")
        raise HTTPException(status_code=502, detail="AI Communication Error")

    # 3. Save to MongoDB (if connected)
    if db is not None:
        try:
            now = datetime.now(timezone.utc).isoformat()
            await db.chat_messages.insert_many([
                {"session_id": session_id, "role": "user", "content": req.message, "ts": now},
                {"session_id": session_id, "role": "assistant", "content": reply, "ts": now},
            ])
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    return ChatResponse(session_id=session_id, reply=reply)

# ── App setup ────────────────────────────────────────────────────────────────
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Wide open for testing, can be restricted to your Netlify URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    if mongo_url and 'client' in globals():
        client.close()