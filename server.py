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
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
else:
    logger.warning("🚨 MONGO_URL not found! Server will run, but chat history won't be saved.")
    db = None

# ── Gemini Setup ─────────────────────────────────────────────────────────────
# This automatically grabs the key we put in Render!
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()
api_router = APIRouter(prefix="/api")

ABNEL_SYSTEM_PROMPT = """You are Abnel's AI Assistant — a sharp, concise, and friendly co-pilot embedded in Abnel Pradhan's 3D portfolio website.

ABOUT ABNEL:
- Name: Abnel Pradhan
- Role: BCA (Bachelor of Computer Applications) student and builder
- Location: Kathmandu, Nepal
- Passions: coding, video editing, and creative AI tools
- Personality: curious, hands-on, loves shipping products
- Status: Open for work

CORE SKILLS:
- JavaScript (ES6+), React & Next.js, Tailwind CSS
- Node.js & Express
- SQL / NoSQL databases (MongoDB)
- Git & DevOps basics
- Full-stack web development
- Three.js / React Three Fiber
- Creative AI tools & video editing

PROJECTS:
1. NewarPrime — Abnel's flagship product. A Learn & Earn ecosystem that rewards users for learning skills and completing missions. Stack: React, Node, MongoDB, AI.
2. Real-time Chat App — Low-latency WebSocket chat with rooms, presence, and typing indicators. Stack: Vue.js, Socket.IO, Express.
3. Data Visualization Dashboard — Interactive charts and filters for complex datasets. Stack: HTML/CSS/JS, D3.js, Tailwind.

HOW TO RESPOND:
- Keep responses short (1-3 sentences) and punchy unless the user asks for more detail.
- Speak about Abnel in the third person ("Abnel built...", "He uses...").
- If asked something outside Abnel's portfolio, politely steer back to his work.
- For contact, tell them to use the Contact form at the bottom of the page.
- Don't invent facts. If unknown, say "I don't have that info — drop Abnel a message via the contact form."
"""

# Gemini Configuration
generation_config = {"temperature": 0.7, "max_output_tokens": 1024}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
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

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(**input.model_dump())
    if db is not None:
        doc = status_obj.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is None:
        return []
    rows = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for r in rows:
        if isinstance(r['timestamp'], str):
            r['timestamp'] = datetime.fromisoformat(r['timestamp'])
    return rows

@api_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or f"abnel-{uuid.uuid4()}"

    # 1. Load history from MongoDB (if connected)
    gemini_history = []
    if db is not None:
        history_docs = await db.chat_messages.find(
            {"session_id": session_id}, {"_id": 0}
        ).sort("ts", 1).to_list(20)

        for h in history_docs:
            role = "model" if h.get("role") == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [h["content"]]})

    # 2. Get Gemini Response
    try:
        chat_session = model.start_chat(history=gemini_history)
        response = await chat_session.send_message_async(req.message)
        reply = response.text
    except Exception as e:
        logger.exception("Gemini API error")
        raise HTTPException(status_code=502, detail=f"AI error: {e}")

    # 3. Save to MongoDB (if connected)
    if db is not None:
        now = datetime.now(timezone.utc).isoformat()
        await db.chat_messages.insert_many([
            {"session_id": session_id, "role": "user", "content": req.message, "ts": now},
            {"session_id": session_id, "role": "assistant", "content": reply, "ts": now},
        ])

    return ChatResponse(session_id=session_id, reply=reply)

@api_router.post("/contact", response_model=ContactResponse)
async def contact(req: ContactRequest):
    doc = {
        "id": str(uuid.uuid4()),
        "name": req.name,
        "email": req.email,
        "message": req.message,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if db is not None:
        await db.contact_messages.insert_one(doc)
    return ContactResponse(id=doc["id"], ok=True)

# ── App setup ────────────────────────────────────────────────────────────────
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    if mongo_url and 'client' in globals():
        client.close()