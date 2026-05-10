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

# ── MongoDB Setup ────────────────────────────────────────────────────────────
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME', 'portfolio')

if mongo_url:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Error: {e}")
        db = None
else:
    db = None

# ── Gemini Setup (THE MAGIC FIX) ─────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 1. Ask Google what models we are actually allowed to use
working_model_name = "models/gemini-1.5-flash" # default fallback
try:
    print("\n🔍 SCANNING FOR ALLOWED GEMINI MODELS...")
    valid_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            valid_models.append(m.name)
    
    print(f"✅ Google says we can use: {valid_models}")
    
    # 2. Auto-select the first working model from the list!
    if valid_models:
        working_model_name = valid_models[0]
        print(f"🚀 AUTO-SELECTING: {working_model_name}\n")
except Exception as e:
    print(f"⚠️ Could not auto-fetch models: {e}\n")

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

HOW TO RESPOND:
- Keep responses short (1-3 sentences).
- Speak about Abnel in the third person.
- Point to the contact form for hiring inquiries.
"""

# 3. Initialize the model with the auto-detected name!
model = genai.GenerativeModel(
    model_name=working_model_name,
    generation_config={"temperature": 0.7, "max_output_tokens": 1024},
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

    gemini_history = []
    if db is not None:
        try:
            history_docs = await db.chat_messages.find(
                {"session_id": session_id}, {"_id": 0}
            ).sort("ts", 1).to_list(20)
            for h in history_docs:
                role = "model" if h.get("role") == "assistant" else "user"
                gemini_history.append({"role": role, "parts": [h["content"]]})
        except Exception:
            pass

    try:
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(req.message) 
        reply = response.text
    except Exception as e:
        print(f"REAL ERROR DURING CHAT: {str(e)}")
        raise HTTPException(status_code=502, detail=f"AI error: {str(e)}")

    if db is not None:
        try:
            now = datetime.now(timezone.utc).isoformat()
            await db.chat_messages.insert_many([
                {"session_id": session_id, "role": "user", "content": req.message, "ts": now},
                {"session_id": session_id, "role": "assistant", "content": reply, "ts": now},
            ])
        except Exception:
            pass

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    if mongo_url and 'client' in globals():
        client.close()