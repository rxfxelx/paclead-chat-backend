import os
import base64
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI SDK v1.x
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Defina OPENAI_API_KEY no ambiente ou no .env")

# Modelos padrão (ajuste conforme necessário)
TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Helsen IA Backend", version="1.0.0")

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str = Field(..., description="user|assistant|system")
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)
    system: Optional[str] = Field(default=None)

class VisionUrlRequest(BaseModel):
    prompt: Optional[str] = Field(default="Descreva a imagem de forma objetiva.")
    image_url: str

def _mk_messages(system: Optional[str], history: List[ChatMessage], user_text: Optional[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for m in history or []:
        messages.append({"role": m.role, "content": m.content})
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})
    return messages

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        messages = _mk_messages(req.system, req.history or [], req.message)
        # OpenAI Chat Completions
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return {"ok": True, "type": "text", "output": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/url")
def vision_url(req: VisionUrlRequest):
    """
    Analisa imagem via URL + prompt opcional.
    """
    try:
        user_content = [
            {"type": "text", "text": req.prompt or "Descreva a imagem."},
            {"type": "image_url", "image_url": {"url": req.image_url}},
        ]
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return {"ok": True, "type": "vision", "output": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision/upload")
async def vision_upload(
    image: UploadFile = File(..., description="arquivo de imagem"),
    prompt: Optional[str] = Form(default="Analise a imagem objetivamente."),
):
    """
    Analisa imagem enviada por upload (gera data URL base64).
    """
    try:
        data = await image.read()
        mime = image.content_type or "image/png"
        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        user_content = [
            {"type": "text", "text": prompt or "Descreva a imagem."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return {"ok": True, "type": "vision", "output": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "text_model": TEXT_MODEL, "vision_model": VISION_MODEL}
