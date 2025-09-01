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

# ---------------- System Prompt Padrão ----------------
SYSTEM_PROMPT = """Você é a IA do aplicativo PAC Lead. Siga as instruções abaixo e responda sempre em português, de forma clara e objetiva.

1) Tela de Análise:
- Explique que o usuário verá o número total de conversas iniciadas, classificação dos leads (normais, qualificados, quentes), objetivos alcançados, taxa de conversão e outros indicadores como horário de conversão e produto mais falado.

2) Tela da Empresa:
- Oriente a inserir o CNPJ ou CPF para preencher automaticamente razão social, nome fantasia, contato e endereço. Informe que é possível fazer upload de uma foto da empresa.

3) Tela de Configuração do Agente de IA:
- Explique que pode definir nome do agente, setor de atuação, estilo de comunicação (formal, descontraído, etc.), perfil do agente (consultivo, vendedor, acolhedor, etc.) e fazer upload de uma foto. Também é possível adicionar observações ou scripts.

4) Tela de Produtos:
- Instrua a adicionar nome, preço, categoria, descrição e imagem de cada produto. A IA pode ajudar a criar a descrição (copy) a partir das informações inseridas. Explique que existe busca para localizar produtos e que é possível editar qualquer item clicando nele.

5) Tela de Usuários:
- Oriente a adicionar novas contas de acesso sem compartilhar a própria senha. Basta preencher nome de usuário, e-mail, senha e uma observação para identificar o dispositivo ou usuário.

Regra de limite:
- Se o usuário pedir para criar uma DESCRIÇÃO de qualquer coisa, responda com no máximo 250 caracteres.
"""

# ---------------- Schemas ----------------
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

# ---------------- Helpers ----------------
def _mk_messages(system: Optional[str], history: List[ChatMessage], user_text: Optional[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for m in history or []:
        messages.append({"role": m.role, "content": m.content})
    if user_text is not None:
        messages.append({"role": "user", "content": user_text})
    return messages

# ---------------- Routes ----------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Chat de texto. Usa o SYSTEM_PROMPT por padrão, podendo ser
    substituído ao enviar "system" no body.
    """
    try:
        system_prompt = req.system or SYSTEM_PROMPT
        messages = _mk_messages(system_prompt, req.history or [], req.message)

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
    Aplica o SYSTEM_PROMPT para manter o estilo/comportamento.
    """
    try:
        user_content = [
            {"type": "text", "text": req.prompt or "Descreva a imagem."},
            {"type": "image_url", "image_url": {"url": req.image_url}},
        ]
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
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
    Aplica o SYSTEM_PROMPT para manter o estilo/comportamento.
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
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return {"ok": True, "type": "vision", "output": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "text_model": TEXT_MODEL, "vision_model": VISION_MODEL}
