from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", override=True)

# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal

from rag_pipeline import answer_question

app = FastAPI(title="RAG API")

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []   # ✅ add this

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    # ✅ pass history to rag pipeline
    return answer_question(req.message, history=req.history)
