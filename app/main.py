from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.db.database import init_db, get_db, ChatMessage
from app.models.schemas import ChatRequest, ChatResponse
from app.agents.agent import run_agent

app = FastAPI(
    title="HR Policy Assistant",
    description="Agentic AI assistant that answers HR policy questions using RAG + tool-calling.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    db.add(ChatMessage(thread_id=request.thread_id, role="user", content=request.message))
    db.commit()

    reply = run_agent(request.message, thread_id=request.thread_id)

    db.add(ChatMessage(thread_id=request.thread_id, role="assistant", content=reply))
    db.commit()

    return ChatResponse(thread_id=request.thread_id, reply=reply)


@app.get("/history/{thread_id}")
def history(thread_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.thread_id == thread_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in messages
    ]
