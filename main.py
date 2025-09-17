import os
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, String, Text, DateTime, Enum, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="AI Search + Review System")

# Allow CORS so WordPress frontend can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your WP domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Database setup
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# -----------------------
# Review Models
# -----------------------
class ReviewStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PUBLISHED = "PUBLISHED"
    QUARANTINED = "QUARANTINED"
    REMOVED = "REMOVED"


class Review(Base):
    __tablename__ = "reviews"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    meta = Column("metadata", JSON, nullable=True)  # ✅ safe: Python attr = meta, DB column = metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    decision_reason = Column(String, nullable=True)
    ai_scores = Column(JSON, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    review_id = Column(String, nullable=False)
    action = Column(String, nullable=False)  # e.g. AUTO_QUARANTINE, MOD_APPROVE
    actor = Column(String, nullable=False, default="system")
    timestamp = Column(DateTime, default=datetime.utcnow)
    note = Column(Text, nullable=True)


Base.metadata.create_all(bind=engine)


# -----------------------
# API Schemas
# -----------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


class IngestRequest(BaseModel):
    content: str
    metadata: dict = {}


# -----------------------
# Demo Search Endpoint
# -----------------------
DEMO_AI_TOOLS = [
    {"id": 1, "title": "DALL·E 3", "url": "https://openai.com/dall-e", "excerpt": "Generates AI images from text prompts."},
    {"id": 2, "title": "ChatGPT", "url": "https://chat.openai.com", "excerpt": "AI chatbot that can write, debug, summarize text."},
    {"id": 3, "title": "MidJourney", "url": "https://www.midjourney.com", "excerpt": "AI tool for generating creative images."},
    {"id": 4, "title": "Whisper", "url": "https://openai.com/research/whisper", "excerpt": "OpenAI model for speech-to-text transcription."},
    {"id": 5, "title": "Claude AI", "url": "https://www.anthropic.com", "excerpt": "AI assistant for research, writing, and summarization."},
    {"id": 6, "title": "Jasper AI", "url": "https://www.jasper.ai", "excerpt": "AI copywriting assistant for marketing and content."},
    {"id": 7, "title": "Runway", "url": "https://runwayml.com", "excerpt": "AI-powered video and image editing platform."}
]


@app.post("/search")
async def search(req: SearchRequest):
    query = req.query.lower()
    filtered = [
        tool for tool in DEMO_AI_TOOLS
        if any(word in query for word in tool["title"].lower().split() + tool["excerpt"].lower().split())
    ]
    if not filtered:
        filtered = DEMO_AI_TOOLS[:4]
    return {"results": filtered[:req.top_k]}


# -----------------------
# Review System Endpoints
# -----------------------
@app.post("/ingest_review")
async def ingest_review(req: IngestRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Missing content")

    db = SessionLocal()
    review = Review(content=req.content, metadata=req.metadata)
    db.add(review)
    db.commit()
    db.refresh(review)   # ✅ ensures we get the ID
    review_id = review.id  # ✅ save ID while session is alive
    db.close()
    return {"status": "ok", "review_id": review_id, "message": "Review queued for processing"}


@app.get("/review/{rid}")
async def get_review(rid: str):
    db = SessionLocal()
    r = db.query(Review).filter(Review.id == rid).first()
    db.close()
    if not r:
        raise HTTPException(status_code=404)
    return {
        "id": r.id,
        "content": r.content,
        "status": r.status,
        "decision_reason": r.decision_reason,
        "ai_scores": r.ai_scores,
        "created_at": r.created_at,
    }


@app.get("/admin/quarantined")
async def list_quarantined():
    db = SessionLocal()
    rows = db.query(Review).filter(Review.status == ReviewStatus.QUARANTINED).all()
    db.close()
    return {"items": [{"id": r.id, "content": r.content, "metadata": r.metadata, "ai_scores": r.ai_scores} for r in rows]}


@app.post("/admin/approve/{rid}")
async def approve_review(rid: str, moderator: str = Body("moderator")):
    db = SessionLocal()
    r = db.query(Review).filter(Review.id == rid).first()
    if not r:
        raise HTTPException(status_code=404)
    r.status = ReviewStatus.PUBLISHED
    r.decision_reason = "Moderator approved"
    db.add(AuditLog(review_id=rid, action="MOD_APPROVE", actor=moderator, note="Approved"))
    db.commit()
    db.close()
    return {"ok": True}


@app.post("/admin/reject/{rid}")
async def reject_review(rid: str, moderator: str = Body("moderator"), note: str = Body(None)):
    db = SessionLocal()
    r = db.query(Review).filter(Review.id == rid).first()
    if not r:
        raise HTTPException(status_code=404)
    r.status = ReviewStatus.REMOVED
    r.decision_reason = "Moderator removed"
    db.add(AuditLog(review_id=rid, action="MOD_REJECT", actor=moderator, note=note))
    db.commit()
    db.close()
    return {"ok": True}
