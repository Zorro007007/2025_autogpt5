"""
Debate Engine — Multi‑Agent, Turn‑Based Threads (Profiles + User in the Loop)
-----------------------------------------------------------------------------

This module adds multi‑agent threads where profiles (PM, FE Dev, BE Dev,
Architect, QA, Critic, etc.) hold a turn‑based discussion that the user can
join at any time. It persists:
  • AgentProfile (persona + system prompt)
  • Thread (participants + phase)
  • Turn (one iteration in a thread)
  • Message (user/agent posts)

It also exposes a FastAPI router so you can bolt it onto your existing app:

  from debate_engine import router as debate_router
  app.include_router(debate_router, prefix="/debate", tags=["debate"])

By default, it reuses the same DATABASE_URL as your main API.
It reuses the LLM client abstraction from Phase‑1 (OpenAI or Mock)
so it honors OPENAI_API_KEY / OPENAI_MODEL env vars.

Endpoints (prefix /debate):
  - POST   /profiles/seed                       -> seed default profiles
  - GET    /profiles                            -> list profiles
  - POST   /projects/{project_id}/threads       -> create a thread
  - GET    /threads/{thread_id}                 -> thread info
  - GET    /threads/{thread_id}/messages        -> list messages (latest first)
  - POST   /threads/{thread_id}/messages        -> post a user message
  - POST   /threads/{thread_id}/step            -> advance one agent turn
  - POST   /threads/{thread_id}/autorun?n=3     -> run n steps
  - POST   /threads/{thread_id}/compile/phase1  -> compile transcript via Phase‑1
  - POST   /threads/{thread_id}/compile/phase2  -> compile transcript via Phase‑2

NOTE: For simplicity, this file has its own SQLAlchemy Base/engine + tables.
That’s fine as long as DATABASE_URL is the same. You can merge metadata
later if you prefer a single Alembic migration set.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, DateTime, String, Text, Integer, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Reuse LLM + orchestrators
from phase1_exploration_orchestrator import (
    LLMClient, OpenAIChatClient, MockLLMClient, default_llm as p1_default_llm,
)
from phase2_requirements_orchestrator import (
    run_phase2_requirements, run_phase2_requirements as p2_run,
)
from phase1_exploration_orchestrator import run_phase1_exploration as p1_run

# =====================
# Database setup
# =====================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./phase1.db")
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# =====================
# ORM models
# =====================

class AgentProfile(Base):
    __tablename__ = "agent_profiles"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # e.g., 'pm', 'frontend', 'backend', 'architect', 'qa', 'critic'
    system_prompt = Column(Text, nullable=False)
    tools = Column(Text, nullable=False, default="[]")  # JSON list
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, nullable=False, index=True)
    phase = Column(String, nullable=False)  # e.g., 'phase1-exploration', 'phase2-requirements', 'decomposition'
    title = Column(String, nullable=False)
    participants = Column(Text, nullable=False, default="[]")  # JSON list of AgentProfile IDs (speaker order)
    cursor = Column(Integer, nullable=False, default=0)  # next speaker index in participants
    status = Column(String, default="active")  # 'active'|'closed'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Turn(Base):
    __tablename__ = "turns"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"), index=True, nullable=False)
    index = Column(Integer, nullable=False)  # 0,1,2,...
    speaker_agent_id = Column(String, ForeignKey("agent_profiles.id"), nullable=True)  # None for user turns
    status = Column(String, default="complete")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("threads.id"), index=True, nullable=False)
    turn_id = Column(String, ForeignKey("turns.id"), index=True, nullable=True)
    sender_type = Column(String, nullable=False)  # 'user'|'agent'
    sender_agent_id = Column(String, ForeignKey("agent_profiles.id"), nullable=True)
    content = Column(Text, nullable=False)  # JSON string or text
    content_type = Column(String, default="json")  # 'json'|'text'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

Index("ix_messages_thread_created", Message.thread_id, Message.created_at)

Base.metadata.create_all(bind=engine)

# =====================
# Defaults / Profiles
# =====================
DEFAULT_PROFILES: List[Dict[str, Any]] = [
    {
        "name": "Project Manager",
        "role": "pm",
        "system_prompt": (
            "You are a pragmatic PM. Focus on scope, risks, KPIs, timeline, dependencies. "
            "Use concise, actionable language and propose next steps."
        ),
        "tools": [],
    },
    {
        "name": "Frontend Developer",
        "role": "frontend",
        "system_prompt": (
            "You are a senior Frontend Engineer. Propose UX flows, components, state, edge cases. "
            "Prefer simple, testable designs."
        ),
        "tools": [],
    },
    {
        "name": "Backend Developer",
        "role": "backend",
        "system_prompt": (
            "You are a senior Backend Engineer. Propose APIs (REST/GraphQL), data models, integration points, "
            "and failure modes."
        ),
        "tools": [],
    },
    {
        "name": "Architect",
        "role": "architect",
        "system_prompt": (
            "You are a Solution Architect. Ensure NFRs (security, performance, reliability, compliance), "
            "interfaces and boundaries are well defined."
        ),
        "tools": [],
    },
    {
        "name": "QA Engineer",
        "role": "qa",
        "system_prompt": (
            "You are a QA Engineer. Derive critical tests, acceptance checks, and edge cases."
        ),
        "tools": [],
    },
    {
        "name": "Critic",
        "role": "critic",
        "system_prompt": (
            "You are a rigorous critic. Identify ambiguities, missing requirements, NFR gaps, and risks."
        ),
        "tools": [],
    },
]

# =====================
# Pydantic I/O models
# =====================

class SeedOut(BaseModel):
    inserted: int

class ProfileOut(BaseModel):
    id: str
    name: str
    role: str

class CreateThreadIn(BaseModel):
    phase: str = Field(..., description="e.g., 'phase1-exploration', 'phase2-requirements', 'decomposition'")
    title: str
    participants: List[str] = Field(..., description="AgentProfile IDs (speaker order)")

class ThreadOut(BaseModel):
    id: str
    project_id: str
    phase: str
    title: str
    participants: List[str]
    cursor: int
    status: str
    created_at: datetime

class MessageOut(BaseModel):
    id: str
    thread_id: str
    sender_type: str
    sender_agent_id: Optional[str]
    content: Any
    created_at: datetime

class PostMessageIn(BaseModel):
    content: str
    content_type: str = Field("text", pattern="^(text|json)$")

class DebateConfig(BaseModel):
    max_context_messages: int = 12
    response_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "proposals": {"type": "array", "items": {"type": "string"}},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
    )

@dataclass
class AgentReply:
    message: str
    proposals: List[str]
    questions: List[str]

# =====================
# Helpers
# =====================

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_llm() -> LLMClient:
    # respect env: if OPENAI_API_KEY set -> OpenAI else Mock
    return OpenAIChatClient(os.getenv("OPENAI_MODEL", "gpt-4o-mini")) if os.getenv("OPENAI_API_KEY") else MockLLMClient()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s


def _fetch_messages(db: Session, thread_id: str, limit: int = 50) -> List[Message]:
    return (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )


def _build_user_prompt(transcript: List[Message], schema: Dict[str, Any], phase: str) -> str:
    # Compress last messages into a simple text block. Keep it short.
    lines: List[str] = []
    for m in reversed(transcript):  # chronological
        who = "USER" if m.sender_type == "user" else f"AGENT({m.sender_agent_id or 'unknown'})"
        content = _json_loads(m.content)
        if isinstance(content, dict) and "message" in content:
            msg_text = content.get("message", "")
        else:
            msg_text = content if isinstance(content, str) else _json_dumps(content)
        lines.append(f"{who}: {msg_text}")
    transcript_txt = "\n".join(lines[-800:])  # naive truncation

    return (
        "You are participating in a structured, turn‑based product discussion for phase: "
        f"{phase}. Respond with a short, actionable contribution.\n\n"
        "CONVERSATION (most recent last):\n" + transcript_txt + "\n\n"
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON for the given schema.\n"
        "- Keep 'message' concise (<= 120 words).\n"
        "- If you have specific suggestions or questions, use 'proposals' and 'questions'.\n\n"
        "JSON_SCHEMA:\n" + _json_dumps(schema)
    )


def _persist_agent_message(db: Session, thread: Thread, turn: Turn, agent_id: str, payload: Dict[str, Any]) -> Message:
    m = Message(
        thread_id=thread.id,
        turn_id=turn.id,
        sender_type="agent",
        sender_agent_id=agent_id,
        content=_json_dumps(payload),
        content_type="json",
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

# =====================
# Router
# =====================
router = APIRouter()

@router.post("/profiles/seed", response_model=SeedOut)
def seed_profiles(db: Session = Depends(get_db)):
    existing = {p.name for p in db.query(AgentProfile).all()}
    inserted = 0
    for p in DEFAULT_PROFILES:
        if p["name"] in existing:
            continue
        row = AgentProfile(
            name=p["name"], role=p["role"], system_prompt=p["system_prompt"], tools=_json_dumps(p.get("tools", []))
        )
        db.add(row)
        inserted += 1
    db.commit()
    return SeedOut(inserted=inserted)

@router.get("/profiles", response_model=List[ProfileOut])
def list_profiles(db: Session = Depends(get_db)):
    rows = db.query(AgentProfile).order_by(AgentProfile.created_at.asc()).all()
    return [ProfileOut(id=r.id, name=r.name, role=r.role) for r in rows]

@router.post("/projects/{project_id}/threads", response_model=ThreadOut)
def create_thread(project_id: str, data: CreateThreadIn, db: Session = Depends(get_db)):
    # Basic validation: ensure participants exist
    if not data.participants:
        raise HTTPException(400, "participants cannot be empty")
    profs = (
        db.query(AgentProfile)
        .filter(AgentProfile.id.in_(data.participants))
        .all()
    )
    if len(profs) != len(set(data.participants)):
        raise HTTPException(404, "One or more participants (AgentProfile IDs) not found")
    row = Thread(
        project_id=project_id,
        phase=data.phase,
        title=data.title,
        participants=_json_dumps(list(data.participants)),
        cursor=0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return ThreadOut(
        id=row.id,
        project_id=row.project_id,
        phase=row.phase,
        title=row.title,
        participants=json.loads(row.participants),
        cursor=row.cursor,
        status=row.status,
        created_at=row.created_at,
    )

@router.get("/threads/{thread_id}", response_model=ThreadOut)
def get_thread(thread_id: str, db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    return ThreadOut(
        id=t.id,
        project_id=t.project_id,
        phase=t.phase,
        title=t.title,
        participants=json.loads(t.participants),
        cursor=t.cursor,
        status=t.status,
        created_at=t.created_at,
    )

@router.get("/threads/{thread_id}/messages", response_model=List[MessageOut])
def list_messages(thread_id: str, db: Session = Depends(get_db)):
    msgs = _fetch_messages(db, thread_id, limit=200)
    out: List[MessageOut] = []
    for m in msgs:
        out.append(
            MessageOut(
                id=m.id,
                thread_id=m.thread_id,
                sender_type=m.sender_type,
                sender_agent_id=m.sender_agent_id,
                content=_json_loads(m.content),
                created_at=m.created_at,
            )
        )
    return out

@router.post("/threads/{thread_id}/messages", response_model=MessageOut)
def post_user_message(thread_id: str, data: PostMessageIn, db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    # Create a user message (no turn needed but we can still attach a synthetic turn)
    turn_idx = (
        db.query(Turn).filter(Turn.thread_id == t.id).count()
    )
    tr = Turn(thread_id=t.id, index=turn_idx, speaker_agent_id=None, status="complete")
    db.add(tr)
    db.commit(); db.refresh(tr)

    msg = Message(
        thread_id=t.id,
        turn_id=tr.id,
        sender_type="user",
        sender_agent_id=None,
        content=data.content if data.content_type == "text" else _json_dumps(_json_loads(data.content)),
        content_type=data.content_type,
    )
    db.add(msg)
    db.commit(); db.refresh(msg)

    return MessageOut(
        id=msg.id, thread_id=msg.thread_id, sender_type=msg.sender_type,
        sender_agent_id=None, content=_json_loads(msg.content), created_at=msg.created_at
    )

@router.post("/threads/{thread_id}/step", response_model=MessageOut)
def debate_step(thread_id: str, cfg: DebateConfig = DebateConfig(), db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    order: List[str] = json.loads(t.participants)
    if not order:
        raise HTTPException(400, "Thread has no participants")
    speaker_id = order[t.cursor % len(order)]
    agent: Optional[AgentProfile] = db.get(AgentProfile, speaker_id)
    if not agent:
        raise HTTPException(404, "Next speaker AgentProfile not found")

    # Build prompt from recent transcript
    transcript = _fetch_messages(db, thread_id, limit=cfg.max_context_messages)
    user_prompt = _build_user_prompt(transcript, cfg.response_schema, t.phase)

    # Call LLM with agent's persona as system
    llm = get_llm()
    raw = llm.complete_json(system=agent.system_prompt, user=user_prompt)

    # Normalize output
    if not isinstance(raw, dict) or "message" not in raw:
        raw = {"message": str(raw)}
    payload = {
        "message": raw.get("message", ""),
        "proposals": raw.get("proposals", []),
        "questions": raw.get("questions", []),
    }

    # Persist turn + message
    next_index = db.query(Turn).filter(Turn.thread_id == t.id).count()
    tr = Turn(thread_id=t.id, index=next_index, speaker_agent_id=agent.id, status="complete")
    db.add(tr); db.commit(); db.refresh(tr)

    msg = _persist_agent_message(db, t, tr, agent.id, payload)

    # Advance cursor
    t.cursor = (t.cursor + 1) % len(order)
    db.add(t); db.commit()

    return MessageOut(
        id=msg.id, thread_id=msg.thread_id, sender_type="agent",
        sender_agent_id=agent.id, content=payload, created_at=msg.created_at
    )

@router.post("/threads/{thread_id}/autorun", response_model=List[MessageOut])
def debate_autorun(thread_id: str, n: int = 3, db: Session = Depends(get_db)):
    out: List[MessageOut] = []
    for _ in range(max(0, n)):
        out.append(debate_step(thread_id, db=db))
    return out

# === Compilation to artifacts (Adapter B) ===

class CompileOut(BaseModel):
    artifact: Dict[str, Any]

@router.post("/threads/{thread_id}/compile/phase1", response_model=CompileOut)
def compile_phase1(thread_id: str, db: Session = Depends(get_db)):
    # Use the earliest user message as the seed idea; if none, use latest text
    msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    idea = None
    for m in msgs:
        if m.sender_type == "user":
            content = _json_loads(m.content)
            idea = content if isinstance(content, str) else json.dumps(content)
            break
    if not idea and msgs:
        content = _json_loads(msgs[0].content)
        idea = content if isinstance(content, str) else json.dumps(content)
    if not idea:
        raise HTTPException(400, "No content in thread to compile")

    llm = get_llm()
    art = p1_run(idea, llm=llm)
    return CompileOut(artifact=art.model_dump())

@router.post("/threads/{thread_id}/compile/phase2", response_model=Dict[str, Any])
def compile_phase2(thread_id: str, db: Session = Depends(get_db)):
    # Same heuristic: seed idea from earliest user message
    msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    idea = None
    for m in msgs:
        if m.sender_type == "user":
            content = _json_loads(m.content)
            idea = content if isinstance(content, str) else json.dumps(content)
            break
    if not idea and msgs:
        content = _json_loads(msgs[0].content)
        idea = content if isinstance(content, str) else json.dumps(content)
    if not idea:
        raise HTTPException(400, "No content in thread to compile")

    llm = get_llm()
    outputs = p2_run(idea, llm=llm)
    # Return as plain dict for convenience
    return {
        "requirements": outputs["requirements"].model_dump(),
        "tests": outputs["tests"].model_dump(),
    }

if __name__ == "__main__":
    # Ad‑hoc demo run: create a thread and run a few steps
    from fastapi import FastAPI
    import uvicorn

    demo = FastAPI()
    demo.include_router(router, prefix="/debate", tags=["debate"])
    print("Running demo debate API at http://localhost:8001/debate ...")
    uvicorn.run(demo, host="0.0.0.0", port=8001)
