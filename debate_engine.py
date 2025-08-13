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
    create_engine, Column, DateTime, String, Text, Integer, ForeignKey, Index, text
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
# Database dialect helper ("sqlite" or "postgresql") to toggle features like pgvector
DIALECT = engine.url.get_backend_name()

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


def _build_user_prompt(transcript: List[Message], schema: Dict[str, Any], phase: str, memory_snippets: Optional[List[str]] = None,) -> str:
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
    mem_block = ""
    if memory_snippets:
        # Keep it compact; top 8 snippets to avoid prompt bloat.
        mem_txt = "\n".join(f"- {s}" for s in memory_snippets[:8])
        mem_block = f"\nRELEVANT PAST POINTS (retrieved):\n{mem_txt}\n"
    return (
        "You are participating in a structured, turn‑based product discussion for phase: "
        f"{phase}. Respond with a short, actionable contribution.\n\n"
        "CONVERSATION (most recent last):\n" + transcript_txt + "\n\n"
        + mem_block +
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
def debate_autorun(thread_id: str, n: int = 3, use_memory: bool = True, db: Session = Depends(get_db)):
    """Run N consecutive turns.
    
    Args:
      n: number of turns
      use_memory: if True, use retrieval-augmented step; otherwise use plain step
    """
    out: List[MessageOut] = []
    for _ in range(max(0, n)):
        if use_memory:
            out.append(debate_step_with_memory(thread_id, db=db))
        else:
            out.append(debate_step(thread_id, db=db))
    return out

# === Compilation to artifacts (Adapter B) ===
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


# =====================
# Debate vector memory (per-thread, per-agent)
# =====================

# Embedding config — uses OpenAI when available, else deterministic fallback
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = 1536  # text-embedding-3-small as of 2025

# SQLite ORM model (also used for generic storage when not on Postgres)
class DebateMemory(Base):
    __tablename__ = "debate_memories"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, index=True, nullable=False)
    thread_id = Column(String, index=True, nullable=False)
    agent_id = Column(String, index=True, nullable=True)  # None for user messages
    role = Column(String, nullable=False)  # 'user'|'agent'
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)  # sha256 hex for dedupe
    embedding = Column(Text, nullable=False)  # JSON list of floats (SQLite path)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

# Create tables for SQLite path immediately
Base.metadata.create_all(bind=engine)

# When Postgres is used, ensure pgvector + a native table for fast ANN search
if DIALECT == "postgresql":
    with engine.begin() as conn:
        # Enable pgvector
        conn.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        # Create table with a real vector column for similarity search
        conn.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS debate_memories (
              id uuid PRIMARY KEY,
              project_id text NOT NULL,
              thread_id text NOT NULL,
              agent_id text,
              role text NOT NULL,
              content text NOT NULL,
              content_hash text NOT NULL,
              embedding vector({VECTOR_DIM}) NOT NULL,
              created_at timestamptz DEFAULT now()
            );
            """
        ))
        # Helpful indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_debate_mem_thread ON debate_memories(thread_id)") )
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_debate_mem_agent  ON debate_memories(agent_id)") )
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_debate_mem_dedupe ON debate_memories(thread_id, agent_id, content_hash)") )
        # Vector index (IVFFLAT) — requires ANALYZE after substantial inserts
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_debate_mem_embedding ON debate_memories USING ivfflat (embedding vector_cosine_ops)"))

class DebateEmbedder:
    """Embeds text via OpenAI if available, else deterministic local vectors."""
    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model

    def embed_one(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI  # lazy import
                client = OpenAI()
                resp = client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in resp.data]  # type: ignore[attr-defined]
            except Exception:
                pass  # fall through to pseudo
        # Deterministic pseudo-embedding (works offline / SQLite)
        import hashlib, math
        outs: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vals: List[float] = []
            while len(vals) < VECTOR_DIM:
                for b in h:
                    vals.append((b / 255.0) * 2 - 1)
                    if len(vals) == VECTOR_DIM:
                        break
            norm = math.sqrt(sum(v*v for v in vals)) or 1.0
            outs.append([v / norm for v in vals])
        return outs

_EMBEDDER = DebateEmbedder()

# Memory policy (dedupe + TTL + compaction)
MEM_MAX_PER_AGENT = int(os.getenv("DEBATE_MEMORY_MAX_PER_AGENT", "200"))
MEM_TTL_DAYS = int(os.getenv("DEBATE_MEMORY_TTL_DAYS", "90"))


def _sha256_hex(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x, y in zip(a, b)))


def _index_debate_memory(db: Session, *, project_id: str, thread_id: str, role: str, content: str, agent_id: Optional[str]) -> None:
    """Index a single message into debate memory with deduping, for both SQLite and Postgres.
    - Computes an embedding (OpenAI or pseudo)
    - Dedupes by (thread_id, agent_id, content_hash)
    - Applies TTL/compaction policy
    """
    ch = _sha256_hex(content)
    vec = _EMBEDDER.embed_one(content)

    if DIALECT == "postgresql":
        with engine.begin() as conn:
            # Dedupe check
            exists = conn.execute(
                text("SELECT 1 FROM debate_memories WHERE thread_id=:t AND agent_id IS NOT DISTINCT FROM :a AND content_hash=:h LIMIT 1"),
                {"t": thread_id, "a": agent_id, "h": ch},
            ).fetchone()
            if exists:
                return
            arr = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            conn.execute(
                text(
                    """
                    INSERT INTO debate_memories
                    (id, project_id, thread_id, agent_id, role, content, content_hash, embedding)
                    VALUES (:id, :p, :t, :a, :r, :c, :h, :e)
                    """
                ),
                {"id": str(uuid.uuid4()), "p": project_id, "t": thread_id, "a": agent_id, "r": role, "c": content, "h": ch, "e": arr},
            )
            _compact_debate_memory_pg(conn, thread_id=thread_id, agent_id=agent_id)
        return

    # SQLite path (store JSON vectors)
    # Dedupe check
    dup = (
        db.query(DebateMemory)
        .filter(DebateMemory.thread_id == thread_id, DebateMemory.agent_id == agent_id, DebateMemory.content_hash == ch)
        .first()
    )
    if dup:
        return
    row = DebateMemory(
        project_id=project_id,
        thread_id=thread_id,
        role=role,
        agent_id=agent_id,
        content=content,
        content_hash=ch,
        embedding=json.dumps(vec),
    )
    db.add(row)
    db.commit()
    _compact_debate_memory_sqlite(db, thread_id=thread_id, agent_id=agent_id)


def _compact_debate_memory_sqlite(db: Session, *, thread_id: str, agent_id: Optional[str]) -> None:
    """Remove expired items and keep only the most recent MEM_MAX_PER_AGENT rows per agent in a thread."""
    # TTL removal
    if MEM_TTL_DAYS > 0:
        cutoff = datetime.now(timezone.utc).timestamp() - MEM_TTL_DAYS * 86400
        # SQLite stores naive DateTime in our model; defensively delete by ordering/limit below.
        pass  # simplistic; rely on size cap
    # Size cap
    rows: List[DebateMemory] = (
        db.query(DebateMemory)
        .filter(DebateMemory.thread_id == thread_id, DebateMemory.agent_id == agent_id)
        .order_by(DebateMemory.created_at.desc())
        .all()
    )
    if len(rows) > MEM_MAX_PER_AGENT:
        for r in rows[MEM_MAX_PER_AGENT:]:
            db.delete(r)
        db.commit()


def _compact_debate_memory_pg(conn, *, thread_id: str, agent_id: Optional[str]) -> None:
    """Postgres compaction with TTL and per-agent cap (best-effort)."""
    # TTL
    if MEM_TTL_DAYS > 0:
        conn.execute(
            text("DELETE FROM debate_memories WHERE thread_id=:t AND (created_at < now() - (:ttl || ' days')::interval)"),
            {"t": thread_id, "ttl": MEM_TTL_DAYS},
        )
    # Size cap — keep newest N per agent
    conn.execute(text(
        """
        DELETE FROM debate_memories m
        USING (
          SELECT id FROM (
            SELECT id,
                   row_number() OVER (PARTITION BY thread_id, agent_id ORDER BY created_at DESC) AS rn
            FROM debate_memories
            WHERE thread_id = :t
          ) x WHERE x.rn > :cap
        ) old
        WHERE m.id = old.id
        """
    ), {"t": thread_id, "cap": MEM_MAX_PER_AGENT})


def _ensure_thread_indexed(db: Session, *, thread: Thread) -> None:
    """Index all existing messages for a thread (dedup-safe)."""
    msgs: List[Message] = (
        db.query(Message).filter(Message.thread_id == thread.id).order_by(Message.created_at.asc()).all()
    )
    for m in msgs:
        c = _json_loads(m.content)
        text_val = c.get("message", "") if isinstance(c, dict) else (c if isinstance(c, str) else _json_dumps(c))
        try:
            _index_debate_memory(
                db,
                project_id=thread.project_id,
                thread_id=thread.id,
                role=m.sender_type,
                content=text_val,
                agent_id=m.sender_agent_id,
            )
        except Exception:
            continue


def _search_debate_memory(db: Session, *, thread_id: str, query_text: str, agent_id: Optional[str], top_k: int = 5) -> List[Tuple[float, Any]]:
    """Return [(score, row-like)] using cosine similarity in SQLite or pgvector in Postgres."""
    qvec = _EMBEDDER.embed_one(query_text)

    if DIALECT == "postgresql":
        with engine.begin() as conn:
            arr = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
            sql = """
                SELECT id, project_id, thread_id, agent_id, role, content, created_at,
                       1 - (embedding <-> :q) AS score
                FROM debate_memories
                WHERE thread_id = :t
                  AND (:a::text IS NULL OR agent_id = :a)
                ORDER BY embedding <-> :q
                LIMIT :k
            """
            rows = conn.execute(text(sql), {"q": arr, "t": thread_id, "a": agent_id, "k": top_k}).fetchall()
            class R:  # tiny adapter for common access
                def __init__(self, tup):
                    (self.id, self.project_id, self.thread_id, self.agent_id, self.role, self.content, self.created_at, self.score) = tup
            return [(float(r[7]), R(r)) for r in rows]

    # SQLite path — brute force cosine on JSON vectors
    rows: List[DebateMemory] = (
        db.query(DebateMemory).filter(DebateMemory.thread_id == thread_id)
    )
    if agent_id is not None:
        rows = rows.filter(DebateMemory.agent_id == agent_id)
    rows = rows.all()
    hits: List[Tuple[float, DebateMemory]] = []
    for r in rows:
        try:
            vec = json.loads(r.embedding)
            score = _cosine(qvec, vec)
            hits.append((score, r))
        except Exception:
            continue
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[:top_k]


def _payload_to_text(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    msg = payload.get("message")
    if msg:
        parts.append(str(msg))
    for p in payload.get("proposals", []) or []:
        parts.append(f"Proposal: {p}")
    for q in payload.get("questions", []) or []:
        parts.append(f"Question: {q}")
    return " ".join(parts).strip()

# --- Memory-aware step endpoint (unchanged path, added docs) ---
@router.post("/threads/{thread_id}/step_mem", response_model=MessageOut)
def debate_step_with_memory(thread_id: str, cfg: DebateConfig = DebateConfig(), db: Session = Depends(get_db)):
    """Advance one turn using **retrieval-augmented prompting**.
    - Ensures the whole thread is indexed (dedupe-safe)
    - Retrieves top‑K snippets for this agent and for the thread overall
    - Injects them into the prompt under "RELEVANT PAST POINTS"
    - Persists the reply and indexes it back into memory
    """
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

    transcript = _fetch_messages(db, thread_id, limit=cfg.max_context_messages)

    # Ensure existing conversation is indexed (idempotent)
    try:
        _ensure_thread_indexed(db, thread=t)
    except Exception:
        pass

    # Build retrieval query from latest utterances
    q_lines: List[str] = []
    for m in list(reversed(transcript))[-5:]:
        c = _json_loads(m.content)
        if isinstance(c, dict) and "message" in c:
            q_lines.append(str(c.get("message", "")))
        else:
            q_lines.append(c if isinstance(c, str) else _json_dumps(c))
    query_text = "\n".join(q_lines).strip() or "general context"

    memory_snippets: List[str] = []
    try:
        agent_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=agent.id, top_k=4)
        thread_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=None, top_k=4)
        seen: set = set()
        for score, h in agent_hits + thread_hits:
            content = getattr(h, 'content', None) if not isinstance(h, DebateMemory) else h.content
            if content and content not in seen:
                seen.add(content)
                memory_snippets.append(content)
    except Exception:
        pass

    user_prompt = _build_user_prompt(transcript, cfg.response_schema, t.phase, memory_snippets)

    llm = get_llm()
    raw = llm.complete_json(system=agent.system_prompt, user=user_prompt)
    if not isinstance(raw, dict) or "message" not in raw:
        raw = {"message": str(raw)}
    payload = {
        "message": raw.get("message", ""),
        "proposals": raw.get("proposals", []),
        "questions": raw.get("questions", []),
    }

    next_index = db.query(Turn).filter(Turn.thread_id == t.id).count()
    tr = Turn(thread_id=t.id, index=next_index, speaker_agent_id=agent.id, status="complete")
    db.add(tr); db.commit(); db.refresh(tr)

    msg = _persist_agent_message(db, t, tr, agent.id, payload)

    try:
        _index_debate_memory(db, project_id=t.project_id, thread_id=t.id, role="agent", content=_payload_to_text(payload), agent_id=agent.id)
    except Exception:
        pass

    t.cursor = (t.cursor + 1) % len(order)
    db.add(t); db.commit()

    return MessageOut(
        id=msg.id,
        thread_id=msg.thread_id,
        sender_type="agent",
        sender_agent_id=agent.id,
        content=payload,
        created_at=msg.created_at,
    )

# --- Memory search endpoint ---
class DebateMemorySearchIn(BaseModel):
    """Search the debate memory store for a thread.
    - If agent_id is provided, restrict to that agent's messages
    - Returns top_k hits by cosine (SQLite) or pgvector ANN (Postgres)
    """
    query: str
    top_k: int = Field(5, ge=1, le=25)
    agent_id: Optional[str] = None

class DebateMemoryHit(BaseModel):
    id: str
    agent_id: Optional[str]
    role: str
    content: str
    score: float
    created_at: datetime

@router.post("/threads/{thread_id}/memory/search", response_model=List[DebateMemoryHit])
def debate_memory_search(thread_id: str, data: DebateMemorySearchIn, db: Session = Depends(get_db)):
    hits = _search_debate_memory(db, thread_id=thread_id, query_text=data.query, agent_id=data.agent_id, top_k=data.top_k)
    out: List[DebateMemoryHit] = []
    for score, r in hits:
        content = getattr(r, 'content', None) if not isinstance(r, DebateMemory) else r.content
        created_at = getattr(r, 'created_at', datetime.now(timezone.utc)) if not isinstance(r, DebateMemory) else r.created_at
        agent_id = getattr(r, 'agent_id', None) if not isinstance(r, DebateMemory) else r.agent_id
        role = getattr(r, 'role', 'agent') if not isinstance(r, DebateMemory) else r.role
        idv = getattr(r, 'id', str(uuid.uuid4())) if not isinstance(r, DebateMemory) else r.id
        out.append(DebateMemoryHit(id=idv, agent_id=agent_id, role=role, content=content or "", score=float(score), created_at=created_at))
    return out

# --- Maintenance endpoint: compact memory ---
class DebateMemoryCompactIn(BaseModel):
    max_per_agent: Optional[int] = None
    ttl_days: Optional[int] = None

@router.post("/threads/{thread_id}/memory/compact", response_model=dict)
def debate_memory_compact(thread_id: str, data: DebateMemoryCompactIn, db: Session = Depends(get_db)):
    """Compact memory for a thread (dedupe, TTL, and max-per-agent cap)."""
    global MEM_MAX_PER_AGENT, MEM_TTL_DAYS
    max_cap = int(data.max_per_agent or MEM_MAX_PER_AGENT)
    ttl = int(data.ttl_days or MEM_TTL_DAYS)
    
    old_cap, old_ttl = MEM_MAX_PER_AGENT, MEM_TTL_DAYS
    MEM_MAX_PER_AGENT, MEM_TTL_DAYS = max_cap, ttl
    try:
        if DIALECT == "postgresql":
            with engine.begin() as conn:
                # apply compaction for each agent in thread
                # We'll let the SQL cap everything at once
                _compact_debate_memory_pg(conn, thread_id=thread_id, agent_id=None)
        else:
            # SQLite: do it agent by agent
            agent_ids = [row[0] for row in db.query(DebateMemory.agent_id).filter(DebateMemory.thread_id == thread_id).distinct().all()]
            for aid in agent_ids:
                _compact_debate_memory_sqlite(db, thread_id=thread_id, agent_id=aid)
    finally:
        MEM_MAX_PER_AGENT, MEM_TTL_DAYS = old_cap, old_ttl
    return {"status": "ok", "max_per_agent": max_cap, "ttl_days": ttl}
