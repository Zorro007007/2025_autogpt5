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
import re
from typing import Set
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

# --- New model: reusable "type" registry for profiles (e.g., pm, frontend, backend) ---
class ProfileType(Base):
    __tablename__ = "profile_types"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, index=True, nullable=False)   # e.g., "pm"
    display_name = Column(String, nullable=False)                   # e.g., "Project Manager"
    default_system_prompt = Column(Text, nullable=False, default="")
    default_tools = Column(Text, nullable=False, default="[]")      # JSON list
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AgentProfile(Base):
    __tablename__ = "agent_profiles"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # legacy free-form; kept for continuity
    system_prompt = Column(Text, nullable=False)
    tools = Column(Text, nullable=False, default="[]")  # JSON list
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Optional reference to a typed registry row; if set, type defaults can be inherited.
    type_id = Column(String, nullable=True, index=True)  # FK soft-linked to profile_types.id (no constraint for SQLite ease)


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
    # Optional JSON context captured at thread creation (goals, constraints, assumptions, etc.)
    context = Column(Text, nullable=True)  # JSON string
    version = Column(Integer, nullable=False, default=0)

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

class ParticipantAddIn(BaseModel):
    participant_id: Optional[str] = None
    type_key: Optional[str] = None
    position: Optional[int] = Field(None, ge=0)


Index("ix_messages_thread_created", Message.thread_id, Message.created_at)




def _ensure_column(table: str, column: str, coltype: str) -> None:
    """Best-effort: add a column if missing (SQLite and Postgres)."""
    try:
        with engine.begin() as conn:
            if DIALECT == "sqlite":
                cols = conn.execute(text(f'PRAGMA table_info("{table}")')).fetchall()
                if any(row[1] == column for row in cols):
                    return
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {column} {coltype}'))
            else:
                # Postgres
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {coltype}'))
    except Exception:
        # Non-fatal on dev; if it already exists or ALTER fails, we carry on.
        pass

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# Add new columns if missing (dev convenience)
_ensure_column("agent_profiles", "type_id", "TEXT")
_ensure_column("threads", "context", "TEXT")
_ensure_column("threads", "version", "INTEGER DEFAULT 0")
_ensure_column("profile_types", "default_response_schema", "TEXT")
_ensure_column("debate_memories", "kind", "TEXT")


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

# --- ProfileType I/O ---
class CreateProfileTypeIn(BaseModel):
    key: str
    display_name: str
    default_system_prompt: str = ""
    default_tools: List[dict] = Field(default_factory=list)

class UpdateProfileTypeIn(BaseModel):
    display_name: Optional[str] = None
    default_system_prompt: Optional[str] = None
    default_tools: Optional[List[dict]] = None

class ProfileTypeOut(BaseModel):
    id: str
    key: str
    display_name: str

# --- Profile CRUD I/O (ad-hoc) ---
class CreateProfileIn(BaseModel):
    name: str
    # Either give a type_key or a type_id; optional role kept for compatibility
    type_key: Optional[str] = None
    type_id: Optional[str] = None
    role: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[dict]] = None

class UpdateProfileIn(BaseModel):
    name: Optional[str] = None
    type_id: Optional[str] = None
    role: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[dict]] = None

# --- Thread creation (ergonomic) ---
class NewThreadIn(BaseModel):
    phase: str
    title: str
    participant_type_keys: Optional[List[str]] = None
    participant_ids: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None   # goals, constraints, assumptions...
    initial_message: Optional[str] = None


class SeedOut(BaseModel):
    inserted: int

class ProfileOut(BaseModel):
    id: str
    name: str
    role: str

class CursorUpdateIn(BaseModel):
    next_speaker_id: Optional[str] = None
    index: Optional[int] = Field(None, ge=0)


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
    context: Optional[Dict[str, Any]] = None

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

BASE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "proposals": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["message"],
    "additionalProperties": False,
}

def _merge_schema(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not extra:
        return base
    merged = json.loads(json.dumps(base))
    props = merged.setdefault("properties", {})
    base_required = set(merged.get("required", []))
    xprops = extra.get("properties", {})
    xrequired = set(extra.get("required", []))
    props.update(xprops)
    merged["required"] = list(base_required.union(xrequired))
    # keep additionalProperties strict if either is strict
    merged["additionalProperties"] = bool(base.get("additionalProperties", False) or extra.get("additionalProperties", False))
    return merged

def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_claim(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _collect_recent_claims(db: Session, thread_id: str, window_msgs: int = 60) -> Tuple[Set[str], Set[str]]:
    """Return (seen_proposals, seen_questions) from recent agent messages."""
    seen_p: Set[str] = set()
    seen_q: Set[str] = set()
    msgs: List[Message] = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.sender_type == "agent")
        .order_by(Message.created_at.desc())
        .limit(window_msgs)
        .all()
    )
    for m in msgs:
        content = _json_loads(m.content)
        if not isinstance(content, dict): 
            continue
        for p in (content.get("proposals") or []):
            if isinstance(p, str): seen_p.add(_norm_claim(p))
        for q in (content.get("questions") or []):
            if isinstance(q, str): seen_q.add(_norm_claim(q))
    return seen_p, seen_q

def _dedupe_payload_against_recent(db: Session, thread_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    seen_p, seen_q = _collect_recent_claims(db, thread_id)
    proposals = [p for p in (payload.get("proposals") or []) if isinstance(p, str) and _norm_claim(p) not in seen_p]
    questions = [q for q in (payload.get("questions") or []) if isinstance(q, str) and _norm_claim(q) not in seen_q]
    out = dict(payload)
    out["proposals"] = proposals
    out["questions"] = questions
    return out


def _thread_version(db: Session, thread_id: str) -> int:
    row = db.query(Thread).filter(Thread.id == thread_id).first()
    return int(getattr(row, "version", 0) or 0)

def _claim_turn_or_conflict(db: Session, thread: Thread) -> None:
    """Optimistically 'claim' a turn by bumping version atomically.
    If another client just claimed, raise 409 so caller retries."""
    current_version = int(getattr(thread, "version", 0) or 0)
    with engine.begin() as conn:
        rc = conn.execute(
            text('UPDATE threads SET version=COALESCE(version,0)+1 WHERE id=:id AND COALESCE(version,0)=:v'),
            {"id": thread.id, "v": current_version}
        ).rowcount
    if rc == 0:
        raise HTTPException(409, "Turn conflict: try again")


def _sha256_hex(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_user_prompt(
    transcript: List[Message],
    schema: Dict[str, Any],
    phase: str,
    memory_snippets: Optional[List[str]] = None,
    thread_context: Optional[Dict[str, Any]] = None,
) -> str:
    """User prompt for an agent turn.
    Includes recent transcript, optional retrieved memory, and optional thread context.
    """
    # 1) Thread context (brief; keeps LLM on-rails)
    ctx_block = ""
    if thread_context:
        try:
            # Compact representation (max ~1K chars)
            ctx_json = _json_dumps(thread_context)
            ctx_block = f"PROJECT CONTEXT:\n{ctx_json[:1000]}\n\n"
        except Exception:
            pass

    # 2) Transcript (recent → last)
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

    # 3) Retrieved memory (RAG)
    mem_block = ""
    if memory_snippets:
        mem_txt = "\n".join(f"- {s}" for s in memory_snippets[:8])
        mem_block = f"RELEVANT PAST POINTS (retrieved):\n{mem_txt}\n\n"

    # 4) Final assembly
    return (
        f"You are participating in a structured, turn-based product discussion for phase: {phase}.\n"
        "Respond with a short, actionable contribution.\n\n"
        + ctx_block
        + "CONVERSATION (most recent last):\n" + transcript_txt + "\n\n"
        + mem_block +
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON for the given schema.\n"
        "- Keep 'message' concise (<= 240 words).\n"
        "- If you have specific suggestions or questions, use 'proposals' and 'questions'.\n\n"
        "- Avoid repeating proposals/questions already raised in this thread; add only novel or refined items.\n"
        "- If you have nothing new, keep 'proposals' and 'questions' empty.\n"
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

def _compose_system_prompt(profile: AgentProfile, type_row: Optional[ProfileType], org_policy: str = "") -> str:
    """Build final system prompt from type defaults + profile override + optional org policy."""
    parts: List[str] = []
    if type_row and type_row.default_system_prompt:
        parts.append(type_row.default_system_prompt.strip())
    if profile.system_prompt:
        parts.append(profile.system_prompt.strip())
    if org_policy:
        parts.append(org_policy.strip())
    return "\n\n".join([p for p in parts if p])

def _get_profile_type(db: Session, profile: AgentProfile) -> Optional[ProfileType]:
    if not profile.type_id:
        return None
    row = db.query(ProfileType).filter(ProfileType.id == profile.type_id).first()
    return row


# =====================
# Router
# =====================
router = APIRouter()

@router.post("/profiles/seed", response_model=SeedOut)
def seed_profiles(db: Session = Depends(get_db)):
    # Per-type extras added on top of BASE_JSON_SCHEMA (all roles still return message/proposals/questions)
    type_schemas = {
        "pm": {
            "properties": {
                "risks": {"type": "array", "items": {"type": "string"}},
                "kpis": {"type": "array", "items": {"type": "string"}},
            }
        },
        "frontend": {
            "properties": {
                "components": {"type": "array", "items": {"type": "string"}},
                "edge_cases": {"type": "array", "items": {"type": "string"}},
            }
        },
        "backend": {
            "properties": {
                "apis": {"type": "array", "items": {"type": "string"}},
                "data_models": {"type": "array", "items": {"type": "string"}},
                "failure_modes": {"type": "array", "items": {"type": "string"}},
            }
        },
        "architect": {
            "properties": {
                "nfrs": {"type": "array", "items": {"type": "string"}},
                "threats": {"type": "array", "items": {"type": "string"}},
            }
        },
        "qa": {
            "properties": {
                "tests": {"type": "array", "items": {"type": "string"}},
            }
        },
        "critic": {
            "properties": {
                "critiques": {"type": "array", "items": {"type": "string"}},
            }
        },
    }
    defaults = {
        "pm":        ("Project Manager", "You are a pragmatic PM...", []),
        "frontend":  ("Frontend Developer", "You are a senior Frontend Engineer...", []),
        "backend":   ("Backend Developer", "You are a senior Backend Engineer...", []),
        "architect": ("Architect", "You are a Solution Architect...", []),
        "qa":        ("QA Engineer", "You are a QA Engineer...", []),
        "critic":    ("Critic", "You are a rigorous critic...", []),
    }

    type_map = {}
    for key, (display, prompt, tools) in defaults.items():
        row = db.query(ProfileType).filter(ProfileType.key == key).first()
        merged_schema = _merge_schema(BASE_JSON_SCHEMA, type_schemas.get(key))
        if not row:
            row = ProfileType(
                key=key,
                display_name=display,
                default_system_prompt=prompt,
                default_tools=_json_dumps(tools),
                # NEW: store default_response_schema
                **{"default_response_schema": _json_dumps(merged_schema)}
            )
            db.add(row); db.commit(); db.refresh(row)
        else:
            # backfill schema if missing
            try:
                if not getattr(row, "default_response_schema", None):
                    row.default_response_schema = _json_dumps(merged_schema)
                    db.add(row); db.commit(); db.refresh(row)
            except Exception:
                pass
        type_map[key] = row.id

    existing = {p.name for p in db.query(AgentProfile).all()}
    inserted = 0
    for key, (display, prompt, tools) in defaults.items():
        name = display
        if name in existing:
            continue
        prof = AgentProfile(
            name=name, role=key, system_prompt=prompt,
            tools=_json_dumps(tools), type_id=type_map.get(key),
        )
        db.add(prof); inserted += 1
    db.commit()
    return SeedOut(inserted=inserted)

@router.post("/threads/{thread_id}/participants", response_model=ThreadOut)
def add_participant(thread_id: str, data: ParticipantAddIn, db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")

    order: List[str] = json.loads(t.participants) or []
    new_id: Optional[str] = None

    if data.participant_id:
        prof = db.get(AgentProfile, data.participant_id)
        if not prof:
            raise HTTPException(404, "participant_id not found")
        new_id = prof.id

    elif data.type_key:
        # pick first profile of that type (or role fallback)
        type_row = db.query(ProfileType).filter(ProfileType.key == data.type_key).first()
        q = db.query(AgentProfile)
        if type_row:
            q = q.filter(AgentProfile.type_id == type_row.id)
        else:
            q = q.filter(AgentProfile.role == data.type_key)
        prof = q.order_by(AgentProfile.created_at.asc()).first()
        if not prof:
            raise HTTPException(404, f"No profile available for type '{data.type_key}'")
        new_id = prof.id
    else:
        raise HTTPException(400, "Provide participant_id or type_key")

    if new_id in order:
        # already present – no-op
        pos = data.position if data.position is not None else len(order)
        t.participants = _json_dumps(order)  # unchanged
    else:
        pos = data.position if data.position is not None else len(order)
        pos = min(max(0, pos), len(order))
        order.insert(pos, new_id)
        t.participants = _json_dumps(order)

        # Adjust cursor if insertion comes before the current cursor
        if pos <= (t.cursor % max(1, len(order))):
            t.cursor = (t.cursor + 1) % len(order)

    db.add(t); db.commit(); db.refresh(t)
    return ThreadOut(
        id=t.id, project_id=t.project_id, phase=t.phase, title=t.title,
        participants=order, cursor=t.cursor, status=t.status, created_at=t.created_at,
        context=_json_loads(t.context) if t.context else None,
    )


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
        context=_json_loads(t.context) if t.context else None,
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

# List threads (optionally filter by project_id)
@router.get("/threads", response_model=List[ThreadOut])
def list_threads(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Thread)
    if project_id:
        q = q.filter(Thread.project_id == project_id)
    rows = q.order_by(Thread.created_at.desc()).limit(200).all()
    out = []
    for t in rows:
        out.append(ThreadOut(
            id=t.id, project_id=t.project_id, phase=t.phase, title=t.title,
            participants=json.loads(t.participants), cursor=t.cursor,
            status=t.status, created_at=t.created_at,
            context=_json_loads(t.context) if t.context else None,
        ))
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
    if not t: raise HTTPException(404, "Thread not found")
    _claim_turn_or_conflict(db, t)  # NEW: optimistic claim

    order: List[str] = json.loads(t.participants)
    if not order: raise HTTPException(400, "Thread has no participants")
    speaker_id = order[t.cursor % len(order)]
    agent: Optional[AgentProfile] = db.get(AgentProfile, speaker_id)
    if not agent: raise HTTPException(404, "Next speaker AgentProfile not found")

    transcript = _fetch_messages(db, thread_id, limit=cfg.max_context_messages)
    type_row = _get_profile_type(db, agent)
    # NEW: per-role schema
    role_schema = None
    try:
        role_schema = _json_loads(getattr(type_row, "default_response_schema", "") or "") if type_row else None
    except Exception:
        pass
    schema = _merge_schema(BASE_JSON_SCHEMA, role_schema)

    user_prompt = _build_user_prompt(transcript, schema, t.phase, None, _json_loads(t.context) if t.context else None)
    llm = get_llm()
    system_prompt = _compose_system_prompt(agent, type_row)
    raw = llm.complete_json(system=system_prompt, user=user_prompt)
    if not isinstance(raw, dict) or "message" not in raw:
        raw = {"message": str(raw)}
    payload = {
        "message": raw.get("message", ""),
        "proposals": raw.get("proposals", []),
        "questions": raw.get("questions", []),
    }

    payload = _dedupe_payload_against_recent(db, t.id, payload)

    next_index = db.query(Turn).filter(Turn.thread_id == t.id).count()
    tr = Turn(thread_id=t.id, index=next_index, speaker_agent_id=agent.id, status="complete")
    db.add(tr); db.commit(); db.refresh(tr)
    msg = _persist_agent_message(db, t, tr, agent.id, payload)

    # NEW: claim-level memory
    try: _index_agent_payload_claims(db, thread=t, agent_id=agent.id, payload=payload)
    except Exception: pass

    # advance cursor
    t.cursor = (t.cursor + 1) % len(order)
    db.add(t); db.commit()

    return MessageOut(id=msg.id, thread_id=msg.thread_id, sender_type="agent", sender_agent_id=agent.id, content=payload, created_at=msg.created_at)


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

# ----------------------
# ProfileType CRUD
# ----------------------
@router.post("/profile-types", response_model=ProfileTypeOut)
def create_profile_type(data: CreateProfileTypeIn, db: Session = Depends(get_db)):
    exists = db.query(ProfileType).filter(ProfileType.key == data.key).first()
    if exists:
        raise HTTPException(400, f"profile_type '{data.key}' already exists")
    row = ProfileType(
        key=data.key,
        display_name=data.display_name,
        default_system_prompt=data.default_system_prompt,
        default_tools=_json_dumps(data.default_tools),
    )
    db.add(row); db.commit(); db.refresh(row)
    return ProfileTypeOut(id=row.id, key=row.key, display_name=row.display_name)

@router.get("/profile-types", response_model=List[ProfileTypeOut])
def list_profile_types(db: Session = Depends(get_db)):
    rows = db.query(ProfileType).order_by(ProfileType.key.asc()).all()
    return [ProfileTypeOut(id=r.id, key=r.key, display_name=r.display_name) for r in rows]

@router.patch("/profile-types/{pt_id}", response_model=ProfileTypeOut)
def update_profile_type(pt_id: str, data: UpdateProfileTypeIn, db: Session = Depends(get_db)):
    row: Optional[ProfileType] = db.get(ProfileType, pt_id)
    if not row:
        raise HTTPException(404, "profile_type not found")
    if data.display_name is not None:
        row.display_name = data.display_name
    if data.default_system_prompt is not None:
        row.default_system_prompt = data.default_system_prompt
    if data.default_tools is not None:
        row.default_tools = _json_dumps(data.default_tools)
    db.add(row); db.commit(); db.refresh(row)
    return ProfileTypeOut(id=row.id, key=row.key, display_name=row.display_name)

@router.delete("/profile-types/{pt_id}", response_model=dict)
def delete_profile_type(pt_id: str, db: Session = Depends(get_db)):
    row: Optional[ProfileType] = db.get(ProfileType, pt_id)
    if not row:
        return {"status": "ok"}  # idempotent
    # Basic guard: prevent deletion if still referenced
    in_use = db.query(AgentProfile).filter(AgentProfile.type_id == row.id).count()
    if in_use:
        raise HTTPException(400, "profile_type is still referenced by one or more profiles")
    db.delete(row); db.commit()
    return {"status": "ok"}

@router.patch("/threads/{thread_id}/cursor", response_model=ThreadOut)
def set_next_speaker(thread_id: str, data: CursorUpdateIn, db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    order: List[str] = json.loads(t.participants) or []
    if not order:
        raise HTTPException(400, "Thread has no participants")

    if data.index is None and not data.next_speaker_id:
        raise HTTPException(400, "Provide index or next_speaker_id")

    if data.index is not None:
        if data.index >= len(order):
            raise HTTPException(400, "index out of range")
        t.cursor = int(data.index) % len(order)
    else:
        try:
            t.cursor = order.index(data.next_speaker_id)
        except ValueError:
            raise HTTPException(404, "next_speaker_id is not in participants")

    db.add(t); db.commit(); db.refresh(t)
    return ThreadOut(
        id=t.id, project_id=t.project_id, phase=t.phase, title=t.title,
        participants=order, cursor=t.cursor, status=t.status, created_at=t.created_at,
        context=_json_loads(t.context) if t.context else None,
    )



# ----------------------
# Profile CRUD (ad-hoc)
# ----------------------
@router.post("/profiles", response_model=ProfileOut)
def create_profile(data: CreateProfileIn, db: Session = Depends(get_db)):
    type_row = None
    type_id = data.type_id
    if not type_id and data.type_key:
        type_row = db.query(ProfileType).filter(ProfileType.key == data.type_key).first()
        if not type_row:
            raise HTTPException(404, f"profile_type '{data.type_key}' not found")
        type_id = type_row.id
    elif type_id:
        type_row = db.get(ProfileType, type_id)

    # Inherit defaults from type unless explicitly provided
    system_prompt = data.system_prompt if data.system_prompt is not None else (type_row.default_system_prompt if type_row else "")
    tools = data.tools if data.tools is not None else (_json_loads(type_row.default_tools) if type_row else [])

    row = AgentProfile(
        name=data.name,
        role=data.role or (type_row.key if type_row else "custom"),
        system_prompt=system_prompt,
        tools=_json_dumps(tools),
        type_id=type_id,
    )
    db.add(row); db.commit(); db.refresh(row)
    return ProfileOut(id=row.id, name=row.name, role=row.role)

@router.patch("/profiles/{profile_id}", response_model=ProfileOut)
def update_profile(profile_id: str, data: UpdateProfileIn, db: Session = Depends(get_db)):
    row: Optional[AgentProfile] = db.get(AgentProfile, profile_id)
    if not row:
        raise HTTPException(404, "profile not found")
    if data.name is not None:
        row.name = data.name
    if data.role is not None:
        row.role = data.role
    if data.system_prompt is not None:
        row.system_prompt = data.system_prompt
    if data.tools is not None:
        row.tools = _json_dumps(data.tools)
    if data.type_id is not None:
        row.type_id = data.type_id
    db.add(row); db.commit(); db.refresh(row)
    return ProfileOut(id=row.id, name=row.name, role=row.role)


# ----------------------
# Ergonomic thread creation
# ----------------------
@router.post("/projects/{project_id}/threads/new", response_model=ThreadOut)
def create_thread_new(project_id: str, data: NewThreadIn, db: Session = Depends(get_db)):
    # Resolve participants by IDs or by type keys
    participants: List[str] = []
    if data.participant_ids:
        participants = list(dict.fromkeys(data.participant_ids))  # dedupe preserve order
        rows = db.query(AgentProfile).filter(AgentProfile.id.in_(participants)).all()
        if len(rows) != len(participants):
            raise HTTPException(404, "One or more participant IDs not found")
    elif data.participant_type_keys:
        # Pick *one* profile per type key (first match by created_at)
        for key in data.participant_type_keys:
            # find any profile with type.key == key OR fallback to role == key
            type_row = db.query(ProfileType).filter(ProfileType.key == key).first()
            q = db.query(AgentProfile)
            if type_row:
                q = q.filter(AgentProfile.type_id == type_row.id)
            else:
                q = q.filter(AgentProfile.role == key)
            prof = q.order_by(AgentProfile.created_at.asc()).first()
            if not prof:
                raise HTTPException(404, f"No profile available for type '{key}'")
            participants.append(prof.id)
    else:
        raise HTTPException(400, "Provide participant_ids or participant_type_keys")

    row = Thread(
        project_id=project_id,
        phase=data.phase,
        title=data.title,
        participants=_json_dumps(participants),
        cursor=0,
        context=_json_dumps(data.context) if data.context else None,
    )
    db.add(row); db.commit(); db.refresh(row)

    # Optional: seed an initial user message
    if data.initial_message:
        tr = Turn(thread_id=row.id, index=0, speaker_agent_id=None, status="complete")
        db.add(tr); db.commit(); db.refresh(tr)
        um = Message(
            thread_id=row.id, turn_id=tr.id, sender_type="user",
            sender_agent_id=None, content=data.initial_message, content_type="text",
        )
        db.add(um); db.commit()

    # Index context into vector memory (as 'seed') to improve early retrieval
    if data.context:
        try:
            _index_debate_memory(
                db,
                project_id=project_id,
                thread_id=row.id,
                role="seed",
                content=f"Thread context: {_json_dumps(data.context)}",
                agent_id=None,
            )
        except Exception:
            pass

    return ThreadOut(
        id=row.id, project_id=row.project_id, phase=row.phase, title=row.title,
        participants=participants, cursor=row.cursor, status=row.status, created_at=row.created_at,
        context=_json_loads(row.context) if row.context else None,
    )


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
    role = Column(String, nullable=False)  # 'user'|'agent'|'seed'
    kind = Column(String, nullable=True)   # 'message'|'proposal'|'question'|...
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)
    embedding = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

# --- NEW: Decisions log (moderator writes deduped proposals) ---
class Decision(Base):
    __tablename__ = "decisions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
    text_hash = Column(String, index=True, nullable=False)  # normalized sha256 for dedupe
    status = Column(String, nullable=False, default="proposed")  # 'proposed'|'accepted'|'rejected'
    by_agent_id = Column(String, nullable=True)
    turn_id = Column(String, nullable=True)
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

if DIALECT == "postgresql":
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE debate_memories ADD COLUMN IF NOT EXISTS kind text"))


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


def _index_debate_memory(db: Session, *, project_id: str, thread_id: str, role: str, content: str, agent_id: Optional[str], kind: str = "message") -> None:
    ch = _sha256_hex(f"{kind}:{content}")
    vec = _EMBEDDER.embed_one(content)

    if DIALECT == "postgresql":
        with engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM debate_memories WHERE thread_id=:t AND agent_id IS NOT DISTINCT FROM :a AND content_hash=:h LIMIT 1"),
                {"t": thread_id, "a": agent_id, "h": ch},
            ).fetchone()
            if exists:
                return
            arr = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            conn.execute(text("""
                INSERT INTO debate_memories
                (id, project_id, thread_id, agent_id, role, kind, content, content_hash, embedding)
                VALUES (:id, :p, :t, :a, :r, :k, :c, :h, :e)
            """), {"id": str(uuid.uuid4()), "p": project_id, "t": thread_id, "a": agent_id, "r": role, "k": kind, "c": content, "h": ch, "e": arr})
            _compact_debate_memory_pg(conn, thread_id=thread_id, agent_id=agent_id)
        return

    dup = (
        db.query(DebateMemory)
        .filter(DebateMemory.thread_id == thread_id, DebateMemory.agent_id == agent_id, DebateMemory.content_hash == ch)
        .first()
    )
    if dup:
        return
    row = DebateMemory(
        project_id=project_id, thread_id=thread_id, role=role, agent_id=agent_id,
        kind=kind, content=content, content_hash=ch, embedding=json.dumps(vec),
    )
    db.add(row); db.commit()
    _compact_debate_memory_sqlite(db, thread_id=thread_id, agent_id=agent_id)

def _index_agent_payload_claims(db: Session, *, thread: Thread, agent_id: str, payload: Dict[str, Any]) -> None:
    # message
    msg = str(payload.get("message", "")).strip()
    if msg:
        _index_debate_memory(db, project_id=thread.project_id, thread_id=thread.id, role="agent", content=msg, agent_id=agent_id, kind="message")
    # proposals
    for p in (payload.get("proposals") or []):
        p = str(p).strip()
        if p:
            _index_debate_memory(db, project_id=thread.project_id, thread_id=thread.id, role="agent", content=p, agent_id=agent_id, kind="proposal")
    # questions
    for q in (payload.get("questions") or []):
        q = str(q).strip()
        if q:
            _index_debate_memory(db, project_id=thread.project_id, thread_id=thread.id, role="agent", content=q, agent_id=agent_id, kind="question")

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


def _search_debate_memory(db: Session, *, thread_id: str, query_text: str, agent_id: Optional[str], top_k: int = 5, kinds: Optional[List[str]] = None) -> List[Tuple[float, Any]]:
    qvec = _EMBEDDER.embed_one(query_text)

    if DIALECT == "postgresql":
        with engine.begin() as conn:
            arr = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
            cond_kind = "TRUE" if not kinds else "kind = ANY(:kinds)"
            sql = f"""
                SELECT id, project_id, thread_id, agent_id, role, content, created_at, 1 - (embedding <-> :q) AS score
                FROM debate_memories
                WHERE thread_id = :t
                  AND (:a::text IS NULL OR agent_id = :a)
                  AND {cond_kind}
                ORDER BY embedding <-> :q
                LIMIT :k
            """
            rows = conn.execute(text(sql), {"q": arr, "t": thread_id, "a": agent_id, "k": top_k, "kinds": kinds}).fetchall()
            class R:
                def __init__(self, tup): (self.id, self.project_id, self.thread_id, self.agent_id, self.role, self.content, self.created_at, self.score) = tup
            return [(float(r[7]), R(r)) for r in rows]

    rows = db.query(DebateMemory).filter(DebateMemory.thread_id == thread_id)
    if agent_id is not None:
        rows = rows.filter(DebateMemory.agent_id == agent_id)
    if kinds:
        rows = rows.filter(DebateMemory.kind.in_(kinds))
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
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t: raise HTTPException(404, "Thread not found")
    _claim_turn_or_conflict(db, t)  # NEW

    order: List[str] = json.loads(t.participants)
    if not order: raise HTTPException(400, "Thread has no participants")
    speaker_id = order[t.cursor % len(order)]
    agent: Optional[AgentProfile] = db.get(AgentProfile, speaker_id)
    if not agent: raise HTTPException(404, "Next speaker AgentProfile not found")

    transcript = _fetch_messages(db, thread_id, limit=cfg.max_context_messages)

    # ensure existing conversation is indexed (best-effort)
    try: _ensure_thread_indexed(db, thread=t)
    except Exception: pass

    # retrieval query from latest utterances
    q_lines: List[str] = []
    for m in list(reversed(transcript))[-5:]:
        c = _json_loads(m.content)
        q_lines.append(c.get("message","") if isinstance(c, dict) else (c if isinstance(c, str) else _json_dumps(c)))
    query_text = "\n".join([x for x in q_lines if x]).strip() or "general context"

    # role schema
    type_row = _get_profile_type(db, agent)
    role_schema = None
    try:
        role_schema = _json_loads(getattr(type_row, "default_response_schema", "") or "") if type_row else None
    except Exception:
        pass
    schema = _merge_schema(BASE_JSON_SCHEMA, role_schema)

    # claim-focused retrieval
    memory_snippets: List[str] = []
    try:
        agent_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=agent.id, top_k=4, kinds=["proposal","question","message"])
        thread_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=None, top_k=4, kinds=["proposal","question","message"])
        seen: set = set()
        for score, h in agent_hits + thread_hits:
            content = getattr(h, 'content', None) if not isinstance(h, DebateMemory) else h.content
            if content and content not in seen:
                seen.add(content); memory_snippets.append(content)
    except Exception:
        pass

    user_prompt = _build_user_prompt(transcript, schema, t.phase, memory_snippets, _json_loads(t.context) if t.context else None)
    llm = get_llm()
    system_prompt = _compose_system_prompt(agent, type_row)
    raw = llm.complete_json(system=system_prompt, user=user_prompt)
    if not isinstance(raw, dict) or "message" not in raw:
        raw = {"message": str(raw)}
    payload = {
        "message": raw.get("message", ""),
        "proposals": raw.get("proposals", []),
        "questions": raw.get("questions", []),
    }

    payload = _dedupe_payload_against_recent(db, t.id, payload)
    next_index = db.query(Turn).filter(Turn.thread_id == t.id).count()
    tr = Turn(thread_id=t.id, index=next_index, speaker_agent_id=agent.id, status="complete")
    db.add(tr); db.commit(); db.refresh(tr)
    msg = _persist_agent_message(db, t, tr, agent.id, payload)

    # index claims
    try: _index_agent_payload_claims(db, thread=t, agent_id=agent.id, payload=payload)
    except Exception: pass

    # --- NEW: moderator pass at end of round (dedupe proposals into decisions) ---
    try:
        if (tr.index + 1) % max(1, len(order)) == 0:
            _moderator_round_sweep(db, thread=t, round_len=len(order))
    except Exception:
        pass

    # advance cursor
    t.cursor = (t.cursor + 1) % len(order)
    db.add(t); db.commit()

    return MessageOut(id=msg.id, thread_id=msg.thread_id, sender_type="agent", sender_agent_id=agent.id, content=payload, created_at=msg.created_at)

class DecisionOut(BaseModel):
    id: str
    thread_id: str
    text: str
    status: str
    by_agent_id: Optional[str]
    turn_id: Optional[str]
    created_at: datetime

class DecisionUpdateIn(BaseModel):
    status: Optional[str] = Field(None, pattern="^(proposed|accepted|rejected)$")
    by_agent_id: Optional[str] = None

@router.get("/threads/{thread_id}/decisions", response_model=List[DecisionOut])
def list_decisions(thread_id: str, db: Session = Depends(get_db)):
    rows: List[Decision] = (
        db.query(Decision).filter(Decision.thread_id == thread_id).order_by(Decision.created_at.desc()).all()
    )
    return [DecisionOut(
        id=r.id, thread_id=r.thread_id, text=r.text, status=r.status,
        by_agent_id=r.by_agent_id, turn_id=r.turn_id, created_at=r.created_at
    ) for r in rows]

@router.patch("/decisions/{decision_id}", response_model=DecisionOut)
def update_decision(decision_id: str, data: DecisionUpdateIn, db: Session = Depends(get_db)):
    r: Optional[Decision] = db.get(Decision, decision_id)
    if not r:
        raise HTTPException(404, "decision not found")
    if data.status is not None:
        r.status = data.status
    if data.by_agent_id is not None:
        r.by_agent_id = data.by_agent_id
    db.add(r); db.commit(); db.refresh(r)
    return DecisionOut(
        id=r.id, thread_id=r.thread_id, text=r.text, status=r.status,
        by_agent_id=r.by_agent_id, turn_id=r.turn_id, created_at=r.created_at
    )


def _moderator_round_sweep(db: Session, *, thread: Thread, round_len: int) -> None:
    """Collect proposals from last full round, dedupe, and write to decisions table."""
    msgs: List[Message] = (
        db.query(Message)
        .filter(Message.thread_id == thread.id, Message.sender_type == "agent")
        .order_by(Message.created_at.desc())
        .limit(round_len)
        .all()
    )
    seen_hashes: Set[str] = set()
    for m in msgs:
        payload = _json_loads(m.content)
        if not isinstance(payload, dict):
            continue
        for p in (payload.get("proposals") or []):
            norm = _norm_text(str(p))
            if not norm:
                continue
            h = _sha256_hex(norm)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # upsert-ish
            exists = db.query(Decision).filter(Decision.thread_id == thread.id, Decision.text_hash == h).first()
            if exists:
                continue
            d = Decision(
                thread_id=thread.id, text=str(p), text_hash=h,
                by_agent_id=m.sender_agent_id, turn_id=m.turn_id, status="proposed",
            )
            db.add(d)
    db.commit()

def _is_stalled_low_novelty(db: Session, thread_id: str, window: int = 6, similarity: float = 0.9) -> bool:
    """Very lightweight novelty check: if last `window` agent messages are too similar, consider stalled."""
    msgs: List[Message] = (
        db.query(Message)
        .filter(Message.thread_id == thread_id, Message.sender_type == "agent")
        .order_by(Message.created_at.desc())
        .limit(window)
        .all()
    )
    if len(msgs) < window:
        return False

    def bag(text: str) -> Set[str]:
        toks = re.findall(r"[a-z0-9]{3,}", text.lower())
        return set(toks)

    bags: List[Set[str]] = []
    for m in msgs:
        c = _json_loads(m.content)
        txt = c.get("message", "") if isinstance(c, dict) else (c if isinstance(c, str) else _json_dumps(c))
        bags.append(bag(txt))

    # avg Jaccard similarity across consecutive messages
    sims = []
    for i in range(len(bags) - 1):
        a, b = bags[i], bags[i+1]
        if not a or not b:
            continue
        sims.append(len(a & b) / max(1, len(a | b)))
    if not sims:
        return False
    avg = sum(sims) / len(sims)
    return avg >= similarity


@router.post("/threads/{thread_id}/autorun", response_model=List[MessageOut])
def debate_autorun(thread_id: str, n: int = 3, use_memory: bool = True, db: Session = Depends(get_db)):
    out: List[MessageOut] = []
    for _ in range(max(0, n)):
        if _is_stalled_low_novelty(db, thread_id):
            raise HTTPException(409, "autorun stalled: low novelty (try steering with a user message or adjust context)")
        out.append(debate_step_with_memory(thread_id, db=db) if use_memory else debate_step(thread_id, db=db))
    return out


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

class StepPreviewOut(BaseModel):
    speaker_agent_id: str
    schema: Dict[str, Any]
    system_prompt: str
    user_prompt: str
    memory_snippets: List[str] = []

@router.get("/threads/{thread_id}/step_preview", response_model=StepPreviewOut)
def step_preview(thread_id: str, db: Session = Depends(get_db)):
    t: Optional[Thread] = db.get(Thread, thread_id)
    if not t:
        raise HTTPException(404, "Thread not found")
    order = json.loads(t.participants) or []
    if not order:
        raise HTTPException(400, "Thread has no participants")
    speaker_id = order[t.cursor % len(order)]
    agent: Optional[AgentProfile] = db.get(AgentProfile, speaker_id)
    if not agent:
        raise HTTPException(404, "Next speaker AgentProfile not found")

    # role schema
    type_row = _get_profile_type(db, agent)
    try:
        role_schema = _json_loads(getattr(type_row, "default_response_schema", "") or "") if type_row else None
    except Exception:
        role_schema = None
    schema = _merge_schema(BASE_JSON_SCHEMA, role_schema)

    # retrieval (claim-focused)
    transcript = _fetch_messages(db, thread_id, limit=12)
    q_lines = []
    for m in list(reversed(transcript))[-5:]:
        c = _json_loads(m.content)
        q_lines.append(c.get("message", "") if isinstance(c, dict) else (c if isinstance(c, str) else _json_dumps(c)))
    query_text = "\n".join([x for x in q_lines if x]).strip() or "general context"
    agent_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=agent.id, top_k=4, kinds=["proposal","question","message"])
    thread_hits = _search_debate_memory(db, thread_id=t.id, query_text=query_text, agent_id=None, top_k=4, kinds=["proposal","question","message"])
    memory_snippets = []
    seen = set()
    for score, h in agent_hits + thread_hits:
        content = getattr(h, 'content', None) if not isinstance(h, DebateMemory) else h.content
        if content and content not in seen:
            seen.add(content); memory_snippets.append(content)

    system_prompt = _compose_system_prompt(agent, type_row)
    user_prompt = _build_user_prompt(transcript, schema, t.phase, memory_snippets, _json_loads(t.context) if t.context else None)
    return StepPreviewOut(speaker_agent_id=agent.id, schema=schema, system_prompt=system_prompt, user_prompt=user_prompt, memory_snippets=memory_snippets)
