"""
FastAPI API for Phase 1 & 2 — Explore → Requirements (with pgvector memory)
--------------------------------------------------------------------------

This service now supports:
  • Phase‑1 (Exploration): BA → Critic → BA revision
  • Phase‑2 (Requirements): BA → QA → Critic → BA revision
  • Artifact persistence (SQLite by default; Postgres via DATABASE_URL)
  • Vector memory with similarity search (Postgres + pgvector) and
    a SQLite fallback (JSON vectors + Python cosine similarity)

Run locally (SQLite + Mock LLM):
  uvicorn api_service:app --reload --port 8000

Use Postgres + pgvector:
  # Example:
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/agentdb
  # Ensure pgvector extension exists (the app tries to create it on startup):
  #   CREATE EXTENSION IF NOT EXISTS vector;
  uvicorn api_service:app --reload --port 8000

Optional OpenAI usage (real models):
  export OPENAI_API_KEY=sk-...
  export OPENAI_MODEL=gpt-4o-mini
  export EMBEDDING_MODEL=text-embedding-3-small  # 1536-d

Endpoints:
  - GET    /health                                   -> health check
  - POST   /projects                                 -> create project
  - POST   /projects/{project_id}/phase1             -> run Phase‑1 from an idea
  - POST   /projects/{project_id}/phase2             -> run Phase‑2 (uses latest Phase‑1 by default)
  - GET    /projects/{project_id}/artifacts          -> list artifact summaries
  - GET    /artifacts/{artifact_id}                  -> fetch one artifact
  - POST   /projects/{project_id}/memory/search      -> semantic search in project memory
  - POST   /projects/{project_id}/memory/reindex     -> rebuild memory from stored artifacts

NOTE: For production, add auth, rate limits, request size limits, and structured logging.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, DateTime, String, Text, ForeignKey, JSON as SA_JSON, Index, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.engine import Engine

# Phase orchestrators and models
from phase1_exploration_orchestrator import (
    run_phase1_exploration,
    ExplorationArtifacts,
    default_llm as p1_default_llm,
)
from phase2_requirements_orchestrator import (
    run_phase2_requirements,
    RequirementsBundle,
    TestPlan,
    default_llm as p2_default_llm,
)

OPENAI_API_KEY="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"
os.environ['OPENAI_API_KEY'] = str(OPENAI_API_KEY)

# ======================================================
# Database setup (SQLite by default, Postgres via env)
# ======================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./phase1.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DIM = 1536  # text-embedding-3-small dimension as of 2024/2025

engine: Engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

DIALECT = engine.url.get_backend_name()  # 'sqlite', 'postgresql', ...
USE_TEXT_JSON = DIALECT == "sqlite"

# =====================
# ORM models
# =====================

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")

class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    phase = Column(String, nullable=False)  # e.g., 'phase1', 'phase2-requirements', 'phase2-testplan'
    if USE_TEXT_JSON:
        payload = Column(Text, nullable=False)  # JSON string for SQLite
    else:
        payload = Column(SA_JSON, nullable=False)  # JSON for Postgres
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="artifacts")
    __table_args__ = (
        Index("ix_artifacts_project_phase_created", "project_id", "phase", "created_at"),
    )

class MemorySqlite(Base):  # used only when sqlite
    __tablename__ = "memories_sqlite"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # e.g., 'BA', 'QA', 'Critic'
    kind = Column(String, nullable=False)  # e.g., 'problem', 'persona', 'user_story', 'test_case'
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON list of floats
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Ensure pgvector on Postgres and create the table if missing
if DIALECT == "postgresql":
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
              id uuid PRIMARY KEY,
              project_id text NOT NULL,
              role text NOT NULL,
              kind text NOT NULL,
              content text NOT NULL,
              embedding vector({VECTOR_DIM}) NOT NULL,
              created_at timestamptz DEFAULT now()
            );
            """
        ))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops)"))

# =====================
# Pydantic I/O models
# =====================

class CreateProjectIn(BaseModel):
    title: str = Field(..., description="Human-friendly project title")

class ProjectOut(BaseModel):
    id: str
    title: str
    status: str
    created_at: datetime

class RunPhase1In(BaseModel):
    idea: str = Field(..., description="The end-user's raw idea text")

class RunPhase2In(BaseModel):
    idea: Optional[str] = Field(None, description="Optional idea text; used if Phase‑1 not provided")
    phase1_artifact_id: Optional[str] = Field(None, description="If set, use this Phase‑1 artifact")

class ArtifactSummaryOut(BaseModel):
    id: str
    phase: str
    created_at: datetime

class ArtifactOut(BaseModel):
    id: str
    project_id: str
    phase: str
    created_at: datetime
    payload: dict

class MemorySearchIn(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=25)

class MemoryHit(BaseModel):
    id: str
    role: str
    kind: str
    content: str
    score: float

class MemorySearchOut(BaseModel):
    hits: List[MemoryHit]

# =====================
# LLM + embeddings utils
# =====================

def get_llm():
    """Choose Phase‑1/2 LLM: prefer Phase‑2's default (same behavior)."""
    return p2_default_llm() if os.getenv("OPENAI_API_KEY") else p1_default_llm()

@dataclass
class Embedder:
    model: str = EMBEDDING_MODEL

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Return vectors for texts. Uses OpenAI if key present, else deterministic pseudo-embeddings."""
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in resp.data]  # type: ignore[attr-defined]
            except Exception:
                return [self._pseudo_embedding(t) for t in texts]
        return [self._pseudo_embedding(t) for t in texts]

    def _pseudo_embedding(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = []
        while len(vals) < VECTOR_DIM:
            for b in h:
                vals.append((b / 255.0) * 2 - 1)
                if len(vals) == VECTOR_DIM:
                    break
        # L2 normalize
        import math as _m
        norm = _m.sqrt(sum(v*v for v in vals)) or 1.0
        return [v / norm for v in vals]

# =====================
# Memory store
# =====================

class MemoryStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.dialect = engine.url.get_backend_name()
        self.embedder = Embedder()

    def clear_project(self, project_id: str) -> None:
        if self.dialect == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM memories WHERE project_id = :p"), {"p": project_id})
        else:
            with SessionLocal() as db:
                db.query(MemorySqlite).filter(MemorySqlite.project_id == project_id).delete()
                db.commit()

    def index_items(self, project_id: str, items: List[Tuple[str, str, str]]):
        if not items:
            return
        contents = [c for _, _, c in items]
        vectors = self.embedder.embed_many(contents)
        now_items = []
        for (role, kind, content), vec in zip(items, vectors):
            now_items.append((str(uuid.uuid4()), project_id, role, kind, content, vec))

        if self.dialect == "postgresql":
            with self.engine.begin() as conn:
                for mid, pid, role, kind, content, vec in now_items:
                    arr = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                    conn.execute(
                        text(
                            """
                            INSERT INTO memories (id, project_id, role, kind, content, embedding)
                            VALUES (:id, :pid, :role, :kind, :content, :emb)
                            """
                        ),
                        {"id": mid, "pid": pid, "role": role, "kind": kind, "content": content, "emb": arr},
                    )
        else:
            with SessionLocal() as db:
                for mid, pid, role, kind, content, vec in now_items:
                    row = MemorySqlite(
                        id=mid,
                        project_id=pid,
                        role=role,
                        kind=kind,
                        content=content,
                        embedding=json.dumps(vec),
                    )
                    db.add(row)
                db.commit()

    def search(self, project_id: str, query: str, top_k: int = 5) -> List[MemoryHit]:
        qvec = self.embedder.embed_many([query])[0]
        if self.dialect == "postgresql":
            with self.engine.begin() as conn:
                sql = text(
                    f"""
                    SELECT id, role, kind, content, 1 - (embedding <-> :q) AS score
                    FROM memories
                    WHERE project_id = :pid
                    ORDER BY embedding <-> :q
                    LIMIT :k
                    """
                )
                arr = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
                rows = conn.execute(sql, {"q": arr, "pid": project_id, "k": top_k}).fetchall()
                return [MemoryHit(id=r[0], role=r[1], kind=r[2], content=r[3], score=float(r[4])) for r in rows]
        else:
            def cosine(a: List[float], b: List[float]) -> float:
                return float(sum(x*y for x, y in zip(a, b)))
            hits: List[MemoryHit] = []
            with SessionLocal() as db:
                rows = db.query(MemorySqlite).filter(MemorySqlite.project_id == project_id).all()
                for r in rows:
                    vec = json.loads(r.embedding)
                    score = cosine(qvec, vec)
                    hits.append(MemoryHit(id=r.id, role=r.role, kind=r.kind, content=r.content, score=score))
            hits.sort(key=lambda h: h.score, reverse=True)
            return hits[:top_k]

MEMORY = MemoryStore(engine)

# =====================
# FastAPI app
# =====================

app = FastAPI(title="Phases 1–2 API with Memory", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helpers

def _artifact_payload_to_dict(art: Artifact) -> dict:
    return json.loads(art.payload) if USE_TEXT_JSON else art.payload  # type: ignore[attr-defined]

def _store_artifact(db: Session, project_id: str, phase: str, payload_dict: dict) -> Artifact:
    payload_store = json.dumps(payload_dict) if USE_TEXT_JSON else payload_dict
    art = Artifact(project_id=project_id, phase=phase, payload=payload_store)
    db.add(art)
    db.commit()
    db.refresh(art)
    return art

# Indexing helpers

def _index_phase1(project_id: str, art: ExplorationArtifacts) -> None:
    items: List[Tuple[str, str, str]] = []
    ps = art.problem_statement
    items.append(("BA", "problem", f"Problem: {ps.problem}\nWhy now: {ps.why_now}\nGoals: {', '.join(ps.goals)}"))
    for p in art.personas:
        items.append(("BA", "persona", f"{p.name}: {p.description}. Goals: {', '.join(p.goals)}. Pains: {', '.join(p.pains)}"))
    for uc in art.use_cases:
        items.append(("BA", "use_case", f"{uc.title}. Trigger: {uc.trigger}. Steps: {'; '.join(uc.steps)}"))
    if art.constraints.technical:
        items.append(("BA", "constraint", "Technical: " + "; ".join(art.constraints.technical)))
    if art.constraints.legal_and_compliance:
        items.append(("BA", "constraint", "Compliance: " + "; ".join(art.constraints.legal_and_compliance)))
    if art.constraints.budget_time:
        items.append(("BA", "constraint", "Budget/Time: " + "; ".join(art.constraints.budget_time)))
    for r in art.risks.items:
        items.append(("Critic", "risk", r))
    for a in art.assumptions.items:
        items.append(("BA", "assumption", a))
    if art.scope.in_scope:
        items.append(("BA", "scope", "In: " + "; ".join(art.scope.in_scope)))
    if art.scope.out_of_scope:
        items.append(("BA", "scope", "Out: " + "; ".join(art.scope.out_of_scope)))
    for q in art.open_questions:
        items.append(("Critic", "open_question", f"{q.question} — {q.rationale or ''}"))
    MEMORY.index_items(project_id, items)


def _index_phase2(project_id: str, reqs: RequirementsBundle, tests: TestPlan) -> None:
    items: List[Tuple[str, str, str]] = []
    for us in reqs.user_stories:
        items.append(("BA", "user_story", f"{us.id} {us.title}: As a {us.as_a}, I want {us.i_want}, so that {us.so_that}. Priority: {us.priority}."))
        for ac in us.acceptance_criteria:
            items.append(("BA", "acceptance_criterion", f"{us.id}/{ac.id}: {ac.statement}"))
    for nfr in reqs.non_functionals:
        items.append(("Architect", "nfr", f"{nfr.category}: {nfr.statement}. Metric: {nfr.metric or ''} Target: {nfr.target or ''}"))
    for kpi in reqs.kpis:
        items.append(("PM", "kpi", f"{kpi.name}: {kpi.target}. {kpi.description or ''}"))
    for tc in tests.cases:
        items.append(("QA", "test_case", f"{tc.id} {tc.title}: {tc.objective}. Covers: {', '.join(tc.covers)}"))
    MEMORY.index_items(project_id, items)

# Routes

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dialect": DIALECT, "time": datetime.utcnow().isoformat()}

@app.post("/projects", response_model=ProjectOut)
def create_project(data: CreateProjectIn, db: Session = Depends(get_db)):
    proj = Project(title=data.title)
    db.add(proj)
    db.commit()
    db.refresh(proj)
    return ProjectOut(id=proj.id, title=proj.title, status=proj.status, created_at=proj.created_at)

@app.post("/projects/{project_id}/phase1", response_model=ArtifactOut)
def run_phase1(project_id: str, data: RunPhase1In, db: Session = Depends(get_db)):
    proj: Optional[Project] = db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    artifacts: ExplorationArtifacts = run_phase1_exploration(data.idea, llm=get_llm())
    payload = artifacts.model_dump()
    art = _store_artifact(db, proj.id, "phase1", payload)
    _index_phase1(proj.id, artifacts)
    return ArtifactOut(id=art.id, project_id=proj.id, phase=art.phase, created_at=art.created_at, payload=payload)

class RunPhase2Out(BaseModel):
    requirements: ArtifactOut
    tests: ArtifactOut

@app.post("/projects/{project_id}/phase2", response_model=RunPhase2Out)
def run_phase2(project_id: str, data: RunPhase2In, db: Session = Depends(get_db)):
    proj: Optional[Project] = db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    phase1_obj: Optional[ExplorationArtifacts] = None
    if data.phase1_artifact_id:
        art_row: Optional[Artifact] = db.get(Artifact, data.phase1_artifact_id)
        if not art_row or art_row.project_id != project_id:
            raise HTTPException(status_code=404, detail="Phase‑1 artifact not found for this project")
        if art_row.phase != "phase1":
            raise HTTPException(status_code=400, detail="Provided artifact is not Phase‑1")
        phase1_obj = ExplorationArtifacts.model_validate(_artifact_payload_to_dict(art_row))
    else:
        row: Optional[Artifact] = (
            db.query(Artifact)
            .filter(Artifact.project_id == project_id, Artifact.phase == "phase1")
            .order_by(Artifact.created_at.desc())
            .first()
        )
        if row is not None:
            phase1_obj = ExplorationArtifacts.model_validate(_artifact_payload_to_dict(row))

    idea = data.idea or ""
    outputs = run_phase2_requirements(idea, phase1=phase1_obj, llm=get_llm())
    reqs: RequirementsBundle = outputs["requirements"]
    tests: TestPlan = outputs["tests"]

    req_art = _store_artifact(db, proj.id, "phase2-requirements", reqs.model_dump())
    t_art = _store_artifact(db, proj.id, "phase2-testplan", tests.model_dump())

    _index_phase2(proj.id, reqs, tests)

    return RunPhase2Out(
        requirements=ArtifactOut(
            id=req_art.id, project_id=proj.id, phase=req_art.phase, created_at=req_art.created_at, payload=reqs.model_dump()
        ),
        tests=ArtifactOut(
            id=t_art.id, project_id=proj.id, phase=t_art.phase, created_at=t_art.created_at, payload=tests.model_dump()
        ),
    )

@app.get("/projects/{project_id}/artifacts", response_model=List[ArtifactSummaryOut])
def list_artifacts(project_id: str, db: Session = Depends(get_db)):
    proj: Optional[Project] = db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    rows: List[Artifact] = (
        db.query(Artifact).filter(Artifact.project_id == project_id).order_by(Artifact.created_at.desc()).all()
    )
    return [ArtifactSummaryOut(id=r.id, phase=r.phase, created_at=r.created_at) for r in rows]

@app.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
def get_artifact(artifact_id: str, db: Session = Depends(get_db)):
    art: Optional[Artifact] = db.get(Artifact, artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactOut(id=art.id, project_id=art.project_id, phase=art.phase, created_at=art.created_at, payload=_artifact_payload_to_dict(art))

@app.post("/projects/{project_id}/memory/reindex", response_model=MemorySearchOut)
def memory_reindex(project_id: str, db: Session = Depends(get_db)):
    proj: Optional[Project] = db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    MEMORY.clear_project(project_id)

    rows: List[Artifact] = (
        db.query(Artifact).filter(Artifact.project_id == project_id).order_by(Artifact.created_at.asc()).all()
    )
    last_reqs: Optional[RequirementsBundle] = None
    for r in rows:
        payload = _artifact_payload_to_dict(r)
        if r.phase == "phase1":
            _index_phase1(project_id, ExplorationArtifacts.model_validate(payload))
        elif r.phase == "phase2-requirements":
            last_reqs = RequirementsBundle.model_validate(payload)
            _index_phase2(project_id, last_reqs, TestPlan(cases=[]))
        elif r.phase == "phase2-testplan":
            tests = TestPlan.model_validate(payload)
            if last_reqs is None:
                # Index tests alone
                items: List[Tuple[str, str, str]] = []
                for tc in tests.cases:
                    items.append(("QA", "test_case", f"{tc.id} {tc.title}: {tc.objective}. Covers: {', '.join(tc.covers)}"))
                MEMORY.index_items(project_id, items)
            else:
                _index_phase2(project_id, last_reqs, tests)
    return MemorySearchOut(hits=[])

@app.post("/projects/{project_id}/memory/search", response_model=MemorySearchOut)
def memory_search(project_id: str, data: MemorySearchIn):
    hits = MEMORY.search(project_id, data.query, top_k=data.top_k)
    return MemorySearchOut(hits=hits)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_service:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
