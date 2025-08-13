"""
./phase2_requirements_orchestrator.py
Phase 2 — Requirements & Success Measures (first draft)
------------------------------------------------------

This module turns Phase‑1 exploration artifacts into a structured set of
**requirements** (user stories with acceptance criteria), **non-functional
requirements**, and **KPIs**, plus an initial **test plan**. It follows a
turn-based flow:

  BA(draft requirements) -> QA(derive tests) -> Critic(review) -> BA(revise)

Design goals:
  - Strong schemas (Pydantic) and JSON-only I/O with the LLM
  - Pluggable LLM client (reuses Phase‑1 client: OpenAI or Mock)
  - Deterministic, simple orchestrator suitable for API wiring later

Usage (quick demo):
  python phase2_requirements_orchestrator.py

You can wire this into FastAPI by creating a router or by calling
`run_phase2_requirements()` inside an endpoint that loads Phase‑1 artifacts
from your DB, then stores Phase‑2 outputs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

# Reuse LLM abstractions and Phase‑1 outputs
from phase1_exploration_orchestrator import (
    ExplorationArtifacts,
    LLMClient,
    OpenAIChatClient,
    MockLLMClient,
    run_phase1_exploration,
)

# ==============================
# 1) Data models (Pydantic)
# ==============================

Priority = Literal["MUST", "SHOULD", "COULD", "WONT"]

class AcceptanceCriterion(BaseModel):
    id: str = Field(..., description="Unique ID, e.g., AC-1")
    statement: str = Field(..., description="Testable outcome in plain language")

class UserStory(BaseModel):
    id: str = Field(..., description="Unique story ID, e.g., US-1")
    title: str
    as_a: str = Field(..., description="Role/persona")
    i_want: str = Field(..., description="Capability desired")
    so_that: str = Field(..., description="Outcome/benefit")
    description: Optional[str] = None
    priority: Priority = "SHOULD"
    dependencies: List[str] = Field(default_factory=list, description="Other story IDs or external deps")
    acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list)

class NFRCategory(BaseModel):
    category: Literal[
        "performance",
        "security",
        "reliability",
        "usability",
        "maintainability",
        "portability",
        "compliance",
        "availability",
        "scalability",
        "observability",
    ]
    statement: str = Field(..., description="Non-functional requirement statement")
    metric: Optional[str] = Field(None, description="How it's measured, e.g., 'p95 latency' or 'CVSS <= X'")
    target: Optional[str] = Field(None, description="Target threshold, e.g., 'p95 < 300ms' or '99.9% uptime'")

class KPI(BaseModel):
    name: str
    description: Optional[str] = None
    baseline: Optional[str] = None
    target: str
    timeframe: Optional[str] = Field(None, description="e.g., 'by end of Q2' or '30 days post-launch'")

class GlossaryTerm(BaseModel):
    term: str
    definition: str

class RequirementsBundle(BaseModel):
    """Primary Phase‑2 output (requirements + measures)."""
    user_stories: List[UserStory] = Field(default_factory=list)
    non_functionals: List[NFRCategory] = Field(default_factory=list)
    kpis: List[KPI] = Field(default_factory=list)
    glossary: List[GlossaryTerm] = Field(default_factory=list)
    constraints_inherited: List[str] = Field(default_factory=list, description="From Phase‑1 constraints")
    open_questions: List[str] = Field(default_factory=list)

class TestStep(BaseModel):
    action: str
    expected: Optional[str] = None

class TestCaseSpec(BaseModel):
    id: str = Field(..., description="Unique test ID, e.g., TC-1")
    title: str
    objective: str
    preconditions: List[str] = Field(default_factory=list)
    steps: List[TestStep] = Field(default_factory=list)
    covers: List[str] = Field(default_factory=list, description="IDs of stories/ACs covered (e.g., ['US-1', 'AC-1'])")

class TestPlan(BaseModel):
    cases: List[TestCaseSpec] = Field(default_factory=list)

class RequirementsCritique(BaseModel):
    issues: List[str] = Field(default_factory=list)
    suggested_changes: Dict[str, Any] = Field(default_factory=dict)
    missing_tests_for: List[str] = Field(default_factory=list, description="Story/AC IDs that lack tests")
    risk_notes: List[str] = Field(default_factory=list)

# ==============================
# 2) System prompts (roles)
# ==============================

BA_REQ_SYSTEM = (
    "You are a meticulous Business Analyst AI. Output ONLY JSON that validates against the provided schema. "
    "Write testable acceptance criteria (use clear, verifiable language). Use MoSCoW for priority."
)

QA_SYSTEM = (
    "You are a pragmatic QA/Test Engineer AI. Output ONLY JSON for a test plan that aligns to the given stories and ACs. "
    "Prefer concise, reproducible steps and explicit coverage links to story/AC IDs."
)

CRITIC_REQ_SYSTEM = (
    "You are a rigorous Requirements Critic AI. Output ONLY JSON for the critique schema. "
    "Flag missing NFRs, unclear ACs, untestable statements, and gaps in test coverage."
)

# ==============================
# 3) Agents
# ==============================

class BusinessAnalystRequirementsAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def draft(self, idea: str, phase1: ExplorationArtifacts) -> RequirementsBundle:
        schema_json = RequirementsBundle.model_json_schema()
        p1_json = phase1.model_dump()
        user_prompt = f"""
        TASK: Derive a requirements bundle from the user's idea and the Phase‑1 exploration artifacts.

        USER_IDEA:\n{idea}\n
        PHASE1_ARTIFACTS_JSON:\n{json.dumps(p1_json, indent=2)}

        RULES:
        - Produce user stories using the As a <role>, I want <capability>, so that <benefit> template.
        - Give each story and AC a stable ID (US-1, AC-1...). Keep IDs short and unique.
        - Add NFRs across security, performance, reliability, usability, maintainability, and compliance when relevant.
        - Propose KPIs tied to outcomes, with clear targets.
        - Inherit relevant constraints from Phase‑1 into constraints_inherited.
        - If information is missing, include precise open_questions.

        Output ONLY JSON for this schema:
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=BA_REQ_SYSTEM, user=user_prompt, model_hint="phase2_requirements")
        return RequirementsBundle.model_validate(raw)

    def revise_with_critique(self, prior: RequirementsBundle, critique: RequirementsCritique) -> RequirementsBundle:
        schema_json = RequirementsBundle.model_json_schema()
        user_prompt = f"""
        TASK: Apply the critique to revise the requirements bundle while preserving correct content.

        PRIOR_REQUIREMENTS_JSON:\n{json.dumps(prior.model_dump(), indent=2)}

        CRITIQUE_JSON:\n{json.dumps(critique.model_dump(), indent=2)}

        RULES:
        - Clarify ambiguous ACs into testable, measurable language.
        - Add missing NFRs and KPIs explicitly when requested.
        - Keep IDs stable where possible; add suffixes (e.g., AC-1a) if you must split.

        Output ONLY JSON for this schema:
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=BA_REQ_SYSTEM, user=user_prompt, model_hint="phase2_requirements")
        return RequirementsBundle.model_validate(raw)

class QAAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def derive_tests(self, reqs: RequirementsBundle) -> TestPlan:
        schema_json = TestPlan.model_json_schema()
        user_prompt = f"""
        TASK: Generate a concise, high-value **manual test plan** from the requirements. Map tests to story/AC IDs.
        - Prefer critical paths and edge cases for MVP.
        - Keep each test focused; use clear preconditions and deterministic checks.

        REQUIREMENTS_JSON:\n{json.dumps(reqs.model_dump(), indent=2)}

        Output ONLY JSON for this schema:
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=QA_SYSTEM, user=user_prompt, model_hint="phase2_testplan")
        return TestPlan.model_validate(raw)

class CriticRequirementsAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def critique(self, reqs: RequirementsBundle, tests: TestPlan) -> RequirementsCritique:
        schema_json = RequirementsCritique.model_json_schema()
        user_prompt = f"""
        TASK: Review the requirements and test plan for clarity, completeness, and testability.
        - Identify missing or weak NFRs.
        - Point out any stories/ACs that lack test coverage.
        - Flag ambiguous or non-measurable language.

        REQUIREMENTS_JSON:\n{json.dumps(reqs.model_dump(), indent=2)}

        TESTPLAN_JSON:\n{json.dumps(tests.model_dump(), indent=2)}

        Output ONLY JSON for this schema:
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=CRITIC_REQ_SYSTEM, user=user_prompt, model_hint="phase2_critique")
        return RequirementsCritique.model_validate(raw)

# ==============================
# 4) Orchestrator
# ==============================

def default_llm() -> LLMClient:
    """Select the LLM backend (OpenAI if key is present, else Mock)."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIChatClient(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return MockLLMClient()

def run_phase2_requirements(
    idea: str,
    *,
    phase1: Optional[ExplorationArtifacts] = None,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Execute Phase‑2: BA -> QA -> Critic -> BA revision.

    Returns a dict with keys: {"requirements": RequirementsBundle, "tests": TestPlan}
    (Both objects are also available under .model_dump() for JSON persistence.)
    """
    llm = llm or default_llm()

    # If Phase‑1 artifacts aren't provided, generate a quick draft from the idea
    if phase1 is None:
        phase1 = run_phase1_exploration(idea, llm=llm)

    ba = BusinessAnalystRequirementsAgent(llm)
    qa = QAAgent(llm)
    critic = CriticRequirementsAgent(llm)

    # 1) BA drafts requirements
    reqs_v1 = ba.draft(idea, phase1)

    # 2) QA derives initial tests
    tests_v1 = qa.derive_tests(reqs_v1)

    # 3) Critic reviews both
    critique = critic.critique(reqs_v1, tests_v1)

    # 4) BA revises requirements
    reqs_v2 = ba.revise_with_critique(reqs_v1, critique)

    # 5) QA updates tests (optional second pass). Keep simple: re-derive from revised reqs.
    tests_v2 = qa.derive_tests(reqs_v2)

    return {"requirements": reqs_v2, "tests": tests_v2}

# ==============================
# 5) Quick demo (CLI)
# ==============================

if __name__ == "__main__":
    # Sample idea (replace with real input)
    idea_text = (
        "An app that helps parents coordinate after-school activities, rides, and payments "
        "for their children, with reminders and last-minute change handling."
    )

    try:
        print("Running Phase‑2 requirements orchestration...\n")
        outputs = run_phase2_requirements(idea_text)
        reqs = outputs["requirements"]
        tests = outputs["tests"]
        print("\n=== REQUIREMENTS (Phase‑2) ===\n")
        print(json.dumps(reqs.model_dump(), indent=2))
        print("\n=== TEST PLAN (Phase‑2) ===\n")
        print(json.dumps(tests.model_dump(), indent=2))
    except ValidationError as ve:
        print("Validation failed:", ve)
    except Exception as e:
        print("Error:", e)
