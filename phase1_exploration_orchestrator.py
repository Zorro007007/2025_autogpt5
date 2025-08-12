"""
3) Run a quick test at the bottom of this file or import `run_phase1_exploration`.

Notes
-----
- This file *does not* include a DB; instead, `persist_artifacts()` is a stub
  where you can connect PostgreSQL/pgvector later.
- The OpenAI client uses "response_format={'type': 'json_object'}" to push the
  model to emit valid JSON. We still include simple repair/retry logic.
- You can drop in your own LLM backend by implementing `LLMClient`.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

OPENAI_API_KEY="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"
os.environ['OPENAI_API_KEY'] = str(OPENAI_API_KEY)

# ==============================
# 1) Data models (Pydantic)
# ==============================

class Persona(BaseModel):
    name: str = Field(..., description="Persona's name (e.g., Primary Teacher)")
    description: str = Field(..., description="Short bio / context")
    goals: List[str] = Field(default_factory=list, description="What this persona wants to achieve")
    pains: List[str] = Field(default_factory=list, description="Frustrations or blockers")

class UseCase(BaseModel):
    title: str
    actors: List[str] = Field(default_factory=list)
    trigger: str = Field(..., description="What starts the scenario")
    steps: List[str] = Field(default_factory=list, description="Key user/system interactions")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Measurable outcomes that define success")

class ProblemStatement(BaseModel):
    problem: str
    why_now: str
    goals: List[str] = Field(default_factory=list)
    non_goals: List[str] = Field(default_factory=list)

class Constraints(BaseModel):
    technical: List[str] = Field(default_factory=list)
    legal_and_compliance: List[str] = Field(default_factory=list)
    budget_time: List[str] = Field(default_factory=list)

class Risks(BaseModel):
    items: List[str] = Field(default_factory=list)
    mitigations: Dict[str, str] = Field(default_factory=dict, description="Map risk -> suggested mitigation")

class Assumptions(BaseModel):
    items: List[str] = Field(default_factory=list)

class Scope(BaseModel):
    in_scope: List[str] = Field(default_factory=list)
    out_of_scope: List[str] = Field(default_factory=list)

class OpenQuestion(BaseModel):
    question: str
    rationale: Optional[str] = None

class ExplorationArtifacts(BaseModel):
    """Canonical output for Phase 1."""
    problem_statement: ProblemStatement
    personas: List[Persona] = Field(default_factory=list)
    use_cases: List[UseCase] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    risks: Risks = Field(default_factory=Risks)
    assumptions: Assumptions = Field(default_factory=Assumptions)
    scope: Scope = Field(default_factory=Scope)
    open_questions: List[OpenQuestion] = Field(default_factory=list)

class Critique(BaseModel):
    """Critic/Judge feedback. The BA will incorporate this into a revision."""
    issues: List[str] = Field(default_factory=list, description="What is wrong or missing")
    suggested_changes: Dict[str, Any] = Field(default_factory=dict, description="Partial JSON patches or fields to tweak")
    priority_open_questions: List[str] = Field(default_factory=list)

# ==============================
# 2) LLM client abstraction
# ==============================

class LLMClient(ABC):
    """Pluggable LLM interface. Swap with your own backend easily."""

    @abstractmethod
    def complete_json(self, *, system: str, user: str, model_hint: Optional[str] = None) -> Dict[str, Any]:
        """Return *only* JSON as a Python dict. Raise on failure."""
        raise NotImplementedError

class OpenAIChatClient(LLMClient):
    """OpenAI Chat Completions client (simple, battle-tested).

    Requires `pip install openai` and OPENAI_API_KEY.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # lazy import
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install the 'openai' package to use OpenAIChatClient") from e

        self._OpenAI = OpenAI
        self._client = OpenAI()
        self.model = model

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def complete_json(self, *, system: str, user: str, model_hint: Optional[str] = None) -> Dict[str, Any]:
        # We instruct the model to emit strict JSON. The SDK supports a JSON response format.
        resp = self._client.chat.completions.create(
            model=self.model, 
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # As a fallback, try a lightweight repair prompt.
            repair_prompt = f"""
            The following text was supposed to be valid JSON but isn't. Fix it and return only valid JSON:
            --- START ---
            {content}
            --- END ---
            """
            resp2 = self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a JSON linter and fixer. Output only corrected JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
            )
            return json.loads(resp2.choices[0].message.content)

class MockLLMClient(LLMClient):
    """Mock backend for local testing without API calls."""

    def complete_json(self, *, system: str, user: str, model_hint: Optional[str] = None) -> Dict[str, Any]:
        # ---- Phase-2 outputs (detected via model_hint) ----
        if model_hint == "phase2_requirements":
            return {
                "user_stories": [
                    {
                        "id": "US-1",
                        "title": "Create activity and invite",
                        "as_a": "Parent Coordinator",
                        "i_want": "to create an activity and invite participants",
                        "so_that": "I can coordinate schedules",
                        "priority": "MUST",
                        "dependencies": [],
                        "acceptance_criteria": [
                            {"id": "AC-1", "statement": "Invites are delivered within 1 minute"},
                            {"id": "AC-2", "statement": "Participants can RSVP from the invite"},
                        ],
                    }
                ],
                "non_functionals": [
                    {"category": "performance", "statement": "Fast creation", "metric": "p95 latency", "target": "p95 < 500ms"},
                    {"category": "security", "statement": "No card data stored", "metric": "PCI scope", "target": "Out of scope"},
                ],
                "kpis": [{"name": "Time-to-schedule", "target": "↓ 50% within 30 days", "timeframe": "30 days post-launch"}],
                "glossary": [],
                "constraints_inherited": ["MVP in 8 weeks"],
                "open_questions": ["Do we require multi-language at launch?"],
            }

        if model_hint == "phase2_testplan":
            return {
                "cases": [
                    {
                        "id": "TC-1",
                        "title": "Create activity and send invites",
                        "objective": "User can create activity and invites are delivered",
                        "preconditions": ["User is authenticated"],
                        "steps": [
                            {"action": "Open Create Activity", "expected": "Form visible"},
                            {"action": "Fill details and submit", "expected": "Activity created"},
                            {"action": "Send invites", "expected": "Invites delivered within 1 minute"},
                        ],
                        "covers": ["US-1", "AC-1"],
                    }
                ]
            }

        if model_hint == "phase2_critique":
            return {
                "issues": ["Clarify payments flow ACs", "Add explicit security NFR for PCI"],
                "suggested_changes": {
                    "non_functionals": [
                        {"category": "security", "statement": "Tokenize PANs; never store card data", "metric": "PCI scope", "target": "Out of scope"}
                    ]
                },
                "missing_tests_for": ["US-1/AC-2"],
                "risk_notes": ["3rd-party payment outages may affect checkouts"],
            }

        # ---- Phase-1 critique (unchanged) ----
        if "CRITIQUE_ONLY_JSON" in system:
            return {
                "issues": [
                    "Non-functional requirements not covered",
                    "No GDPR/compliance constraint referenced",
                ],
                "suggested_changes": {
                    "constraints": {"legal_and_compliance": ["Consider GDPR, data residency"]},
                    "assumptions": {"items": ["Pilot with 100 users in month 1"]},
                },
                "priority_open_questions": [
                    "What is the primary KPI for launch?",
                    "Which markets are in scope in phase 1?",
                ],
            }

        # ---- Phase-1 exploration artifacts (default) ----
        return {
            "problem_statement": {
                "problem": "Users struggle to coordinate after-school activities and logistics.",
                "why_now": "Mobile-first adoption and API access to transport/payment providers make it feasible.",
                "goals": ["Reduce coordination time by 50%", "Increase attendance reliability"],
                "non_goals": ["Build a full-fledged social network"],
            },
            "personas": [
                {
                    "name": "Parent Coordinator",
                    "description": "Schedules and communicates with other parents and children.",
                    "goals": ["Fast scheduling", "Reliable notifications"],
                    "pains": ["Last-minute changes", "Fragmented channels"],
                }
            ],
            "use_cases": [
                {
                    "title": "Create activity and invite",
                    "actors": ["Parent Coordinator"],
                    "trigger": "Parent creates a new weekly activity",
                    "steps": ["Enter details and schedule", "Select participants", "Send invites"],
                    "acceptance_criteria": ["Invitees receive notification within 1 minute"],
                }
            ],
            "constraints": {
                "technical": ["Mobile-first", "Stripe/payments integration"],
                "legal_and_compliance": [],
                "budget_time": ["MVP in 8 weeks"],
            },
            "risks": {"items": ["Low adoption", "Payment disputes"], "mitigations": {"Low adoption": "Seed with a pilot group"}},
            "assumptions": {"items": ["Parents pay via card", "Stable internet access"]},
            "scope": {"in_scope": ["Scheduling", "Invites", "Payments"], "out_of_scope": ["Chat"]},
            "open_questions": [{"question": "Do we need multi-language at launch?", "rationale": "International audience"}],
        }


# ==============================
# 3) Role prompts (system messages)
# ==============================

BA_SYSTEM = (
    "You are a Business Analyst AI. Produce only a JSON object that validates against the provided schema. "
    "Be specific, practical, and realistic. Avoid marketing fluff."
)

CRITIC_SYSTEM = (
    "You are a rigorous Critic/Judge AI. Return only JSON that validates against the CRITIQUE schema. "
    "Name substantive issues and propose concise, actionable changes.\n"
    "Tag missing compliance/security concerns if applicable.\n"
    "CRITIQUE_ONLY_JSON"
)

# ==============================
# 4) Agent wrappers
# ==============================

class BusinessAnalystAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def draft_artifacts(self, idea: str) -> ExplorationArtifacts:
        # The "user" prompt includes the JSON schema so the model knows the contract.
        schema_json = ExplorationArtifacts.model_json_schema()
        user_prompt = f"""
        TASK: Given the user's product idea, produce Phase-1 exploration artifacts.

        USER_IDEA:\n{idea}\n
        STRICT REQUIREMENTS:
        - Output ONLY JSON that conforms to the following JSON Schema (no Markdown, no commentary):
        - If certain fields are unknown, provide sensible defaults and include related open_questions.

        JSON_SCHEMA (for reference):
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=BA_SYSTEM, user=user_prompt)
        return ExplorationArtifacts.model_validate(raw)

    def revise_with_critique(self, prior: ExplorationArtifacts, critique: Critique) -> ExplorationArtifacts:
        schema_json = ExplorationArtifacts.model_json_schema()
        critique_json = critique.model_dump()
        prior_json = prior.model_dump()
        user_prompt = f"""
        TASK: Apply the Critic's feedback to revise the artifacts. Return ONLY valid JSON for the artifacts.

        PRIOR_ARTIFACTS_JSON:\n{json.dumps(prior_json, indent=2)}

        CRITIC_FEEDBACK_JSON:\n{json.dumps(critique_json, indent=2)}

        RULES:
        - Preserve good content, only modify what the critique highlights.
        - Ensure final output validates against the same schema and addresses priority_open_questions where possible.

        JSON_SCHEMA (for reference):
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=BA_SYSTEM, user=user_prompt)
        return ExplorationArtifacts.model_validate(raw)

class CriticAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def critique(self, artifacts: ExplorationArtifacts) -> Critique:
        schema_json = Critique.model_json_schema()
        artifacts_json = artifacts.model_dump()
        user_prompt = f"""
        TASK: Review the exploration artifacts for completeness, feasibility, testability, and compliance awareness.
        Return ONLY JSON that validates against the following CRITIQUE schema.

        ARTIFACTS_JSON:\n{json.dumps(artifacts_json, indent=2)}

        CRITIQUE_SCHEMA:
        {json.dumps(schema_json, indent=2)}
        """
        raw = self.llm.complete_json(system=CRITIC_SYSTEM, user=user_prompt)
        return Critique.model_validate(raw)

# ==============================
# 5) Orchestrator for Phase 1
# ==============================

def run_phase1_exploration(idea: str, *, llm: Optional[LLMClient] = None) -> ExplorationArtifacts:
    """Run a single exploration cycle: BA -> Critic -> BA revision.

    - Validates and re-validates JSON at each step using Pydantic.
    - If anything fails to validate, we raise an exception for now.
      (You can catch and implement retries or human-in-the-loop.)
    """
    llm = llm or default_llm()

    ba = BusinessAnalystAgent(llm)
    critic = CriticAgent(llm)

    # 1) BA drafts initial artifacts
    artifacts_v1 = ba.draft_artifacts(idea)

    # 2) Critic reviews and suggests changes
    critique = critic.critique(artifacts_v1)

    # 3) BA revises accordingly
    artifacts_v2 = ba.revise_with_critique(artifacts_v1, critique)

    # 4) Final validation is already guaranteed by Pydantic; return the improved set
    return artifacts_v2

# ==============================
# 6) Persistence hook (stub)
# ==============================

def persist_artifacts(project_id: str, artifacts: ExplorationArtifacts) -> None:
    """Persist artifacts where you like (DB, S3, Git). This is a stub.

    Example wiring to PostgreSQL later:
      - Serialize: payload = artifacts.model_dump_json(indent=2)
      - Store in a table `artifacts(project_id text, phase text, payload jsonb, created_at timestamptz)`
      - Optionally index key subfields for search/filters.
    """
    # For now, we just print a short confirmation.
    print(f"[persist] project_id={project_id}, bytes={len(artifacts.model_dump_json())}")

# ==============================
# 7) Defaults & helpers
# ==============================

def default_llm() -> LLMClient:
    """Choose an LLM backend based on environment.

    - If OPENAI_API_KEY is set -> use OpenAI.
    - Else -> use Mock client (offline dev mode).
    """

    if os.getenv("OPENAI_API_KEY"):
        return OpenAIChatClient(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return MockLLMClient()

# ==============================
# 8) Quick demo (CLI)
# ==============================

if __name__ == "__main__":
    # Provide a sample idea. Replace with your user's input.
    idea_text = (
        "An app that helps parents coordinate after-school activities, rides, and payments "
        "for their children, with reminders and last-minute change handling."
    )

    print("Running Phase 1 exploration...\n")
    try:
        result = run_phase1_exploration(idea_text)
        # Pretty-print final artifacts
        print(json.dumps(result.model_dump(), indent=2))
        # Persist wherever you like
        persist_artifacts(project_id="demo-project-001", artifacts=result)
    except ValidationError as ve:
        print("Validation failed:", ve)
    except Exception as e:
        print("Error:", e)