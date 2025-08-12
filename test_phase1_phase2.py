
from phase1_exploration_orchestrator import run_phase1_exploration, MockLLMClient
from phase2_requirements_orchestrator import run_phase2_requirements

def test_phase1_with_mock():
    idea = "Simple idea for coordination app"
    art = run_phase1_exploration(idea, llm=MockLLMClient())
    d = art.model_dump()
    assert "problem_statement" in d and d["problem_statement"]["problem"]
    assert isinstance(art.personas, list) and len(art.personas) >= 1
    assert isinstance(art.use_cases, list) and len(art.use_cases) >= 1

def test_phase2_with_mock():
    idea = "Simple idea for coordination app"
    outputs = run_phase2_requirements(idea, llm=MockLLMClient())
    reqs = outputs["requirements"]
    tests = outputs["tests"]
    assert len(reqs.user_stories) >= 1
    assert hasattr(tests, "cases")
