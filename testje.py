from phase1_exploration_orchestrator import run_phase1_exploration, MockLLMClient
from phase2_requirements_orchestrator import run_phase2_requirements

idea = "Parents need help coordinating rides & payments"
p1 = run_phase1_exploration(idea, llm=MockLLMClient())
print(p1.model_dump())

#out = run_phase2_requirements(idea, llm=MockLLMClient())
out = run_phase2_requirements(idea, llm=MockLLMClient())
#gpt-4o-mini
print(out["requirements"].model_dump())
print(out["tests"].model_dump())