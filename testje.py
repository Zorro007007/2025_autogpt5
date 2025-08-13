from phase1_exploration_orchestrator import run_phase1_exploration, OpenAIChatClient
from phase2_requirements_orchestrator import run_phase2_requirements

idea = "Parents need help coordinating rides & payments"

# Use the real model explicitly (you can also omit llm=... and rely on env)
llm = OpenAIChatClient("gpt-4o-mini")

# Phase 1
p1 = run_phase1_exploration(idea, llm=llm)
print("=== Phase 1: Problem Statement ===")
print(p1.problem_statement.problem)

# Phase 2
out = run_phase2_requirements(idea, phase1=p1, llm=llm)
reqs = out["requirements"]
tests = out["tests"]

# Peek at some results
print("\n=== First user story ===")
print(reqs.user_stories[0].model_dump())

print("\n=== First test case ===")
print(tests.cases[0].model_dump())
