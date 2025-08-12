
import os
from fastapi.testclient import TestClient

os.environ.pop("OPENAI_API_KEY", None)
OPENAI_API_KEY="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"
os.environ['OPENAI_API_KEY'] = str(OPENAI_API_KEY)
os.environ.setdefault("DATABASE_URL", "sqlite:///./phase1.db")

import api_service as svc

client = TestClient(svc.app)

def test_end_to_end_api(tmp_path):
    r = client.post("/projects", json={"title": "Demo Project"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    r = client.post(f"/projects/{pid}/phase1", json={"idea": "Parents need help coordinating rides & payments"})
    assert r.status_code == 200, r.text

    r = client.post(f"/projects/{pid}/phase2", json={})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "requirements" in payload and "tests" in payload

    r = client.post(f"/projects/{pid}/memory/search", json={"query": "payments", "top_k": 3})
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["hits"], list)
