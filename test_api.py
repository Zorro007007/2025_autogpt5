import os
from fastapi.testclient import TestClient
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
