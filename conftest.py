import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def _env_setup():
    os.environ.setdefault("DATABASE_URL", "sqlite:///./phase1.db")
    os.getenv("OPENAI_API_KEY")
    yield
