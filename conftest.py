import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def _env_setup():
    os.environ.setdefault("DATABASE_URL", "sqlite:///./phase1.db")
    os.environ.pop("OPENAI_API_KEY", None)
    OPENAI_API_KEY="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"
    os.environ['OPENAI_API_KEY'] = str(OPENAI_API_KEY)
    yield
