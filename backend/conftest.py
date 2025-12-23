import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# IMPORTANT: Set env vars at import time.
# pytest imports `conftest.py` before importing test modules, which ensures our
# env overrides take effect before `app.core.config.settings` is instantiated.
# ---------------------------------------------------------------------------
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="rag_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_ROOT / 'test.db'}")
os.environ.setdefault("UPLOAD_DIR", str(_TEST_ROOT / "uploads"))
os.environ.setdefault("OCR_ENABLED", "false")

# Ensure static mount path exists before importing `app.main`.
Path(os.environ["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def app():
    """Import the FastAPI app after env is configured."""
    from app.core.config import settings
    from app.db.session import Base
    from app.models import conversation as _c
    from app.models import document as _d
    from app.main import app as fastapi_app

    # Ensure upload dirs exist
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.UPLOAD_DIR, "documents").mkdir(parents=True, exist_ok=True)
    Path(settings.UPLOAD_DIR, "images").mkdir(parents=True, exist_ok=True)
    Path(settings.UPLOAD_DIR, "tables").mkdir(parents=True, exist_ok=True)

    # Create SQLite tables
    engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    # Override get_engine() to return our SQLite engine
    import app.db.session as session_mod

    session_mod._engine = engine

    return fastapi_app


@pytest.fixture()
def client(app):
    return TestClient(app)
