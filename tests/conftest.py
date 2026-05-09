import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def setup_env(temp_dir, monkeypatch):
    monkeypatch.setenv("HOME", str(temp_dir))
    db_dir = temp_dir / ".video_summarizer"
    db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "video_summarizer.config.settings.DB_PATH",
        db_dir / "test.db"
    )
    monkeypatch.setattr(
        "video_summarizer.config.settings.OUTPUT_DIR",
        temp_dir / "output"
    )
