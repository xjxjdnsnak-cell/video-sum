import sqlite3
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from .config import settings


def get_db_connection() -> sqlite3.Connection:
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_path TEXT,
                url TEXT,
                title TEXT,
                author TEXT,
                duration REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text TEXT NOT NULL,
                source TEXT DEFAULT 'asr',
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                source_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS final_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL UNIQUE,
                one_sentence_summary TEXT,
                detailed_summary TEXT,
                key_points TEXT,
                questions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        conn.commit()
