import sqlite3
import tempfile
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from pathlib import Path
from datetime import datetime

from .config import settings


def setup_logging():
    log_dir = settings.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_stage TEXT DEFAULT 'created',
                failed_stage TEXT,
                last_error TEXT,
                output_markdown_path TEXT,
                output_json_path TEXT,
                output_srt_path TEXT
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
                extra_data TEXT,
                note_style TEXT DEFAULT 'detailed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                final_status TEXT,
                error_message TEXT,
                stages_completed TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                chunk_indices TEXT,
                summary TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                explanation TEXT,
                first_seen_time TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            )
        """)
        # Indexes on the video_id FK columns: every hot query
        # (has_transcript_segments, has_summary_chunks, get_video_info's COUNT
        # subqueries, get_transcript, ...) filters by video_id. Without these
        # each one is a full table scan once the tables grow.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transcript_segments_video
            ON transcript_segments(video_id, start)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_chunks_video
            ON summary_chunks(video_id, start)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_chapters_video
            ON video_chapters(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_quotes_video
            ON video_quotes(video_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_terms_video
            ON video_terms(video_id)
        """)
        conn.commit()


def update_video_stage(video_id: int, stage: str, error: Optional[str] = None):
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        if error:
            cursor.execute("""
                UPDATE videos
                SET current_stage = ?, failed_stage = ?, last_error = ?, updated_at = ?
                WHERE id = ?
            """, (stage, stage, error, now, video_id))
        else:
            cursor.execute("""
                UPDATE videos
                SET current_stage = ?, updated_at = ?
                WHERE id = ?
            """, (stage, now, video_id))
        conn.commit()


def update_video_status(video_id: int, status: str, stage: Optional[str] = None):
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        if stage:
            cursor.execute("""
                UPDATE videos
                SET status = ?, current_stage = ?, updated_at = ?
                WHERE id = ?
            """, (status, stage, now, video_id))
        else:
            cursor.execute("""
                UPDATE videos
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status, now, video_id))
        conn.commit()


def update_video_outputs(video_id: int, markdown_path: Optional[str] = None,
                        json_path: Optional[str] = None, srt_path: Optional[str] = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT output_markdown_path, output_json_path, output_srt_path FROM videos WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        if row:
            m = markdown_path if markdown_path else row["output_markdown_path"]
            j = json_path if json_path else row["output_json_path"]
            s = srt_path if srt_path else row["output_srt_path"]
            cursor.execute("""
                UPDATE videos
                SET output_markdown_path = ?, output_json_path = ?, output_srt_path = ?, updated_at = ?
                WHERE id = ?
            """, (m, j, s, datetime.now().isoformat(), video_id))
            conn.commit()


def get_video_info(video_id: int) -> Optional[dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT v.*,
                   (SELECT COUNT(*) FROM transcript_segments WHERE video_id = v.id) as transcript_count,
                   (SELECT COUNT(*) FROM summary_chunks WHERE video_id = v.id) as chunk_count,
                   (SELECT COUNT(*) FROM final_summaries WHERE video_id = v.id) as has_final_summary
            FROM videos v WHERE v.id = ?
        """, (video_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def has_transcript_segments(video_id: int) -> bool:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transcript_segments WHERE video_id = ?", (video_id,))
        return cursor.fetchone()[0] > 0


def has_summary_chunks(video_id: int) -> bool:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM summary_chunks WHERE video_id = ?", (video_id,))
        return cursor.fetchone()[0] > 0


def has_final_summary(video_id: int) -> bool:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM final_summaries WHERE video_id = ?", (video_id,))
        return cursor.fetchone()[0] > 0


def find_video_by_source(conn: sqlite3.Connection, *sources: str) -> Optional[dict]:
    """Find an existing video record by one of its source identities.

    `sources` are candidate identity strings for the same video, e.g. the raw
    user input plus its normalized form (absolute local path or normalized
    URL). The query is parameterized; the oldest matching record (lowest id)
    wins. Callers pass a connection taken from `get_db()` so the lookup can be
    combined with other work in the same transaction.
    """
    candidates = [s for s in sources if s]
    if not candidates:
        return None
    cursor = conn.cursor()
    placeholders = ", ".join("?" for _ in candidates)
    cursor.execute(
        f"""
        SELECT * FROM videos
        WHERE source_path IN ({placeholders}) OR url IN ({placeholders})
        ORDER BY id
        LIMIT 1
        """,
        candidates + candidates,
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def clear_transcript_segments(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transcript_segments WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_summary_chunks(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM summary_chunks WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_final_summary(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM final_summaries WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_video_quotes(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM video_quotes WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_video_chapters(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM video_chapters WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_video_terms(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM video_terms WHERE video_id = ?", (video_id,))
        conn.commit()


def clear_summary_outputs(video_id: int):
    """Delete all summary-stage derived rows (chunk summaries, final summary,
    quotes, chapters, terms) so a forced re-run cannot accumulate duplicates."""
    clear_summary_chunks(video_id)
    clear_final_summary(video_id)
    clear_video_quotes(video_id)
    clear_video_chapters(video_id)
    clear_video_terms(video_id)


def clear_video_outputs(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE videos
            SET output_markdown_path = NULL, output_json_path = NULL, output_srt_path = NULL,
                current_stage = 'created', failed_stage = NULL, last_error = NULL,
                status = 'pending', updated_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), video_id))
        conn.commit()


def clear_video_work_data(video_id: int):
    """Delete every derived row (transcript + summaries) for a video so a full
    re-run starts clean. Used by flows that always re-execute all stages."""
    clear_transcript_segments(video_id)
    clear_summary_outputs(video_id)
    clear_video_outputs(video_id)


def get_all_videos():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def create_video_record(
    source_type: str,
    source_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    duration: Optional[float] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (source_type, source_path, url, title, author, duration, status, current_stage)
            VALUES (?, ?, ?, ?, ?, ?, 'processing', 'created')
        """, (source_type, source_path, url, title, author, duration))
        return cursor.lastrowid


def update_video_duration(video_id: int, duration: float):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE videos SET duration = ?, updated_at = ? WHERE id = ?",
            (duration, datetime.now().isoformat(), video_id)
        )
        conn.commit()
