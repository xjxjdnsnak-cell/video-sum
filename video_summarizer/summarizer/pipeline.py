from typing import List, Optional, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import settings
from ..models import Video, TranscriptSegment, SummaryChunk, FinalSummary
from ..db import get_db
from ..utils.timefmt import format_timestamp
from .chunker import create_chunks_from_segments, merge_short_chunks, TextChunk
from .llm_client import get_llm_client, BaseLLMClient


console = Console()


class PipelineError(Exception):
    pass


def save_transcript(video_id: int, segments: List[Dict]) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        for seg in segments:
            cursor.execute("""
                INSERT INTO transcript_segments (video_id, start, end, text, source)
                VALUES (?, ?, ?, ?, ?)
            """, (video_id, seg["start"], seg["end"], seg["text"], seg.get("source", "asr")))
        return len(segments)


def get_transcript(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT start, end, text, source FROM transcript_segments
            WHERE video_id = ? ORDER BY start
        """, (video_id,))
        rows = cursor.fetchall()
        return [{"start": r["start"], "end": r["end"], "text": r["text"], "source": r["source"]} for r in rows]


def save_summary_chunks(video_id: int, chunks: List[Dict]) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO summary_chunks (video_id, start, end, source_text, summary)
                VALUES (?, ?, ?, ?, ?)
            """, (video_id, chunk["start"], chunk["end"], chunk["source_text"], chunk["summary"]))
        return len(chunks)


def get_summary_chunks(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT start, end, source_text, summary FROM summary_chunks
            WHERE video_id = ? ORDER BY start
        """, (video_id,))
        rows = cursor.fetchall()
        return [{"start": r["start"], "end": r["end"], "source_text": r["source_text"], "summary": r["summary"]} for r in rows]


def save_final_summary(video_id: int, summary: Dict) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO final_summaries 
            (video_id, one_sentence_summary, detailed_summary, key_points, questions)
            VALUES (?, ?, ?, ?, ?)
        """, (
            video_id,
            summary.get("one_sentence_summary", ""),
            summary.get("detailed_summary", ""),
            summary.get("key_points", ""),
            summary.get("questions", "")
        ))
        conn.commit()
        return cursor.lastrowid


def get_final_summary(video_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM final_summaries WHERE video_id = ?
        """, (video_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def update_video_status(video_id: int, status: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
        conn.commit()


def summarize_video_pipeline(
    video_id: int,
    llm_provider: Optional[str] = None,
    chunk_min: int = 3,
    chunk_max: int = 5
) -> Dict:
    update_video_status(video_id, "processing")
    console.print(f"[yellow]Starting summarization pipeline for video {video_id}...[/yellow]")

    try:
        segments = get_transcript(video_id)
        if not segments:
            raise PipelineError("No transcript found for this video")

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            video_title = row["title"] if row else "Unknown"

        llm_client = get_llm_client(llm_provider)

        console.print("[cyan]Creating text chunks...[/cyan]")
        chunks = create_chunks_from_segments(
            segments,
            min_duration=chunk_min * 60,
            max_duration=chunk_max * 60
        )
        chunks = merge_short_chunks(chunks, min_duration=60)
        console.print(f"[dim]Created {len(chunks)} chunks[/dim]")

        console.print("[cyan]Generating chunk summaries...[/cyan]")
        chunk_summaries = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Summarizing chunks...", total=len(chunks))
            for chunk in chunks:
                summary = llm_client.summarize_chunk(
                    text=chunk.text,
                    start_time=chunk.start_time_str,
                    end_time=chunk.end_time_str
                )
                chunk_summaries.append({
                    "start": chunk.start,
                    "end": chunk.end,
                    "start_time": chunk.start_time_str,
                    "end_time": chunk.end_time_str,
                    "source_text": chunk.text[:500],
                    "summary": summary
                })
                progress.update(task, advance=1)

        save_summary_chunks(video_id, chunk_summaries)

        console.print("[cyan]Generating final summary...[/cyan]")
        final_summary = llm_client.generate_final_summary(
            video_title=video_title,
            chunk_summaries=chunk_summaries
        )
        save_final_summary(video_id, final_summary)

        update_video_status(video_id, "completed")
        console.print("[green]Summarization complete![/green]")

        return {
            "video_id": video_id,
            "chunks": chunk_summaries,
            "final_summary": final_summary
        }

    except Exception as e:
        update_video_status(video_id, "failed")
        raise PipelineError(f"Pipeline failed: {e}")
