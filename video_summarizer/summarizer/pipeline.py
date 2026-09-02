from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import settings
from ..db import get_db
from ..utils.timefmt import format_timestamp
from .chunker import create_chunks_from_segments, merge_short_chunks, TextChunk
from .llm_client import get_llm_client, BaseLLMClient, format_timestamp as llm_format_timestamp
from .prompts import NoteStyle


console = Console()

# Chunk summaries are independent LLM calls, so they run concurrently. 4
# workers is a good trade-off for provider rate limits (B站-style 429/412
# bursts); more workers rarely help and risk throttling.
MAX_CHUNK_SUMMARY_WORKERS = 4


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
        return [
            {
                "start": r["start"],
                "end": r["end"],
                # Timestamp strings mirror the shape produced by the summarize
                # path (TextChunk.start_time_str) so consumers like export_json
                # work identically whether chunks were just generated or loaded
                # from a previous run's checkpoint.
                "start_time": format_timestamp(r["start"]),
                "end_time": format_timestamp(r["end"]),
                "source_text": r["source_text"],
                "summary": r["summary"]
            }
            for r in rows
        ]


def save_final_summary(video_id: int, summary: Dict, note_style: NoteStyle = NoteStyle.DETAILED) -> int:
    import json
    
    with get_db() as conn:
        cursor = conn.cursor()
        extra_data = {}
        key_points_val = summary.get("key_points", [])
        questions_val = summary.get("review_questions", [])
        
        if isinstance(key_points_val, list):
            key_points_val = "\n".join(f"- {p}" for p in key_points_val)
        if isinstance(questions_val, list):
            questions_val = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions_val))
        
        for key, value in summary.items():
            if key not in ["one_sentence_summary", "detailed_summary", "key_points", "review_questions", "questions"]:
                if isinstance(value, (list, dict)):
                    extra_data[key] = json.dumps(value, ensure_ascii=False)
                else:
                    extra_data[key] = value
        
        cursor.execute("""
            INSERT OR REPLACE INTO final_summaries 
            (video_id, one_sentence_summary, detailed_summary, key_points, questions, extra_data, note_style)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            summary.get("one_sentence_summary", ""),
            summary.get("detailed_summary", summary.get("timeline_summary", "")),
            key_points_val,
            questions_val,
            json.dumps(extra_data, ensure_ascii=False) if extra_data else None,
            note_style.value
        ))
        conn.commit()
        return cursor.lastrowid


def get_final_summary(video_id: int) -> Optional[Dict]:
    import json
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM final_summaries WHERE video_id = ?
        """, (video_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result.get("extra_data"):
                try:
                    extra = json.loads(result["extra_data"])
                    result.update(extra)
                except json.JSONDecodeError:
                    pass
            return result
        return None


def save_chapters(video_id: int, chapters: List[Dict]) -> int:
    import json
    
    with get_db() as conn:
        cursor = conn.cursor()
        for chapter in chapters:
            cursor.execute("""
                INSERT INTO video_chapters (video_id, title, start_time, end_time, chunk_indices, summary)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                chapter.get("title", ""),
                chapter.get("start_time", ""),
                chapter.get("end_time", ""),
                json.dumps(chapter.get("chunks", [])),
                chapter.get("summary", "")
            ))
        return len(chapters)


def get_chapters(video_id: int) -> List[Dict]:
    import json
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM video_chapters WHERE video_id = ? ORDER BY start_time
        """, (video_id,))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            chapter = dict(r)
            if chapter.get("chunk_indices"):
                try:
                    chapter["chunks"] = json.loads(chapter["chunk_indices"])
                except json.JSONDecodeError:
                    chapter["chunks"] = []
            else:
                chapter["chunks"] = []
            result.append(chapter)
        return result


def save_quotes(video_id: int, quotes: List[Dict]) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        for quote in quotes:
            cursor.execute("""
                INSERT INTO video_quotes (video_id, text, start_time, end_time)
                VALUES (?, ?, ?, ?)
            """, (
                video_id,
                quote.get("text", ""),
                quote.get("start_time", ""),
                quote.get("end_time", "")
            ))
        return len(quotes)


def get_quotes(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM video_quotes WHERE video_id = ? ORDER BY start_time
        """, (video_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def save_terms(video_id: int, terms: List[Dict]) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        for term in terms:
            cursor.execute("""
                INSERT INTO video_terms (video_id, term, explanation, first_seen_time)
                VALUES (?, ?, ?, ?)
            """, (
                video_id,
                term.get("term", ""),
                term.get("explanation", ""),
                term.get("first_seen_time", "")
            ))
        return len(terms)


def get_terms(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM video_terms WHERE video_id = ? ORDER BY first_seen_time
        """, (video_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_video_status(video_id: int, status: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
        conn.commit()


def _summarize_single_chunk(llm_client: BaseLLMClient, chunk: TextChunk) -> Dict:
    """Summarize one chunk and shape it into the dict persisted by
    save_summary_chunks. Runs inside a worker thread; llm_client calls are
    thread-safe IO."""
    summary = llm_client.summarize_chunk(
        text=chunk.text,
        start_time=chunk.start_time_str,
        end_time=chunk.end_time_str
    )
    return {
        "start": chunk.start,
        "end": chunk.end,
        "start_time": chunk.start_time_str,
        "end_time": chunk.end_time_str,
        "source_text": chunk.text[:500],
        "topic": summary.get("topic", ""),
        "key_points": summary.get("key_points", []),
        "important_terms": summary.get("important_terms", []),
        "quote": summary.get("quote", ""),
        "chapter_hint": summary.get("chapter_hint", ""),
        "summary": summary.get("summary", "")
    }


def summarize_video_pipeline(
    video_id: int,
    llm_provider: Optional[str] = None,
    chunk_min: int = 3,
    chunk_max: int = 5,
    note_style: NoteStyle = NoteStyle.DETAILED
) -> Dict:
    update_video_status(video_id, "processing")
    console.print(f"[yellow]Starting summarization pipeline for video {video_id}...[/yellow]")
    console.print(f"[dim]Note style: {note_style.value}[/dim]")

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
        # Chunk summaries are independent: fan them out to a small thread pool.
        # Results are stored by chunk index so persistence order always follows
        # chunk order; progress advances in the main thread as futures complete
        # (rich Progress only ever touched from this thread).
        chunk_summaries: List = [None] * len(chunks)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Summarizing chunks...", total=len(chunks))
            max_workers = min(MAX_CHUNK_SUMMARY_WORKERS, len(chunks)) or 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(_summarize_single_chunk, llm_client, chunk): index
                    for index, chunk in enumerate(chunks)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    chunk_summaries[index] = future.result()
                    progress.update(task, advance=1)

        save_summary_chunks(video_id, chunk_summaries)

        console.print("[cyan]Extracting representative quotes...[/cyan]")
        quotes = llm_client.extract_quotes(segments)
        if quotes:
            save_quotes(video_id, quotes)
            console.print(f"[dim]Extracted {len(quotes)} quotes[/dim]")

        console.print("[cyan]Aggregating chapters...[/cyan]")
        chapters = llm_client.aggregate_chapters(chunk_summaries)
        if chapters:
            save_chapters(video_id, chapters)
            console.print(f"[dim]Created {len(chapters)} chapters[/dim]")

        console.print("[cyan]Generating final summary...[/cyan]")
        final_summary = llm_client.generate_final_summary(
            video_title=video_title,
            chunk_summaries=chunk_summaries,
            note_style=note_style
        )

        if note_style == NoteStyle.STUDY and final_summary:
            console.print("[cyan]Extracting terms...[/cyan]")
            terms = llm_client.extract_terms(
                video_title=video_title,
                transcript=segments,
                chapter_summaries=chunk_summaries
            )
            if terms:
                save_terms(video_id, terms)
                console.print(f"[dim]Extracted {len(terms)} terms[/dim]")

        save_final_summary(video_id, final_summary, note_style)

        update_video_status(video_id, "completed")
        console.print("[green]Summarization complete![/green]")

        return {
            "video_id": video_id,
            "chunks": chunk_summaries,
            "final_summary": final_summary,
            "chapters": chapters,
            "quotes": quotes,
            "note_style": note_style.value
        }

    except Exception as e:
        update_video_status(video_id, "failed")
        raise PipelineError(f"Pipeline failed: {e}")
