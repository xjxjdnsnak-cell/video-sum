import json
from pathlib import Path
from typing import List, Optional, Dict, Union
from datetime import datetime

from ..utils.filename import sanitize_filename


def export_json(
    video_id: int,
    video_title: str,
    video_url: Optional[str],
    video_author: Optional[str],
    duration: Optional[float],
    transcript: List[Dict],
    chunk_summaries: List[Dict],
    final_summary: Optional[Dict],
    chapters: List[Dict] = None,
    quotes: List[Dict] = None,
    terms: List[Dict] = None,
    note_style: str = "detailed",
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None
) -> str:
    if chapters is None:
        chapters = []
    if quotes is None:
        quotes = []
    if terms is None:
        terms = []

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_filename:
            output_path = output_dir / output_filename
        else:
            safe_title = sanitize_filename(video_title)
            output_path = output_dir / f"{safe_title}.json"
    else:
        from ..config import settings
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = sanitize_filename(video_title)
        output_path = settings.OUTPUT_DIR / f"{safe_title}.json"

    data = {
        "video_id": video_id,
        "title": video_title,
        "url": video_url,
        "author": video_author,
        "duration": duration,
        "note_style": note_style,
        "exported_at": datetime.now().isoformat(),
        "transcript": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "start_time": seg.get("start_time", ""),
                "end_time": seg.get("end_time", ""),
                "text": seg["text"],
                "source": seg.get("source", "asr")
            }
            for seg in transcript
        ],
        "chunk_summaries": [
            {
                "start": chunk["start"],
                "end": chunk["end"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "topic": chunk.get("topic", ""),
                "key_points": chunk.get("key_points", []),
                "important_terms": chunk.get("important_terms", []),
                "quote": chunk.get("quote", ""),
                "chapter_hint": chunk.get("chapter_hint", ""),
                "summary": chunk.get("summary", ""),
                "source_text": chunk.get("source_text", "")[:500]
            }
            for chunk in chunk_summaries
        ],
        "chapters": [
            {
                "title": chapter.get("title", ""),
                "start_time": chapter.get("start_time", ""),
                "end_time": chapter.get("end_time", ""),
                "chunks": chapter.get("chunks", []),
                "summary": chapter.get("summary", "")
            }
            for chapter in chapters
        ],
        "quotes": [
            {
                "text": quote.get("text", ""),
                "start_time": quote.get("start_time", ""),
                "end_time": quote.get("end_time", "")
            }
            for quote in quotes
        ],
        "terms": [
            {
                "term": term.get("term", ""),
                "explanation": term.get("explanation", ""),
                "first_seen_time": term.get("first_seen_time", "")
            }
            for term in terms
        ],
        "final_summary": final_summary
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(output_path)
