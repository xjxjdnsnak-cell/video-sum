import json
from pathlib import Path
from typing import List, Optional, Dict, Union
from datetime import datetime


def export_json(
    video_id: int,
    video_title: str,
    video_url: Optional[str],
    video_author: Optional[str],
    duration: Optional[float],
    transcript: List[Dict],
    chunk_summaries: List[Dict],
    final_summary: Optional[Dict],
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None
) -> str:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_filename:
            output_path = output_dir / output_filename
        else:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)[:50]
            output_path = output_dir / f"{safe_title}.json"
    else:
        from ..config import settings
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)[:50]
        output_path = settings.OUTPUT_DIR / f"{safe_title}.json"

    data = {
        "video_id": video_id,
        "title": video_title,
        "url": video_url,
        "author": video_author,
        "duration": duration,
        "exported_at": datetime.now().isoformat(),
        "transcript": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in transcript
        ],
        "chunk_summaries": [
            {
                "start": chunk["start"],
                "end": chunk["end"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "summary": chunk["summary"]
            }
            for chunk in chunk_summaries
        ],
        "final_summary": final_summary
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(output_path)
