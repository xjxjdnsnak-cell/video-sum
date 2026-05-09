import srt
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ..utils.timefmt import parse_timestamp


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


def parse_srt(srt_path: str) -> List[SubtitleSegment]:
    srt_path = Path(srt_path)
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        subs = list(srt.parse(content))
    except Exception as e:
        raise ValueError(f"Failed to parse SRT file: {e}")

    segments = []
    for sub in subs:
        segments.append(SubtitleSegment(
            start=sub.start.total_seconds(),
            end=sub.end.total_seconds(),
            text=sub.content.strip()
        ))

    return segments


def merge_subtitle_segments(
    segments: List[SubtitleSegment],
    max_gap: float = 2.0
) -> List[SubtitleSegment]:
    if not segments:
        return []

    merged = []
    current = SubtitleSegment(
        start=segments[0].start,
        end=segments[0].end,
        text=segments[0].text
    )

    for seg in segments[1:]:
        if seg.start - current.end <= max_gap:
            current.end = seg.end
            current.text += " " + seg.text
        else:
            merged.append(current)
            current = SubtitleSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text
            )

    merged.append(current)
    return merged
