from dataclasses import dataclass
from typing import List

from ..config import settings
from ..utils.timefmt import format_timestamp


@dataclass
class TextChunk:
    start: float
    end: float
    text: str

    @property
    def start_time_str(self) -> str:
        return format_timestamp(self.start)

    @property
    def end_time_str(self) -> str:
        return format_timestamp(self.end)

    @property
    def duration(self) -> float:
        return self.end - self.start


def create_chunks_from_segments(
    segments: List[dict],
    min_duration: int = None,
    max_duration: int = None
) -> List[TextChunk]:
    min_duration = min_duration or settings.CHUNK_DURATION_MIN * 60
    max_duration = max_duration or settings.CHUNK_DURATION_MAX * 60

    if not segments:
        return []

    chunks = []
    current_texts = []
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        current_texts.append(text)
        current_end = seg["end"]

        if current_end - current_start >= max_duration:
            if current_texts:
                chunks.append(TextChunk(
                    start=current_start,
                    end=current_end,
                    text=" ".join(current_texts)
                ))
            current_texts = []
            current_start = seg["end"]

        elif len(current_texts) >= 3 and current_end - current_start >= min_duration:
            chunks.append(TextChunk(
                start=current_start,
                end=current_end,
                text=" ".join(current_texts)
            ))
            current_texts = []
            current_start = seg["end"]

    if current_texts:
        chunks.append(TextChunk(
            start=current_start,
            end=current_end,
            text=" ".join(current_texts)
        ))

    return chunks


def merge_short_chunks(chunks: List[TextChunk], min_duration: float = 60) -> List[TextChunk]:
    if not chunks:
        return []

    merged = [chunks[0]]

    for chunk in chunks[1:]:
        if chunk.duration < min_duration and merged[-1].duration < min_duration * 2:
            last = merged.pop()
            merged.append(TextChunk(
                start=last.start,
                end=chunk.end,
                text=last.text + " " + chunk.text
            ))
        else:
            merged.append(chunk)

    return merged
