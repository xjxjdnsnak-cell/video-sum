from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal


@dataclass
class Video:
    source_type: Literal["local", "url"] = "local"
    source_path: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    duration: Optional[float] = None
    status: str = "pending"
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class TranscriptSegment:
    video_id: int
    start: float
    end: float
    text: str
    source: str = "asr"
    id: Optional[int] = None


@dataclass
class SummaryChunk:
    video_id: int
    start: float
    end: float
    source_text: str
    summary: str
    id: Optional[int] = None


@dataclass
class FinalSummary:
    video_id: int
    one_sentence_summary: str
    detailed_summary: str
    key_points: str
    questions: str
    created_at: Optional[datetime] = None
    id: Optional[int] = None
