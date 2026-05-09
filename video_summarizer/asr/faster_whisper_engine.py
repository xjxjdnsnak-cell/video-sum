from typing import List, Optional
from dataclasses import dataclass
from faster_whisper import WhisperModel
from pathlib import Path
from rich.console import Console

from ..config import settings

console = Console()


@dataclass
class Segment:
    start: float
    end: float
    text: str


class FasterWhisperError(Exception):
    pass


class FasterWhisperEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self._model = None

    @property
    def model(self):
        if self._model is None:
            console.print(f"[dim]Loading Whisper model: {self.model_name} ({self.device})[/dim]")
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type
                )
            except Exception as e:
                raise FasterWhisperError(f"Failed to load Whisper model: {e}")
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> List[Segment]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = language or settings.SRT_LANGUAGE

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                beam_size=5,
                vad_filter=True
            )

            result = []
            for segment in segments:
                result.append(Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip()
                ))

            console.print(f"[green]Transcription complete: {info.language} ({info.language_probability:.2%})[/green]")
            return result

        except Exception as e:
            raise FasterWhisperError(f"Transcription failed: {e}")
