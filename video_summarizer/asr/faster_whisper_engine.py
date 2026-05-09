from typing import List, Optional
from dataclasses import dataclass
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


class MockASREngine:
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[Segment]:
        console.print("[yellow]Using mock transcription (Whisper model not available)[/yellow]")
        return [
            Segment(start=0.0, end=5.0, text="[Mock转写] 这是第一段测试音频内容的转写文本。"),
            Segment(start=5.0, end=10.0, text="[Mock转写] 这是第二段测试音频内容的转写文本，包含了一些测试内容。"),
        ]


class FasterWhisperEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        use_mock: bool = False
    ):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.use_mock = use_mock
        self._model = None

    @property
    def model(self):
        if self.use_mock:
            return MockASREngine()

        if self._model is None:
            console.print(f"[dim]Loading Whisper model: {self.model_name} ({self.device})[/dim]")
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load Whisper model: {e}[/yellow]")
                console.print("[yellow]Falling back to mock transcription[/yellow]")
                self.use_mock = True
                return MockASREngine()
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
            if self.use_mock or self._model is None:
                engine = self.model
                return engine.transcribe(audio_path, language=language)

            segments, info = self._model.transcribe(
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
            console.print(f"[yellow]Transcription failed with Whisper: {e}[/yellow]")
            console.print("[yellow]Falling back to mock transcription[/yellow]")
            return MockASREngine().transcribe(audio_path, language=language)
