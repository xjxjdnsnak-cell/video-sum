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
        console.print("[yellow][Mock ASR] Generating mock transcription[/yellow]")
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
        use_mock: bool = False,
        model_dir: Optional[str] = None,
        language: Optional[str] = None
    ):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = self._resolve_device(device or settings.WHISPER_DEVICE)
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.use_mock = use_mock
        self.model_dir = Path(model_dir) if model_dir else None
        self.language = language
        self._model = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    console.print("[dim]Auto-detected CUDA, using GPU[/dim]")
                    return "cuda"
            except ImportError:
                pass
            console.print("[dim]Auto-detected CPU[/dim]")
            return "cpu"
        return device

    @property
    def model(self):
        if self.use_mock:
            return MockASREngine()

        if self._model is None:
            console.print(f"[dim]Loading Whisper model: {self.model_name} ({self.device})[/dim]")
            if self.model_dir:
                console.print(f"[dim]Model directory: {self.model_dir}[/dim]")

            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.model_dir) if self.model_dir else None
                )
            except ImportError:
                raise FasterWhisperError(
                    f"faster-whisper is not installed. Run: pip install faster-whisper"
                )
            except Exception as e:
                raise FasterWhisperError(
                    f"Failed to load Whisper model '{self.model_name}'.\n"
                    f"Error: {e}\n\n"
                    f"解决方案:\n"
                    f"1. 检查网络连接\n"
                    f"2. 预下载模型: video-summarizer download-model --model {self.model_name}\n"
                    f"3. 尝试更小的模型: --model tiny\n"
                    f"4. 或使用 Mock ASR 测试流程: --asr-provider mock"
                )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> List[Segment]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = language or self.language or settings.SRT_LANGUAGE

        if self.use_mock or self._model is None:
            engine = self.model
            return engine.transcribe(audio_path, language=language)

        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
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
            raise FasterWhisperError(
                f"Transcription failed: {e}\n\n"
                f"如果需要使用 Mock 转写进行测试，请添加参数: --asr-provider mock"
            )
