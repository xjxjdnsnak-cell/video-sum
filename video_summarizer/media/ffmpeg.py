import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


class FFmpegError(Exception):
    pass


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        str(output_path)
    ]

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise FFmpegError(f"FFmpeg failed: {result.stderr}")

    return str(output_path)


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"FFprobe failed: {result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise FFmpegError(f"Could not parse duration: {result.stdout}")


def check_ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
