import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from rich.console import Console

console = Console()


class DownloaderError(Exception):
    pass


@dataclass
class VideoInfo:
    title: str
    author: Optional[str]
    duration: float
    thumbnail: Optional[str]
    subtitles: Dict[str, Any]


def check_ytdlp_installed() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(url: str) -> VideoInfo:
    if not check_ytdlp_installed():
        raise DownloaderError("yt-dlp is not installed. Please install it: pip install yt-dlp")

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-warnings",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloaderError(f"Failed to get video info: {result.stderr}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise DownloaderError(f"Failed to parse video info: {e}")

    subtitles = data.get("subtitles", {}) or {}
    auto_subs = data.get("automatic_captions", {}) or {}

    return VideoInfo(
        title=data.get("title", "Unknown Title"),
        author=data.get("uploader") or data.get("channel", None),
        duration=float(data.get("duration", 0)),
        thumbnail=data.get("thumbnail", None),
        subtitles={**subtitles, **auto_subs}
    )


def download_subtitles(url: str, output_dir: Path, languages: list[str] = None) -> Optional[Path]:
    if languages is None:
        languages = ["zh-Hans", "zh-Hant", "zh", "en"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        cmd = [
            "yt-dlp",
            "--write-subs",
            "--sub-langs", lang,
            "--skip-download",
            "--convert-subs", "srt",
            "-o", str(output_dir / "%(id)s.%(ext)s"),
            "--no-warnings",
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            srt_files = list(output_dir.glob("*.srt"))
            if srt_files:
                return srt_files[0]

    return None


def download_audio(url: str, output_path: Path) -> Path:
    if not check_ytdlp_installed():
        raise DownloaderError("yt-dlp is not installed. Please install it: pip install yt-dlp")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        "--no-warnings",
        url
    ]

    console.print(f"[dim]Downloading audio from: {url}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise DownloaderError(f"Failed to download audio: {result.stderr}")

    if not output_path.exists():
        raise DownloaderError(f"Audio file was not created: {output_path}")

    return output_path
