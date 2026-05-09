import subprocess
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
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


def _build_ytdlp_args(url: str, extra_args: List[str] = None) -> List[str]:
    default_args = [
        "--no-warnings",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    cookies_file = os.path.expanduser("~/.video_summarizer/cookies.txt")
    if os.path.exists(cookies_file):
        default_args.extend(["--cookies", cookies_file])

    if extra_args:
        default_args.extend(extra_args)

    return default_args


def get_video_info(url: str) -> VideoInfo:
    if not check_ytdlp_installed():
        raise DownloaderError("yt-dlp is not installed. Please install it: pip install yt-dlp")

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
    ] + _build_ytdlp_args(url) + [url]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr.strip()
        if "HTTP Error 412" in error_msg or "Precondition Failed" in error_msg:
            raise DownloaderError(
                f"B站访问受限 (HTTP 412)。可能需要登录或设置Cookie。\n"
                f"解决方案：\n"
                f"1. 在浏览器中登录B站\n"
                f"2. 导出Cookie为Netscape格式\n"
                f"3. 保存到 ~/.video_summarizer/cookies.txt\n"
                f"或者尝试使用 --cookies 参数"
            )
        elif "HTTP Error 403" in error_msg:
            raise DownloaderError(
                f"B站访问被拒绝 (HTTP 403)。视频可能需要登录才能观看。"
            )
        raise DownloaderError(f"Failed to get video info: {error_msg}")

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


def download_subtitles(url: str, output_dir: Path, languages: List[str] = None) -> Optional[Path]:
    if languages is None:
        languages = ["zh-Hans", "zh-Hant", "zh", "en", "zh-CN"]

    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = None
    cmd_info = ["yt-dlp", "--get-id"] + _build_ytdlp_args(url) + [url]
    result = subprocess.run(cmd_info, capture_output=True, text=True)
    if result.returncode == 0:
        video_id = result.stdout.strip().split('\n')[0]

    for lang in languages:
        cmd = [
            "yt-dlp",
            "--write-subs",
            "--sub-langs", lang,
            "--skip-download",
            "--convert-subs", "srt",
        ]

        if video_id:
            output_template = str(output_dir / f"{video_id}.%(ext)s")
            cmd.extend(["-o", output_template])
        else:
            cmd.extend(["-o", str(output_dir / "%(id)s.%(ext)s")])

        cmd.extend(_build_ytdlp_args(url))
        cmd.append(url)

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
    ] + _build_ytdlp_args(url)

    console.print(f"[dim]Downloading audio from: {url}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        if "HTTP Error 412" in error_msg or "HTTP Error 403" in error_msg:
            raise DownloaderError(
                f"B站下载失败。视频可能需要登录。\n"
                f"提示：请设置Cookie文件到 ~/.video_summarizer/cookies.txt"
            )
        raise DownloaderError(f"Failed to download audio: {error_msg}")

    if not output_path.exists():
        raise DownloaderError(f"Audio file was not created: {output_path}")

    return output_path
