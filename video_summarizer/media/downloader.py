import subprocess
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from rich.console import Console

from ..config import settings

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
    video_id: Optional[str]
    webpage_url: Optional[str]
    page_count: int = 1
    requires_login: bool = False


def check_ytdlp_installed() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_bilibili_url(url: str) -> bool:
    patterns = [
        r"^BV[0-9A-Za-z]+$",
        r"^av\d+$",
        r"bilibili\.com",
        r"b23\.tv",
    ]
    for pattern in patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def normalize_bilibili_url(url_or_id: str) -> str:
    url = url_or_id.strip()
    
    if re.match(r"^BV[0-9A-Za-z]+$", url, re.IGNORECASE):
        return f"https://www.bilibili.com/video/{url}"
    
    if re.match(r"^av\d+$", url, re.IGNORECASE):
        return f"https://www.bilibili.com/video/{url}"
    
    if url.startswith("//"):
        return f"https:{url}"
    
    if url.startswith("/video/"):
        return f"https://www.bilibili.com{url}"
    
    return url


def build_ytdlp_args(
    url: str,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    extra_args: List[str] = None
) -> List[str]:
    args = ["--no-warnings"]

    # Fall back to the globally configured cookie file (DOWNLOAD_COOKIES in
    # .env) so both the CLI and the Web UI pick it up automatically.
    if not cookies_file and not cookies_from_browser:
        cookies_file = settings.DOWNLOAD_COOKIES or None

    if user_agent:
        args.extend(["--user-agent", user_agent])
    else:
        args.extend([
            "--user-agent", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ])

    if cookies_file:
        cookies_path = Path(cookies_file).expanduser()
        if cookies_path.exists():
            args.extend(["--cookies", str(cookies_path)])
        else:
            raise DownloaderError(f"Cookie file not found: {cookies_file}")
    
    if cookies_from_browser:
        valid_browsers = ["chrome", "firefox", "edge", "brave", "safari"]
        if cookies_from_browser.lower() not in valid_browsers:
            raise DownloaderError(f"Invalid browser: {cookies_from_browser}. Valid options: {', '.join(valid_browsers)}")
        args.extend(["--cookies-from-browser", cookies_from_browser])

    if proxy:
        args.extend(["--proxy", proxy])

    if extra_args:
        args.extend(extra_args)

    return args


def get_video_info(
    url: str,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None
) -> VideoInfo:
    if not check_ytdlp_installed():
        raise DownloaderError(
            "yt-dlp 未安装。请安装：pip install yt-dlp"
        )

    normalized_url = normalize_bilibili_url(url)
    
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
    ] + build_ytdlp_args(normalized_url, cookies_file, cookies_from_browser, proxy, user_agent) + [normalized_url]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr.strip()
        handle_ytdlp_error(error_msg)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise DownloaderError(f"解析视频信息失败: {e}")

    subtitles = data.get("subtitles", {}) or {}
    auto_subs = data.get("automatic_captions", {}) or {}
    
    requires_login = False
    if "requires_login" in data and data["requires_login"]:
        requires_login = True
    elif "access denied" in (data.get("error", "") or "").lower():
        requires_login = True

    return VideoInfo(
        title=data.get("title", "未知标题"),
        author=data.get("uploader") or data.get("channel", None),
        duration=float(data.get("duration", 0)),
        thumbnail=data.get("thumbnail", None),
        subtitles={**subtitles, **auto_subs},
        video_id=data.get("id", None),
        webpage_url=data.get("webpage_url", None),
        page_count=data.get("playlist_count", 1) if "playlist_count" in data else 1,
        requires_login=requires_login
    )


def handle_ytdlp_error(error_msg: str):
    if "HTTP Error 412" in error_msg or "Precondition Failed" in error_msg:
        raise DownloaderError(
            f"B站访问受限 (HTTP 412)。可能需要登录或设置Cookie。\n\n"
            f"解决方案：\n"
            f"1. 在浏览器中登录B站账号\n"
            f"2. 导出Cookie为Netscape格式\n"
            f"3. 使用命令参数：--cookies path/to/cookies.txt\n"
            f"或者使用：--cookies-from-browser chrome/firefox/edge"
        )
    elif "HTTP Error 403" in error_msg:
        raise DownloaderError(
            f"B站访问被拒绝 (HTTP 403)。\n\n"
            f"可能原因：\n"
            f"- 视频需要登录才能观看\n"
            f"- 视频已被删除或私密\n"
            f"- 地区限制\n\n"
            f"建议：\n"
            f"1. 尝试使用 --cookies 参数提供登录Cookie\n"
            f"2. 检查视频链接是否有效"
        )
    elif "Video unavailable" in error_msg or "This video is unavailable" in error_msg:
        raise DownloaderError(
            "视频不存在或已被删除。\n\n"
            "请检查链接是否正确，或视频是否已被UP主删除。"
        )
    elif "geographic restriction" in error_msg.lower() or "region" in error_msg.lower():
        raise DownloaderError(
            "视频存在地区限制。\n\n"
            "该视频可能仅限特定地区观看，建议使用代理或VPN。\n"
            "可尝试使用 --proxy 参数。"
        )
    elif "cookies" in error_msg.lower() and "invalid" in error_msg.lower():
        raise DownloaderError(
            "Cookie文件无效或格式不正确。\n\n"
            "请确保Cookie文件为Netscape格式，\n"
            "并包含B站相关的Cookie。"
        )
    elif "login" in error_msg.lower() or "sign in" in error_msg.lower():
        raise DownloaderError(
            "需要登录才能访问此视频。\n\n"
            "请使用 --cookies 或 --cookies-from-browser 参数提供登录凭证。"
        )
    raise DownloaderError(f"获取视频信息失败: {error_msg}")


def _pick_subtitle_file(
    output_dir: Path,
    languages: List[str]
) -> Optional[Path]:
    """Pick the best subtitle file yt-dlp produced, preferring the configured
    language order. yt-dlp names subtitle files "<id>.<lang>.srt" (the lang may
    carry a suffix like "-orig"); files for unlisted languages lose to any
    configured one and are ordered by name among themselves."""
    srt_files = sorted(output_dir.glob("*.srt"))
    if not srt_files:
        return None

    def lang_priority(path: Path):
        parts = path.stem.split(".")  # stem strips ".srt"
        file_lang = parts[-1] if len(parts) > 1 else ""
        for priority, lang in enumerate(languages):
            if file_lang == lang or file_lang.startswith(f"{lang}-") or file_lang.startswith(f"{lang}."):
                return (priority, path.name)
        return (len(languages), path.name)

    return min(srt_files, key=lang_priority)


def download_subtitles(
    url: str,
    output_dir: Path,
    languages: List[str] = None,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Optional[Path]:
    if languages is None:
        languages = ["zh-Hans", "zh-Hant", "zh", "en", "zh-CN"]

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_url = normalize_bilibili_url(url)

    video_id = None
    cmd_info = ["yt-dlp", "--get-id"] + build_ytdlp_args(
        normalized_url, cookies_file, cookies_from_browser, proxy, user_agent
    ) + [normalized_url]
    result = subprocess.run(cmd_info, capture_output=True, text=True)
    if result.returncode == 0:
        video_id = result.stdout.strip().split('\n')[0]

    # Single yt-dlp call for every configured language. The old per-language
    # loop re-fetched the page once per language (up to 5 subprocess calls,
    # each a multi-second network round-trip), which both wasted time and
    # raised the odds of B站's HTTP 412 rate limiting. One call with a joined
    # --sub-langs list fetches all available languages at once; we then pick
    # the best file by configured language order. If no subtitle file is
    # produced (none available, or yt-dlp failed) we return None, matching the
    # previous behavior of falling through the loop and returning None.
    cmd = [
        "yt-dlp",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", ",".join(languages),
        "--skip-download",
        "--convert-subs", "srt",
    ]

    if video_id:
        output_template = str(output_dir / f"{video_id}.%(ext)s")
        cmd.extend(["-o", output_template])
    else:
        cmd.extend(["-o", str(output_dir / "%(id)s.%(ext)s")])

    cmd.extend(build_ytdlp_args(normalized_url, cookies_file, cookies_from_browser, proxy, user_agent))
    cmd.append(normalized_url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    return _pick_subtitle_file(output_dir, languages)


def download_audio(
    url: str,
    output_path: Path,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Path:
    if not check_ytdlp_installed():
        raise DownloaderError("yt-dlp 未安装。请安装：pip install yt-dlp")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_url = normalize_bilibili_url(url)

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
    ] + build_ytdlp_args(normalized_url, cookies_file, cookies_from_browser, proxy, user_agent)
    cmd.append(normalized_url)

    console.print(f"[dim]正在下载音频: {normalized_url}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        if "HTTP Error 412" in error_msg or "HTTP Error 403" in error_msg:
            raise DownloaderError(
                f"B站下载失败。视频可能需要登录。\n\n"
                f"提示：请使用 --cookies 参数提供登录Cookie"
            )
        handle_ytdlp_error(error_msg)

    if not output_path.exists():
        raise DownloaderError(f"音频文件未创建: {output_path}")

    return output_path
