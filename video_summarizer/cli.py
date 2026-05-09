import sys
import os
import logging
from pathlib import Path
from typing import Optional
import tempfile

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .db import (
    get_db, init_db, setup_logging,
    update_video_stage, update_video_status, update_video_outputs,
    get_video_info, has_transcript_segments, has_summary_chunks, has_final_summary,
    clear_transcript_segments, clear_summary_chunks, clear_final_summary, clear_video_outputs,
    get_all_videos
)
from .media.ffmpeg import extract_audio, get_video_duration, check_ffmpeg_installed, FFmpegError
from .media.downloader import (
    download_audio, download_subtitles, get_video_info as get_url_info, 
    check_ytdlp_installed, DownloaderError, is_bilibili_url, normalize_bilibili_url
)
from .asr.faster_whisper_engine import FasterWhisperEngine, FasterWhisperError, Segment
from .asr.subtitle_parser import parse_srt
from .summarizer.pipeline import summarize_video_pipeline, get_transcript, get_summary_chunks, get_final_summary, save_transcript, get_chapters, get_quotes, get_terms
from .summarizer.prompts import NoteStyle
from .exporters.markdown import export_markdown
from .exporters.srt import export_srt
from .exporters.json_exporter import export_json

app = typer.Typer(
    name="video-summarizer",
    help="B站/本地视频总结器 - 自动提取音频/字幕，转成文字并生成摘要"
)
console = Console()
logger = logging.getLogger(__name__)


def check_dependencies(require_asr: bool = True):
    errors = []
    if not check_ffmpeg_installed():
        errors.append("FFmpeg is not installed. Please install FFmpeg first.")
    if not check_ytdlp_installed():
        errors.append("yt-dlp is not installed. Run: pip install yt-dlp")
    if require_asr:
        try:
            import faster_whisper
        except ImportError:
            errors.append("faster-whisper is not installed. Run: pip install faster-whisper")

    if errors:
        for err in errors:
            console.print(f"[red]Error: {err}[/red]")
        raise typer.Exit(code=1)


@app.callback()
def main():
    log_file = setup_logging()
    init_db()
    logger.info(f"Command started. Log file: {log_file}")


@app.command("doctor")
def doctor():
    """检查环境配置"""
    console.print(Panel("[bold cyan]Environment Diagnostic Report[/bold cyan]", expand=False))
    console.print()

    checks = []

    checks.append(("Python Version", sys.version.split()[0], True))

    try:
        import faster_whisper
        checks.append(("faster-whisper", "installed", True))
    except ImportError:
        checks.append(("faster-whisper", "NOT installed", False))

    checks.append(("FFmpeg", "installed" if check_ffmpeg_installed() else "NOT installed",
                   check_ffmpeg_installed()))
    checks.append(("yt-dlp", "installed" if check_ytdlp_installed() else "NOT installed",
                   check_ytdlp_installed()))

    db_dir = settings.DB_PATH.parent
    db_writable = os.access(str(db_dir), os.W_OK)
    checks.append(("DB Directory", f"{db_dir} ({'writable' if db_writable else 'NOT writable'})", db_writable))

    output_writable = os.access(str(settings.OUTPUT_DIR), os.W_OK)
    checks.append(("Output Directory", f"{settings.OUTPUT_DIR} ({'writable' if output_writable else 'NOT writable'})",
                   output_writable))

    checks.append(("LLM Provider", settings.LLM_PROVIDER, True))
    checks.append(("Whisper Model", settings.WHISPER_MODEL, True))
    checks.append(("Whisper Device", settings.WHISPER_DEVICE, True))

    from .db import get_all_videos
    videos = get_all_videos()
    checks.append(("Total Videos", str(len(videos)), True))

    console.print("[bold]System Checks:[/bold]")
    for name, value, ok in checks:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {status} {name}: [dim]{value}[/dim]")

    all_ok = all(ok for _, _, ok in checks)
    console.print()
    if all_ok:
        console.print("[bold green]All checks passed![/bold green]")
    else:
        console.print("[bold yellow]Some checks failed. Please fix the issues above.[/bold yellow]")
        raise typer.Exit(code=1)


@app.command("web")
def web():
    """启动 Web UI (Streamlit)"""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "web_ui" / "app.py"),
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])


@app.command("inspect-url")
def inspect_url(
    url: str = typer.Argument(..., help="B站视频链接或BV号"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="Cookie文件路径"),
    cookies_from_browser: Optional[str] = typer.Option(None, "--cookies-from-browser", 
                                                      help="从浏览器导入Cookie: chrome/firefox/edge"),
    proxy: Optional[str] = typer.Option(None, "--proxy", help="代理服务器地址"),
    user_agent: Optional[str] = typer.Option(None, "--user-agent", help="自定义User-Agent"),
):
    if not is_bilibili_url(url):
        console.print("[yellow]警告：这可能不是一个有效的B站链接[/yellow]")
    
    normalized_url = normalize_bilibili_url(url)
    console.print(f"[cyan]正在检查: {normalized_url}[/cyan]")
    console.print()

    try:
        info = get_url_info(
            url,
            cookies_file=cookies,
            cookies_from_browser=cookies_from_browser,
            proxy=proxy,
            user_agent=user_agent
        )
    except DownloaderError as e:
        console.print(f"[red]获取视频信息失败: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(Panel("[bold cyan]视频信息[/bold cyan]", expand=False))
    console.print()

    console.print(f"[bold]标题:[/bold] {info.title}")
    console.print(f"[bold]UP主:[/bold] {info.author or '未知'}")
    console.print(f"[bold]时长:[/bold] {int(info.duration // 60)}分{int(info.duration % 60)}秒")
    console.print(f"[bold]视频ID:[/bold] {info.video_id or '未知'}")
    console.print(f"[bold]分P数量:[/bold] {info.page_count}")
    console.print(f"[bold]需要登录:[/bold] {'[red]是[/red]' if info.requires_login else '[green]否[/green]'}")
    
    console.print()
    console.print(f"[bold]字幕信息:[/bold]")
    if info.subtitles:
        console.print(f"  [green]✓[/green] 存在字幕")
        console.print(f"  可用语言: {', '.join(info.subtitles.keys())}")
    else:
        console.print(f"  [yellow]✗[/yellow] 无字幕")
        console.print(f"  将使用ASR进行语音转写")

    console.print()
    console.print(f"[bold]推荐命令:[/bold]")
    cmd_parts = ["video-summarizer summarize-url", f"'{url}'", "--llm-provider mock"]
    
    if cookies:
        cmd_parts.append(f"--cookies {cookies}")
    elif info.requires_login:
        console.print(f"[dim]视频需要登录，建议添加 --cookies 参数[/dim]")
    
    console.print(f"[green]{' '.join(cmd_parts)}[/green]")


@app.command("status")
def status(video_id: int = typer.Argument(..., help="视频ID")):
    info = get_video_info(video_id)
    if not info:
        console.print(f"[red]Video {video_id} not found[/red]")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]Video #{video_id} Status[/bold cyan]", expand=False))
    console.print()

    console.print(f"[bold]Basic Info:[/bold]")
    console.print(f"  Title: {info.get('title', 'N/A')}")
    console.print(f"  Source: {info.get('source_type', 'N/A')}")
    if info.get('source_path'):
        console.print(f"  Path: {info.get('source_path')}")
    if info.get('url'):
        console.print(f"  URL: {info.get('url')}")
    console.print(f"  Duration: {info.get('duration', 'N/A')}s")
    console.print()

    status_color = {
        'pending': 'yellow',
        'processing': 'cyan',
        'transcribed': 'blue',
        'completed': 'green',
        'failed': 'red'
    }.get(info.get('status', ''), 'yellow')

    console.print(f"[bold]Pipeline Status:[/bold]")
    console.print(f"  Overall Status: [{status_color}]{info.get('status', 'unknown')}[/{status_color}]")
    console.print(f"  Current Stage: {info.get('current_stage', 'unknown')}")
    if info.get('failed_stage'):
        console.print(f"  Failed Stage: [red]{info.get('failed_stage')}[/red]")
    if info.get('last_error'):
        console.print(f"  Last Error: [red]{info.get('last_error')[:100]}...[/red]" if len(str(info.get('last_error', ''))) > 100 else f"  Last Error: [red]{info.get('last_error')}[/red]")
    console.print()

    console.print(f"[bold]Content:[/bold]")
    console.print(f"  Transcript Segments: {info.get('transcript_count', 0)}")
    console.print(f"  Summary Chunks: {info.get('chunk_count', 0)}")
    console.print(f"  Has Final Summary: {'[green]Yes[/green]' if info.get('has_final_summary') else '[yellow]No[/yellow]'}")
    console.print()

    console.print(f"[bold]Output Files:[/bold]")
    md_path = info.get('output_markdown_path')
    json_path = info.get('output_json_path')
    srt_path = info.get('output_srt_path')
    console.print(f"  Markdown: {md_path if md_path else '[yellow]Not exported[/yellow]'}")
    console.print(f"  JSON: {json_path if json_path else '[yellow]Not exported[/yellow]'}")
    console.print(f"  SRT: {srt_path if srt_path else '[yellow]Not exported[/yellow]'}")
    console.print()

    console.print(f"[bold]Timestamps:[/bold]")
    console.print(f"  Created: {info.get('created_at', 'N/A')}")
    console.print(f"  Updated: {info.get('updated_at', 'N/A')}")


@app.command("clean")
def clean(
    temp_only: bool = typer.Option(False, "--temp-only", help="Only delete temporary audio/cache files"),
    video_id: Optional[int] = typer.Option(None, "--video-id", help="Clean specific video's data"),
    all_cache: bool = typer.Option(False, "--all-cache", help="Delete all cached models and temp files"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Show what would be deleted without actually deleting"),
    yes: bool = typer.Option(False, "--yes", help="Actually perform the deletion"),
):
    if dry_run and not yes:
        console.print("[yellow]Running in dry-run mode. Use --yes to actually delete.[/yellow]")
        console.print()

    files_to_delete = []
    dirs_to_delete = []

    if all_cache:
        cache_dirs = [
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "torch",
            settings.OUTPUT_DIR / "logs",
        ]
        for d in cache_dirs:
            if d.exists():
                dirs_to_delete.append(d)
        temp_audio_files = list(Path(tempfile.gettempdir()).glob("video_summarizer_*.wav"))
        files_to_delete.extend(temp_audio_files)

    if video_id:
        info = get_video_info(video_id)
        if info:
            for path_key in ['output_markdown_path', 'output_json_path', 'output_srt_path']:
                path = info.get(path_key)
                if path and Path(path).exists():
                    files_to_delete.append(Path(path))

    if temp_only:
        temp_audio_files = list(Path(tempfile.gettempdir()).glob("video_summarizer_*.wav"))
        files_to_delete.extend(temp_audio_files)
        log_files = list(settings.OUTPUT_DIR.glob("logs/run-*.log")) if settings.OUTPUT_DIR.exists() else []
        files_to_delete.extend(log_files)

    if not files_to_delete and not dirs_to_delete:
        console.print("[green]Nothing to delete.[/green]")
        return

    console.print("[bold]Files to delete:[/bold]")
    for f in files_to_delete:
        console.print(f"  [red]{f}[/red]")
    for d in dirs_to_delete:
        console.print(f"  [red]{d}/[/red]")

    console.print()

    if dry_run or not yes:
        console.print("[yellow]Dry run - no files were deleted.[/yellow]")
        if not yes:
            console.print("[dim]Run with --yes to actually delete.[/dim]")
    else:
        for f in files_to_delete:
            try:
                f.unlink()
                console.print(f"[green]Deleted: {f}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to delete {f}: {e}[/red]")

        for d in dirs_to_delete:
            try:
                import shutil
                shutil.rmtree(d)
                console.print(f"[green]Deleted: {d}/[/green]")
            except Exception as e:
                console.print(f"[red]Failed to delete {d}: {e}[/red]")

        console.print()
        console.print("[green]Cleanup complete![/green]")


@app.command("download-model")
def download_model(
    model: str = typer.Option("base", "--model", "-m", help="Whisper model: tiny, base, small, medium, large"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Local model cache directory"),
):
    console.print(f"[cyan]Downloading Whisper model: {model}[/cyan]")
    if model_dir:
        console.print(f"[cyan]Model directory: {model_dir}[/cyan]")

    try:
        engine = FasterWhisperEngine(
            model_name=model,
            model_dir=model_dir
        )
        _ = engine.model
        console.print(f"[green]Model {model} downloaded successfully![/green]")
        if model_dir:
            console.print(f"[green]Saved to: {model_dir}[/green]")
    except FasterWhisperError as e:
        console.print(f"[red]ASR error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Failed: {e}[/red]")
        raise typer.Exit(code=1)


def _run_summarize(
    video_path_or_url: str,
    is_url: bool,
    llm_provider: str,
    asr_provider: str,
    chunk_min: int,
    chunk_max: int,
    output: Optional[str],
    model: str,
    device: str,
    language: Optional[str],
    model_dir: Optional[str],
    keep_audio: bool,
    force: bool,
    resume: bool = True,
    cookies: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    download_subtitle_only: bool = False,
    download_audio_only: bool = False,
    note_style: NoteStyle = NoteStyle.DETAILED,
):
    use_mock_asr = asr_provider.lower() == "mock"
    check_dependencies(require_asr=not use_mock_asr)

    output_dir = Path(output) if output else settings.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if is_url:
            console.print(f"[cyan]Processing URL: {video_path_or_url}[/cyan]")
            if use_mock_asr:
                console.print(f"[yellow]Using Mock ASR (--asr-provider mock)[/yellow]")

            console.print("[cyan]Fetching video info...[/cyan]")
            try:
                info = get_url_info(
                    video_path_or_url,
                    cookies_file=cookies,
                    cookies_from_browser=cookies_from_browser,
                    proxy=proxy,
                    user_agent=user_agent
                )
            except DownloaderError as e:
                console.print(f"[red]yt-dlp error: {e}[/red]")
                raise typer.Exit(code=1)

            video_id = create_video_record(
                source_type="url",
                url=video_path_or_url,
                title=info.title,
                author=info.author,
                duration=info.duration
            )
            logger.info(f"Created video record {video_id} for URL: {video_path_or_url}")

            if info.requires_login and not cookies and not cookies_from_browser:
                console.print("[yellow]警告：视频可能需要登录才能访问。建议使用 --cookies 参数。[/yellow]")

        else:
            video_path = Path(video_path_or_url)
            if not video_path.exists():
                console.print(f"[red]Error: Video file not found: {video_path}[/red]")
                raise typer.Exit(code=1)

            console.print(f"[cyan]Processing local video: {video_path.name}[/cyan]")
            if use_mock_asr:
                console.print(f"[yellow]Using Mock ASR (--asr-provider mock)[/yellow]")

            video_id = create_video_record(
                source_type="local",
                source_path=str(video_path),
                title=video_path.stem
            )
            logger.info(f"Created video record {video_id} for file: {video_path}")

        update_video_status(video_id, "processing", "created")
        logger.info(f"Starting pipeline for video {video_id}")

        duration = None
        audio_path = None
        transcript_data = None

        if is_url:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                has_subtitles = False
                if not download_audio_only:
                    console.print("[cyan]Checking for subtitles...[/cyan]")
                    srt_path = download_subtitles(
                        video_path_or_url, temp_path,
                        cookies_file=cookies,
                        cookies_from_browser=cookies_from_browser,
                        proxy=proxy,
                        user_agent=user_agent
                    )
                    if srt_path and srt_path.exists():
                        console.print(f"[green]✓ 找到字幕: {srt_path}[/green]")
                        logger.info(f"Found subtitles for video {video_id}")
                        console.print("[cyan]Parsing subtitles...[/cyan]")
                        subtitle_segments = parse_srt(str(srt_path))
                        transcript_data = [
                            {"start": s.start, "end": s.end, "text": s.text, "source": "subtitle"}
                            for s in subtitle_segments
                        ]
                        save_transcript(video_id, transcript_data)
                        update_video_stage(video_id, "transcribed")
                        has_subtitles = True
                    else:
                        console.print("[yellow]✗ 未找到字幕，将使用ASR[/yellow]")
                        logger.info(f"No subtitles found for video {video_id}")

                if not has_subtitles and not download_subtitle_only:
                    console.print("[cyan]Downloading audio for ASR...[/cyan]")
                    audio_path = temp_path / "audio.wav"
                    try:
                        download_audio(
                            video_path_or_url, audio_path,
                            cookies_file=cookies,
                            cookies_from_browser=cookies_from_browser,
                            proxy=proxy,
                            user_agent=user_agent
                        )
                    except DownloaderError as e:
                        console.print(f"[red]yt-dlp error: {e}[/red]")
                        update_video_stage(video_id, "transcribed", str(e))
                        update_video_status(video_id, "failed")
                        raise typer.Exit(code=1)

                    transcript_data = _do_transcribe(
                        video_id, str(audio_path), use_mock_asr, model, device, language, model_dir, force, resume
                    )

                if download_subtitle_only and not has_subtitles:
                    console.print("[red]错误：未找到字幕且指定了 --download-subtitle-only[/red]")
                    raise typer.Exit(code=1)

        else:
            duration = get_video_duration(video_path_or_url)
            update_video_duration(video_id, duration)

            audio_path = Path(tempfile.gettempdir()) / f"video_summarizer_{video_id}.wav"
            if Path(audio_path).exists() and not force:
                console.print(f"[green]Using cached audio: {audio_path}[/green]")
                logger.info(f"Using cached audio for video {video_id}")
            else:
                console.print("[cyan]Extracting audio...[/cyan]")
                logger.info(f"Extracting audio for video {video_id}")
                try:
                    extract_audio(video_path_or_url, str(audio_path))
                except FFmpegError as e:
                    console.print(f"[red]FFmpeg error: {e}[/red]")
                    update_video_stage(video_id, "audio_extracted", str(e))
                    update_video_status(video_id, "failed")
                    raise typer.Exit(code=1)

            transcript_data = _do_transcribe(
                video_id, str(audio_path), use_mock_asr, model, device, language, model_dir, force, resume
            )

        if audio_path and not keep_audio and not is_url:
            audio_path.unlink(missing_ok=True)

        if download_subtitle_only or download_audio_only:
            console.print(f"[green]字幕提取完成! Video ID: {video_id}[/green]")
            return

        if has_summary_chunks(video_id) and not force:
            console.print(f"[green]Using existing summary chunks for video {video_id}[/green]")
            logger.info(f"Using existing summary chunks for video {video_id}")
            result_chunks = get_summary_chunks(video_id)
        else:
            console.print("[cyan]Summarizing video...[/cyan]")
            logger.info(f"Generating summaries for video {video_id}")
            update_video_stage(video_id, "chunked")
            result = summarize_video_pipeline(
                video_id,
                llm_provider=llm_provider,
                chunk_min=chunk_min,
                chunk_max=chunk_max,
                note_style=note_style
            )
            result_chunks = result["chunks"]
            result_chapters = result.get("chapters", [])
            result_quotes = result.get("quotes", [])

        video_title = get_video_info(video_id).get('title', 'Untitled') if not is_url else info.title

        result_chapters = result.get("chapters", [])
        result_quotes = result.get("quotes", [])

        update_video_stage(video_id, "exported")

        md_path = export_markdown(
            video_id=video_id,
            video_title=video_title,
            transcript=transcript_data,
            chunk_summaries=result_chunks,
            final_summary=get_final_summary(video_id),
            chapters=result_chapters,
            quotes=result_quotes,
            note_style=note_style,
            output_dir=output_dir
        )
        json_path = export_json(
            video_id=video_id,
            video_title=video_title,
            video_url=video_path_or_url if is_url else None,
            video_author=info.author if is_url else None,
            duration=info.duration if is_url else duration,
            transcript=transcript_data,
            chunk_summaries=result_chunks,
            final_summary=get_final_summary(video_id),
            chapters=result_chapters,
            quotes=result_quotes,
            note_style=note_style.value,
            output_dir=output_dir
        )
        srt_path = export_srt(transcript_data, output_dir / f"{video_title[:50]}.srt")

        update_video_outputs(video_id, md_path, json_path, srt_path)
        update_video_status(video_id, "completed", "exported")

        logger.info(f"Pipeline completed for video {video_id}")
        logger.info(f"Outputs: {md_path}, {json_path}, {srt_path}")

        console.print(f"[green]Done! Video ID: {video_id}[/green]")
        console.print(f"[green]Markdown: {md_path}[/green]")
        console.print(f"[green]JSON: {json_path}[/green]")
        console.print(f"[green]SRT: {srt_path}[/green]")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


def _do_transcribe(video_id, audio_path, use_mock_asr, model, device, language, model_dir, force, resume):
    has_existing = has_transcript_segments(video_id)

    if has_existing and resume and not force:
        console.print(f"[green]Using existing transcript for video {video_id} (use --force to re-transcribe)[/green]")
        logger.info(f"Using existing transcript for video {video_id}")
        return get_transcript(video_id)

    if has_existing and force:
        console.print(f"[yellow]Clearing existing transcript for video {video_id}[/yellow]")
        logger.info(f"Clearing existing transcript for video {video_id} (--force)")
        clear_transcript_segments(video_id)

    console.print("[cyan]Transcribing audio...[/cyan]")
    logger.info(f"Transcribing audio for video {video_id}")
    update_video_stage(video_id, "audio_extracted")

    try:
        engine = FasterWhisperEngine(
            model_name=model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            use_mock=use_mock_asr,
            model_dir=model_dir,
            language=language
        )
        segments = engine.transcribe(audio_path, language=language)
        transcript_data = [
            {"start": s.start, "end": s.end, "text": s.text, "source": "mock" if use_mock_asr else "asr"}
            for s in segments
        ]
        save_transcript(video_id, transcript_data)
        update_video_stage(video_id, "transcribed")
        logger.info(f"Transcription complete for video {video_id}: {len(segments)} segments")
        return transcript_data

    except FasterWhisperError as e:
        console.print(f"[red]ASR error: {e}[/red]")
        update_video_stage(video_id, "transcribed", str(e))
        update_video_status(video_id, "failed")
        raise typer.Exit(code=1)


@app.command()
def summarize_local(
    video_path: str = typer.Argument(..., help="本地视频文件路径"),
    llm_provider: str = typer.Option("mock", "--llm-provider", help="LLM provider: mock, openai, ollama"),
    asr_provider: str = typer.Option("faster-whisper", "--asr-provider", help="ASR provider: faster-whisper, mock"),
    chunk_min: int = typer.Option(3, "--chunk-min", help="Minimum chunk duration in minutes"),
    chunk_max: int = typer.Option(5, "--chunk-max", help="Maximum chunk duration in minutes"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    model: str = typer.Option("base", "--model", help="Whisper model: tiny, base, small, medium, large"),
    device: str = typer.Option("cpu", "--device", help="Device: cpu, cuda, auto"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language: zh, en, auto"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Local model cache directory"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Keep extracted audio file"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from existing checkpoint (default: true)"),
    force: bool = typer.Option(False, "--force", help="Force re-transcribe even if transcript exists"),
    note_style: NoteStyle = typer.Option(NoteStyle.DETAILED, "--note-style",
                                        help="笔记模板: brief, detailed, study, meeting, tutorial"),
):
    _run_summarize(
        video_path_or_url=video_path,
        is_url=False,
        llm_provider=llm_provider,
        asr_provider=asr_provider,
        chunk_min=chunk_min,
        chunk_max=chunk_max,
        output=output,
        model=model,
        device=device,
        language=language,
        model_dir=model_dir,
        keep_audio=keep_audio,
        force=force,
        resume=resume,
        note_style=note_style
    )


@app.command()
def summarize_url(
    url: str = typer.Argument(..., help="B站视频链接或BV号"),
    llm_provider: str = typer.Option("mock", "--llm-provider", help="LLM provider: mock, openai, ollama"),
    asr_provider: str = typer.Option("faster-whisper", "--asr-provider", help="ASR provider: faster-whisper, mock"),
    chunk_min: int = typer.Option(3, "--chunk-min", help="Minimum chunk duration in minutes"),
    chunk_max: int = typer.Option(5, "--chunk-max", help="Maximum chunk duration in minutes"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    model: str = typer.Option("base", "--model", help="Whisper model: tiny, base, small, medium, large"),
    device: str = typer.Option("cpu", "--device", help="Device: cpu, cuda, auto"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language: zh, en, auto"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Local model cache directory"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Keep downloaded audio file"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from existing checkpoint (default: true)"),
    force: bool = typer.Option(False, "--force", help="Force re-transcribe even if transcript exists"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="Cookie文件路径 (Netscape格式)"),
    cookies_from_browser: Optional[str] = typer.Option(None, "--cookies-from-browser", 
                                                      help="从浏览器导入Cookie: chrome/firefox/edge/brave/safari"),
    proxy: Optional[str] = typer.Option(None, "--proxy", help="代理服务器地址"),
    user_agent: Optional[str] = typer.Option(None, "--user-agent", help="自定义User-Agent"),
    download_subtitle_only: bool = typer.Option(False, "--download-subtitle-only", help="只下载字幕，不进行ASR"),
    download_audio_only: bool = typer.Option(False, "--download-audio-only", help="只下载音频，不进行字幕提取"),
    note_style: NoteStyle = typer.Option(NoteStyle.DETAILED, "--note-style", 
                                        help="笔记模板: brief, detailed, study, meeting, tutorial"),
):
    _run_summarize(
        video_path_or_url=url,
        is_url=True,
        llm_provider=llm_provider,
        asr_provider=asr_provider,
        chunk_min=chunk_min,
        chunk_max=chunk_max,
        output=output,
        model=model,
        device=device,
        language=language,
        model_dir=model_dir,
        keep_audio=keep_audio,
        force=force,
        resume=resume,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        proxy=proxy,
        user_agent=user_agent,
        download_subtitle_only=download_subtitle_only,
        download_audio_only=download_audio_only,
        note_style=note_style
    )


@app.command()
def transcribe(
    video_path: str = typer.Argument(..., help="本地视频文件路径"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: srt, json, txt"),
    asr_provider: str = typer.Option("faster-whisper", "--asr-provider", help="ASR provider: faster-whisper, mock"),
    model: str = typer.Option("base", "--model", help="Whisper model: tiny, base, small, medium, large"),
    device: str = typer.Option("cpu", "--device", help="Device: cpu, cuda, auto"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language: zh, en, auto"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Local model cache directory"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Keep extracted audio file"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from existing checkpoint (default: true)"),
    force: bool = typer.Option(False, "--force", "-F", help="Force re-transcribe"),
):
    video_path = Path(video_path)
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(code=1)

    use_mock_asr = asr_provider.lower() == "mock"
    check_dependencies(require_asr=not use_mock_asr)

    try:
        console.print(f"[cyan]Transcribing: {video_path.name}[/cyan]")
        if use_mock_asr:
            console.print(f"[yellow]Using Mock ASR (--asr-provider mock)[/yellow]")

        video_id = create_video_record(
            source_type="local",
            source_path=str(video_path),
            title=video_path.stem
        )

        transcript_data = _do_transcribe(
            video_id, str(video_path), use_mock_asr, model, device, language, model_dir, force, resume
        )

        if output is None:
            settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output = str(settings.OUTPUT_DIR / f"{video_path.stem}_transcript.{format}")

        if format == "srt":
            output_path = export_srt(transcript_data, output)
        elif format == "json":
            output_path = export_json(
                video_id=video_id,
                video_title=video_path.stem,
                video_url=None,
                video_author=None,
                duration=None,
                transcript=transcript_data,
                chunk_summaries=[],
                final_summary=None,
                output_dir=Path(output).parent,
                output_filename=Path(output).name
            )
        else:
            output_path = output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for seg in transcript_data:
                    f.write(f"{seg['start']:.2f} - {seg['end']:.2f}: {seg['text']}\n\n")

        update_video_status(video_id, "transcribed")
        console.print(f"[green]Done! Output saved to: {output_path}[/green]")
        console.print(f"[green]Video ID: {video_id}[/green]")

        console.print(f"[dim]Transcript preview (first 3 segments):[/dim]")
        for seg in transcript_data[:3]:
            console.print(f"[dim]  {seg['start']:.2f} - {seg['end']:.2f}: {seg['text'][:50]}...[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(code=1)


@app.command()
def export_cmd(
    video_id: int = typer.Argument(..., help="视频ID"),
    format: str = typer.Option("markdown", "--format", "-f", help="Export format: markdown, srt, json"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file/directory path"),
):
    info = get_video_info(video_id)
    if not info:
        console.print(f"[red]Error: Video not found: {video_id}[/red]")
        raise typer.Exit(code=1)

    transcript = get_transcript(video_id)
    chunk_summaries = get_summary_chunks(video_id)
    final_summary = get_final_summary(video_id)

    if not transcript:
        console.print(f"[yellow]Warning: No transcript found for video {video_id}[/yellow]")

    try:
        if format == "markdown":
            output_path = export_markdown(
                video_id=video_id,
                video_title=info["title"] or "Untitled",
                transcript=transcript,
                chunk_summaries=chunk_summaries,
                final_summary=final_summary,
                output_dir=Path(output) if output else None
            )
        elif format == "srt":
            if not transcript:
                console.print("[red]Error: No transcript to export[/red]")
                raise typer.Exit(code=1)
            output_path = export_srt(transcript, output or str(settings.OUTPUT_DIR / f"video_{video_id}.srt"))
        elif format == "json":
            output_path = export_json(
                video_id=video_id,
                video_title=info["title"] or "Untitled",
                video_url=info["url"],
                video_author=info["author"],
                duration=info["duration"],
                transcript=transcript,
                chunk_summaries=chunk_summaries,
                final_summary=final_summary,
                output_dir=Path(output).parent if output else None,
                output_filename=Path(output).name if output else None
            )
        else:
            console.print(f"[red]Error: Unknown format: {format}[/red]")
            raise typer.Exit(code=1)

        console.print(f"[green]Exported to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    video_id: int = typer.Argument(..., help="视频ID"),
    format: str = typer.Option("markdown", "--format", "-f", help="输出格式: markdown, json"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径"),
):
    from .summarizer.pipeline import get_transcript, get_summary_chunks, get_final_summary, get_chapters, get_quotes, get_terms
    from .evaluator.evaluate import evaluate_video
    
    info = get_video_info(video_id)
    if not info:
        console.print(f"[red]Error: Video not found: {video_id}[/red]")
        raise typer.Exit(code=1)
    
    transcript = get_transcript(video_id)
    if not transcript:
        console.print(f"[red]Error: No transcript found for video {video_id}[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[cyan]Evaluating video {video_id}...[/cyan]")
    
    chunks = get_summary_chunks(video_id)
    final_summary = get_final_summary(video_id)
    quotes = get_quotes(video_id)
    terms = get_terms(video_id)
    chapters = get_chapters(video_id)
    
    output_content, result = evaluate_video(
        video_id=video_id,
        transcript=transcript,
        chunks=chunks,
        final_summary=final_summary,
        quotes=quotes,
        terms=terms,
        chapters=chapters,
        format=format
    )
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        console.print(f"[green]Evaluation report saved to: {output_path}[/green]")
    else:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        eval_dir = settings.OUTPUT_DIR / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        if format == "json":
            output_path = eval_dir / f"{video_id}_evaluation.json"
        else:
            output_path = eval_dir / f"{video_id}_evaluation.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        console.print(f"[green]Evaluation report saved to: {output_path}[/green]")
    
    console.print()
    console.print(Panel(f"[bold cyan]评估结果[/bold cyan]", expand=False))
    console.print()
    console.print(f"[bold]总分:[/bold] {result.overall_score} / 100")
    
    score_color = "green" if result.overall_score >= 80 else "yellow" if result.overall_score >= 60 else "red"
    console.print(f"[bold]等级:[/bold] [{score_color}]{_get_score_emoji(result.overall_score)} ({result.overall_score}分)[/{score_color}]")
    console.print()
    
    if result.warnings:
        console.print(f"[yellow]发现 {len(result.warnings)} 个问题[/yellow]")
        for warning in result.warnings[:3]:
            console.print(f"  [yellow]• {warning[:80]}[/yellow]")
        if len(result.warnings) > 3:
            console.print(f"  [dim]... 和其他 {len(result.warnings) - 3} 个问题[/dim]")
    
    if result.issues:
        console.print(f"[red]发现 {len(result.issues)} 个严重问题[/red]")
        for issue in result.issues[:3]:
            console.print(f"  [red]• {issue[:80]}[/red]")
        if len(result.issues) > 3:
            console.print(f"  [dim]... 和其他 {len(result.issues) - 3} 个问题[/dim]")


def _get_score_emoji(score: float) -> str:
    if score >= 90:
        return "🌟 优秀"
    elif score >= 80:
        return "✅ 良好"
    elif score >= 70:
        return "👍 一般"
    elif score >= 60:
        return "⚠️ 及格"
    else:
        return "❌ 需改进"


@app.command("list")
def list_videos(
    status_filter: Optional[str] = typer.Option(None, "--status", help="Filter by status: pending, processing, transcribed, completed, failed"),
):
    videos = get_all_videos()

    if status_filter:
        videos = [v for v in videos if v.get('status') == status_filter]

    if not videos:
        console.print("[yellow]No videos found.[/yellow]")
        return

    table = Table(title="Videos")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Source", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Stage", style="dim")
    table.add_column("Created", style="dim")

    for v in videos:
        status_style = {
            'completed': 'green',
            'failed': 'red',
            'processing': 'cyan',
            'transcribed': 'blue',
            'pending': 'yellow'
        }.get(v.get('status', ''), 'yellow')

        table.add_row(
            str(v["id"]),
            (v["title"] or "Untitled")[:30],
            v["source_type"],
            f"[{status_style}]{v['status']}[/{status_style}]",
            v.get('current_stage', '')[:15],
            v["created_at"][:10] if v.get("created_at") else ""
        )

    console.print(table)


def create_video_record(
    source_type: str,
    source_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    duration: Optional[float] = None
) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (source_type, source_path, url, title, author, duration, status, current_stage)
            VALUES (?, ?, ?, ?, ?, ?, 'processing', 'created')
        """, (source_type, source_path, url, title, author, duration))
        return cursor.lastrowid


def update_video_duration(video_id: int, duration: float):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET duration = ? WHERE id = ?", (duration, video_id))
        conn.commit()


if __name__ == "__main__":
    app()
