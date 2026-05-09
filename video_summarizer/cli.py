import sys
from pathlib import Path
from typing import Optional
import tempfile

import typer
from rich.console import Console
from rich.table import Table

from .config import settings
from .db import get_db, init_db
from .models import Video
from .media.ffmpeg import extract_audio, get_video_duration, check_ffmpeg_installed, FFmpegError
from .media.downloader import download_audio, download_subtitles, get_video_info, check_ytdlp_installed, DownloaderError
from .asr.faster_whisper_engine import FasterWhisperEngine, FasterWhisperError
from .asr.subtitle_parser import parse_srt
from .summarizer.pipeline import summarize_video_pipeline, get_transcript, get_summary_chunks, get_final_summary
from .exporters.markdown import export_markdown
from .exporters.srt import export_srt
from .exporters.json_exporter import export_json

app = typer.Typer(
    name="video-summarizer",
    help="B站/本地视频总结器 - 自动提取音频/字幕，转成文字并生成摘要"
)
console = Console()


def check_dependencies():
    errors = []
    if not check_ffmpeg_installed():
        errors.append("FFmpeg is not installed. Please install FFmpeg first.")
    if not check_ytdlp_installed():
        errors.append("yt-dlp is not installed. Run: pip install yt-dlp")
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
    init_db()


@app.command()
def summarize_local(
    video_path: str = typer.Argument(..., help="本地视频文件路径"),
    llm_provider: str = typer.Option("mock", "--llm-provider", help="LLM provider: mock, openai, ollama"),
    chunk_min: int = typer.Option(3, "--chunk-min", help="Minimum chunk duration in minutes"),
    chunk_max: int = typer.Option(5, "--chunk-max", help="Maximum chunk duration in minutes"),
):
    video_path = Path(video_path)
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(code=1)

    check_dependencies()

    try:
        console.print(f"[cyan]Processing local video: {video_path.name}[/cyan]")

        video_id = create_video_record(
            source_type="local",
            source_path=str(video_path),
            title=video_path.stem
        )

        duration = get_video_duration(str(video_path))
        update_video_duration(video_id, duration)

        console.print("[cyan]Extracting audio...[/cyan]")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            extract_audio(str(video_path), audio_path)
        except FFmpegError as e:
            console.print(f"[red]FFmpeg error: {e}[/red]")
            update_video_status(video_id, "failed")
            raise typer.Exit(code=1)

        console.print("[cyan]Transcribing audio...[/cyan]")
        engine = FasterWhisperEngine()
        segments = engine.transcribe(audio_path)

        transcript_data = [
            {"start": s.start, "end": s.end, "text": s.text, "source": "asr"}
            for s in segments
        ]
        from .summarizer.pipeline import save_transcript
        save_transcript(video_id, transcript_data)

        console.print("[cyan]Summarizing video...[/cyan]")
        result = summarize_video_pipeline(
            video_id,
            llm_provider=llm_provider,
            chunk_min=chunk_min,
            chunk_max=chunk_max
        )

        output_path = export_markdown(
            video_id=video_id,
            video_title=video_path.stem,
            transcript=transcript_data,
            chunk_summaries=result["chunks"],
            final_summary=result["final_summary"]
        )

        console.print(f"[green]Done! Output saved to: {output_path}[/green]")
        console.print(f"[green]Video ID: {video_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def summarize_url(
    url: str = typer.Argument(..., help="B站视频链接"),
    llm_provider: str = typer.Option("mock", "--llm-provider", help="LLM provider: mock, openai, ollama"),
    chunk_min: int = typer.Option(3, "--chunk-min", help="Minimum chunk duration in minutes"),
    chunk_max: int = typer.Option(5, "--chunk-max", help="Maximum chunk duration in minutes"),
):
    check_dependencies()

    try:
        console.print(f"[cyan]Processing URL: {url}[/cyan]")

        console.print("[cyan]Fetching video info...[/cyan]")
        try:
            info = get_video_info(url)
        except DownloaderError as e:
            console.print(f"[red]yt-dlp error: {e}[/red]")
            raise typer.Exit(code=1)

        video_id = create_video_record(
            source_type="url",
            url=url,
            title=info.title,
            author=info.author,
            duration=info.duration
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            srt_path = download_subtitles(url, temp_path)
            if srt_path and srt_path.exists():
                console.print(f"[green]Found subtitles: {srt_path}[/green]")
                console.print("[cyan]Parsing subtitles...[/cyan]")
                subtitle_segments = parse_srt(str(srt_path))
                transcript_data = [
                    {"start": s.start, "end": s.end, "text": s.text, "source": "subtitle"}
                    for s in subtitle_segments
                ]
                from .summarizer.pipeline import save_transcript
                save_transcript(video_id, transcript_data)
            else:
                console.print("[yellow]No subtitles found, downloading audio for ASR...[/yellow]")

                audio_path = temp_path / "audio.wav"
                try:
                    download_audio(url, audio_path)
                except DownloaderError as e:
                    console.print(f"[red]yt-dlp error: {e}[/red]")
                    update_video_status(video_id, "failed")
                    raise typer.Exit(code=1)

                console.print("[cyan]Transcribing audio...[/cyan]")
                engine = FasterWhisperEngine()
                segments = engine.transcribe(str(audio_path))
                transcript_data = [
                    {"start": s.start, "end": s.end, "text": s.text, "source": "asr"}
                    for s in segments
                ]
                from .summarizer.pipeline import save_transcript
                save_transcript(video_id, transcript_data)

        console.print("[cyan]Summarizing video...[/cyan]")
        result = summarize_video_pipeline(
            video_id,
            llm_provider=llm_provider,
            chunk_min=chunk_min,
            chunk_max=chunk_max
        )

        output_path = export_markdown(
            video_id=video_id,
            video_title=info.title,
            transcript=transcript_data,
            chunk_summaries=result["chunks"],
            final_summary=result["final_summary"]
        )

        console.print(f"[green]Done! Output saved to: {output_path}[/green]")
        console.print(f"[green]Video ID: {video_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    video_path: str = typer.Argument(..., help="本地视频文件路径"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("srt", "--format", "-f", help="Output format: srt, json, txt"),
):
    video_path = Path(video_path)
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        raise typer.Exit(code=1)

    check_dependencies()

    try:
        console.print(f"[cyan]Transcribing: {video_path.name}[/cyan]")

        video_id = create_video_record(
            source_type="local",
            source_path=str(video_path),
            title=video_path.stem
        )

        console.print("[cyan]Extracting audio...[/cyan]")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            extract_audio(str(video_path), audio_path)
        except FFmpegError as e:
            console.print(f"[red]FFmpeg error: {e}[/red]")
            update_video_status(video_id, "failed")
            raise typer.Exit(code=1)

        console.print("[cyan]Transcribing audio...[/cyan]")
        engine = FasterWhisperEngine()
        segments = engine.transcribe(audio_path)

        transcript_data = [
            {"start": s.start, "end": s.end, "text": s.text, "source": "asr"}
            for s in segments
        ]

        from .summarizer.pipeline import save_transcript
        save_transcript(video_id, transcript_data)

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
                output_path=output
            )
        else:
            output_path = output
            with open(output_path, "w", encoding="utf-8") as f:
                for seg in transcript_data:
                    f.write(f"{seg['start']:.2f} - {seg['end']:.2f}: {seg['text']}\n\n")

        update_video_status(video_id, "transcribed")
        console.print(f"[green]Done! Output saved to: {output_path}[/green]")
        console.print(f"[green]Video ID: {video_id}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def export(
    video_id: int = typer.Argument(..., help="视频ID"),
    format: str = typer.Option("markdown", "--format", "-f", help="Export format: markdown, srt, json"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = cursor.fetchone()

        if not video:
            console.print(f"[red]Error: Video not found: {video_id}[/red]")
            raise typer.Exit(code=1)

        video = dict(video)

    transcript = get_transcript(video_id)
    chunk_summaries = get_summary_chunks(video_id)
    final_summary = get_final_summary(video_id)

    if not transcript:
        console.print(f"[yellow]Warning: No transcript found for video {video_id}[/yellow]")

    try:
        if format == "markdown":
            output_path = export_markdown(
                video_id=video_id,
                video_title=video["title"] or "Untitled",
                transcript=transcript,
                chunk_summaries=chunk_summaries,
                final_summary=final_summary,
                output_path=output
            )
        elif format == "srt":
            if not transcript:
                console.print("[red]Error: No transcript to export[/red]")
                raise typer.Exit(code=1)
            output_path = export_srt(transcript, output or str(settings.OUTPUT_DIR / f"video_{video_id}.srt"))
        elif format == "json":
            output_path = export_json(
                video_id=video_id,
                video_title=video["title"] or "Untitled",
                video_url=video["url"],
                video_author=video["author"],
                duration=video["duration"],
                transcript=transcript,
                chunk_summaries=chunk_summaries,
                final_summary=final_summary,
                output_path=output
            )
        else:
            console.print(f"[red]Error: Unknown format: {format}[/red]")
            raise typer.Exit(code=1)

        console.print(f"[green]Exported to: {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("list")
def list_videos():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
        videos = cursor.fetchall()

        if not videos:
            console.print("[yellow]No videos found.[/yellow]")
            return

        table = Table(title="Videos")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Source", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="dim")

        for v in videos:
            table.add_row(
                str(v["id"]),
                v["title"][:40] if v["title"] else "Untitled",
                v["source_type"],
                v["status"],
                v["created_at"][:19] if v["created_at"] else ""
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
            INSERT INTO videos (source_type, source_path, url, title, author, duration, status)
            VALUES (?, ?, ?, ?, ?, ?, 'processing')
        """, (source_type, source_path, url, title, author, duration))
        return cursor.lastrowid


def update_video_duration(video_id: int, duration: float):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET duration = ? WHERE id = ?", (duration, video_id))
        conn.commit()


def update_video_status(video_id: int, status: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET status = ? WHERE id = ?", (status, video_id))
        conn.commit()


if __name__ == "__main__":
    app()
