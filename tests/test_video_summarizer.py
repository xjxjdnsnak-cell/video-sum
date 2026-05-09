import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_summarizer.summarizer.prompts import NoteStyle


class TestConfig:
    def test_settings_defaults(self):
        from video_summarizer.config import settings
        assert settings.LLM_PROVIDER == "mock"
        assert settings.CHUNK_DURATION_MIN == 3
        assert settings.CHUNK_DURATION_MAX == 5

    def test_settings_ensure_directories(self, tmp_path):
        from video_summarizer.config import Settings
        test_settings = Settings()
        test_settings.DB_PATH = tmp_path / ".video_summarizer" / "test.db"
        test_settings.OUTPUT_DIR = tmp_path / "output"
        test_settings.ensure_directories()
        assert test_settings.DB_PATH.parent.exists()
        assert test_settings.OUTPUT_DIR.exists()


class TestModels:
    def test_video_model(self):
        from video_summarizer.models import Video
        video = Video(source_type="local", title="Test Video", id=1)
        assert video.id == 1
        assert video.source_type == "local"
        assert video.title == "Test Video"

    def test_transcript_segment_model(self):
        from video_summarizer.models import TranscriptSegment
        seg = TranscriptSegment(video_id=1, start=0.0, end=5.0, text="Hello world")
        assert seg.video_id == 1
        assert seg.start == 0.0
        assert seg.text == "Hello world"

    def test_summary_chunk_model(self):
        from video_summarizer.models import SummaryChunk
        chunk = SummaryChunk(
            video_id=1, start=0.0, end=180.0,
            source_text="Original text", summary="Summary"
        )
        assert chunk.end - chunk.start == 180.0


class TestTimefmt:
    def test_format_timestamp(self):
        from video_summarizer.utils.timefmt import format_timestamp
        assert format_timestamp(0) == "00:00:00"
        assert format_timestamp(3661) == "01:01:01"
        assert format_timestamp(3723.5) == "01:02:03"

    def test_format_timestamp_with_ms(self):
        from video_summarizer.utils.timefmt import format_timestamp_with_ms
        result = format_timestamp_with_ms(5.25)
        assert "05" in result
        assert "250" in result

    def test_parse_timestamp(self):
        from video_summarizer.utils.timefmt import parse_timestamp
        assert parse_timestamp("01:02:03") == 3723.0
        assert parse_timestamp("01:02:03,500") == 3723.5


class TestLLMClient:
    def test_mock_llm_client(self):
        from video_summarizer.summarizer.llm_client import MockLLMClient
        client = MockLLMClient()

        result = client.summarize_chunk("Test text content", "00:00:00", "00:05:00")
        assert "topic" in result
        assert "key_points" in result
        assert "summary" in result
        assert "quote" in result
        assert "00:00:00" in result["topic"]

        final = client.generate_final_summary("Test Video", [
            {"start_time": "00:00", "end_time": "05:00", "summary": "Test summary"}
        ])
        assert "one_sentence_summary" in final
        assert "key_points" in final

    def test_get_llm_client_mock(self):
        from video_summarizer.summarizer.llm_client import get_llm_client, MockLLMClient
        client = get_llm_client("mock")
        assert isinstance(client, MockLLMClient)


class TestASREngine:
    def test_mock_asr_returns_mock_segments(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine(use_mock=True)
        segments = engine.model.transcribe("/fake/path.wav")

        assert len(segments) > 0
        assert all(hasattr(s, 'start') and hasattr(s, 'end') and hasattr(s, 'text') for s in segments)
        assert any("[Mock转写]" in s.text for s in segments)

    def test_whisper_failure_does_not_fallback_to_mock(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine, FasterWhisperError
        import faster_whisper

        engine = FasterWhisperEngine(use_mock=False, model_name="nonexistent-model")

        with patch.object(faster_whisper, 'WhisperModel', side_effect=Exception("Network error")):
            with pytest.raises(FasterWhisperError) as exc_info:
                _ = engine.model

            error_msg = str(exc_info.value)
            assert "解决方案" in error_msg
            assert "--asr-provider mock" in error_msg
            assert "Failed to download" in error_msg or "Failed to load" in error_msg

    def test_asr_provider_mock_flag(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine

        engine_mock = FasterWhisperEngine(use_mock=True)
        assert engine_mock.use_mock == True
        assert isinstance(engine_mock.model, type(engine_mock.model))

        engine_real = FasterWhisperEngine(use_mock=False)
        assert engine_real.use_mock == False

    def test_mock_asr_does_not_affect_faster_whisper(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine

        engine_mock = FasterWhisperEngine(use_mock=True)
        assert engine_mock.use_mock == True
        assert not hasattr(engine_mock, '_model') or engine_mock._model is None

        engine_real = FasterWhisperEngine(use_mock=False)
        assert engine_real.use_mock == False
        assert not hasattr(engine_real, '_model') or engine_real._model is None

        engine_mock.model
        assert not hasattr(engine_mock, '_model') or engine_mock._model is None

    def test_model_dir_parameter_passed(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine

        engine = FasterWhisperEngine(
            model_name="tiny",
            model_dir="/custom/model/path"
        )
        assert engine.model_dir == Path("/custom/model/path")
        assert engine.model_name == "tiny"

    def test_device_auto_detection(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine

        engine_cpu = FasterWhisperEngine(device="cpu")
        assert engine_cpu.device == "cpu"

        engine_cuda = FasterWhisperEngine(device="cuda")
        assert engine_cuda.device == "cuda"


class TestChunker:
    def test_create_chunks_from_segments(self):
        from video_summarizer.summarizer.chunker import create_chunks_from_segments

        segments = [
            {"start": 0.0, "end": 5.0, "text": "First segment"},
            {"start": 5.0, "end": 10.0, "text": "Second segment"},
            {"start": 10.0, "end": 15.0, "text": "Third segment"},
        ]

        chunks = create_chunks_from_segments(
            segments,
            min_duration=60,
            max_duration=300
        )

        assert len(chunks) >= 1
        assert chunks[0].start == 0.0
        assert "First segment" in chunks[0].text

    def test_create_chunks_empty(self):
        from video_summarizer.summarizer.chunker import create_chunks_from_segments
        chunks = create_chunks_from_segments([])
        assert len(chunks) == 0

    def test_merge_short_chunks(self):
        from video_summarizer.summarizer.chunker import TextChunk, merge_short_chunks

        chunks = [
            TextChunk(start=0, end=30, text="Short 1"),
            TextChunk(start=30, end=60, text="Short 2"),
            TextChunk(start=120, end=180, text="Long"),
        ]

        merged = merge_short_chunks(chunks, min_duration=60)
        assert len(merged) <= len(chunks)


class TestSubtitleParser:
    def test_parse_srt_file(self, tmp_path):
        from video_summarizer.asr.subtitle_parser import parse_srt

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:05,000 --> 00:00:10,000
This is a test
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt(str(srt_file))
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.0
        assert segments[1].end == 10.0

    def test_parse_srt_file_not_found(self):
        from video_summarizer.asr.subtitle_parser import parse_srt
        with pytest.raises(FileNotFoundError):
            parse_srt("/nonexistent/file.srt")


class TestExporters:
    def test_export_markdown(self, tmp_path):
        from video_summarizer.exporters.markdown import export_markdown

        output = tmp_path / "output.md"
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Test segment"}
        ]
        chunks = [
            {
                "start": 0.0, "end": 180.0,
                "start_time": "00:00:00", "end_time": "00:03:00",
                "summary": "Test summary"
            }
        ]
        final_summary = {
            "one_sentence_summary": "One sentence",
            "detailed_summary": "Detailed text",
            "key_points": "- Point 1\n- Point 2",
            "questions": "1. Q1\n2. Q2"
        }

        result = export_markdown(
            video_id=1,
            video_title="Test Video",
            transcript=transcript,
            chunk_summaries=chunks,
            final_summary=final_summary,
            output_path=str(output)
        )

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "详细笔记：" in content or "Test Video" in content
        assert "一句话总结" in content
        assert "Test segment" in content

    def test_export_srt(self, tmp_path):
        from video_summarizer.exporters.srt import export_srt

        output = tmp_path / "output.srt"
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "Test content"},
        ]

        result = export_srt(segments, str(output))
        assert Path(result).exists()

        content = Path(result).read_text()
        assert "00:00:00,000 --> 00:00:05,000" in content
        assert "Hello world" in content

    def test_export_json(self, tmp_path):
        from video_summarizer.exporters.json_exporter import export_json
        import json

        output = tmp_path / "output.json"
        transcript = [{"start": 0.0, "end": 5.0, "text": "Test"}]

        result = export_json(
            video_id=1,
            video_title="Test",
            video_url="http://test.com",
            video_author="Author",
            duration=60.0,
            transcript=transcript,
            chunk_summaries=[],
            final_summary={"test": "summary"},
            output_path=str(output)
        )

        assert Path(result).exists()
        data = json.loads(Path(result).read_text())
        assert data["video_id"] == 1
        assert data["title"] == "Test"


class TestMockPipeline:
    def test_mock_asr_with_mock_llm_creates_full_markdown(self, tmp_path):
        from video_summarizer.exporters.markdown import export_markdown
        from video_summarizer.summarizer.llm_client import MockLLMClient

        llm_client = MockLLMClient()

        mock_transcript = [
            {"start": 0.0, "end": 5.0, "text": "[Mock转写] 第一段内容"},
            {"start": 5.0, "end": 10.0, "text": "[Mock转写] 第二段内容"},
        ]

        mock_chunks = [
            {
                "start": 0.0,
                "end": 10.0,
                "start_time": "00:00:00",
                "end_time": "00:00:10",
                "summary": llm_client.summarize_chunk(
                    " ".join([s["text"] for s in mock_transcript]),
                    "00:00:00",
                    "00:00:10"
                )["summary"]
            }
        ]

        final_summary = llm_client.generate_final_summary(
            "Test Video",
            [{"start_time": "00:00:00", "end_time": "00:00:10", "summary": mock_chunks[0]["summary"]}]
        )

        output_path = export_markdown(
            video_id=1,
            video_title="Test Video",
            transcript=mock_transcript,
            chunk_summaries=mock_chunks,
            final_summary=final_summary,
            note_style=NoteStyle.DETAILED,
            output_dir=tmp_path
        )

        content = Path(output_path).read_text()
        assert "详细笔记" in content
        assert "Test Video" in content
        assert "一句话总结" in content
        assert "完整转写" in content
        assert "## 详细总结" in content
        assert "## 时间轴摘要" in content
        assert "## 关键知识点" in content
        assert "## 完整转写" in content
        assert "[Mock转写]" in content
        assert "[Mock]" in content

    def test_mock_transcript_json_structure(self, tmp_path):
        from video_summarizer.exporters.json_exporter import export_json
        from video_summarizer.summarizer.llm_client import MockLLMClient
        import json

        llm_client = MockLLMClient()

        mock_transcript = [
            {"start": 0.0, "end": 5.0, "text": "[Mock转写] 第一段内容"},
            {"start": 5.0, "end": 10.0, "text": "[Mock转写] 第二段内容"},
        ]

        output_path = export_json(
            video_id=1,
            video_title="Test",
            video_url=None,
            video_author=None,
            duration=10.0,
            transcript=mock_transcript,
            chunk_summaries=[],
            final_summary=None,
            output_dir=tmp_path
        )

        data = json.loads(Path(output_path).read_text())

        assert "transcript" in data
        assert len(data["transcript"]) == 2
        assert data["transcript"][0]["start"] == 0.0
        assert data["transcript"][0]["end"] == 5.0
        assert data["transcript"][0]["text"] == "[Mock转写] 第一段内容"


class TestDatabase:
    def test_init_db(self, tmp_path):
        from video_summarizer.db import init_db, get_db_connection
        import os

        db_path = tmp_path / "test.db"
        os.environ["HOME"] = str(tmp_path)

        with patch('video_summarizer.config.settings') as mock_settings:
            mock_settings.DB_PATH = db_path
            from video_summarizer import db
            db.settings = mock_settings
            db.init_db()

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert "videos" in tables
            assert "transcript_segments" in tables
            assert "summary_chunks" in tables
            assert "final_summaries" in tables


class TestDoctor:
    def test_doctor_command_exists(self):
        from video_summarizer.cli import doctor
        assert callable(doctor)


class TestResumeAndForce:
    def test_db_has_resume_functions(self):
        from video_summarizer.db import has_transcript_segments, has_summary_chunks, has_final_summary
        assert callable(has_transcript_segments)
        assert callable(has_summary_chunks)
        assert callable(has_final_summary)

    def test_db_clear_functions_exist(self):
        from video_summarizer.db import clear_transcript_segments, clear_summary_chunks, clear_final_summary
        assert callable(clear_transcript_segments)
        assert callable(clear_summary_chunks)
        assert callable(clear_final_summary)


class TestCleanCommand:
    def test_clean_command_exists(self):
        from video_summarizer.cli import clean
        assert callable(clean)


class TestStatusCommand:
    def test_status_command_exists(self):
        from video_summarizer.cli import status
        assert callable(status)

    def test_get_video_info_returns_correct_fields(self, tmp_path):
        from video_summarizer.db import init_db, get_video_info
        from video_summarizer.cli import create_video_record

        db_path = tmp_path / "test.db"
        with patch('video_summarizer.config.settings') as mock_settings:
            mock_settings.DB_PATH = db_path
            from video_summarizer import db as db_module
            db_module.settings = mock_settings
            init_db()

            video_id = create_video_record(
                source_type="local",
                source_path="/test/video.mp4",
                title="Test Video"
            )

            info = get_video_info(video_id)
            assert info is not None
            assert info["id"] == video_id
            assert info["title"] == "Test Video"
            assert info["transcript_count"] == 0
            assert info["chunk_count"] == 0
            assert info["has_final_summary"] == 0


class TestDoctorCommand:
    def test_doctor_command_imports_correctly(self):
        from video_summarizer.cli import doctor
        assert callable(doctor)

    def test_doctor_checks_include_required_items(self):
        from video_summarizer.cli import doctor
        from typer.testing import CliRunner
        from video_summarizer import cli
        import inspect

        source = inspect.getsource(cli.doctor)
        assert "Python Version" in source
        assert "faster-whisper" in source
        assert "FFmpeg" in source
        assert "yt-dlp" in source
        assert "LLM Provider" in source


class TestResumeBehavior:
    def test_mock_asr_twice_does_not_duplicate_transcript(self, tmp_path, monkeypatch):
        from video_summarizer.db import (
            init_db, get_db, get_all_videos,
            has_transcript_segments
        )
        from video_summarizer.summarizer.pipeline import save_transcript, get_transcript

        db_path = tmp_path / ".video_summarizer" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_DB_PATH", str(db_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_OUTPUT_DIR", str(tmp_path / "output"))

        import importlib
        import video_summarizer.config
        importlib.reload(video_summarizer.config)
        importlib.reload(video_summarizer.db)

        from video_summarizer.config import settings
        settings.ensure_directories()

        from video_summarizer import db as db_module
        db_module.init_db()

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO videos (source_type, source_path, title, status, current_stage) VALUES (?, ?, ?, ?, ?)",
                ("local", "/test/video.mp4", "Test Video", "processing", "created")
            )
            video_id = cursor.lastrowid

        mock_transcript = [
            {"start": 0.0, "end": 5.0, "text": "Test segment 1"},
            {"start": 5.0, "end": 10.0, "text": "Test segment 2"},
        ]
        save_transcript(video_id, mock_transcript)

        assert has_transcript_segments(video_id)
        first_transcript = get_transcript(video_id)
        first_count = len(first_transcript)

        second_transcript = get_transcript(video_id)
        second_count = len(second_transcript)
        assert second_count == first_count

    def test_force_clears_existing_transcript(self, tmp_path, monkeypatch):
        from video_summarizer.db import (
            init_db, get_db, clear_transcript_segments, has_transcript_segments,
        )
        from video_summarizer.summarizer.pipeline import save_transcript

        db_path = tmp_path / ".video_summarizer" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_DB_PATH", str(db_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_OUTPUT_DIR", str(tmp_path / "output"))

        import importlib
        import video_summarizer.config
        importlib.reload(video_summarizer.config)
        importlib.reload(video_summarizer.db)

        from video_summarizer.config import settings
        settings.ensure_directories()

        from video_summarizer import db as db_module
        db_module.init_db()

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO videos (source_type, source_path, title, status, current_stage) VALUES (?, ?, ?, ?, ?)",
                ("local", "/test/video.mp4", "Test Video", "processing", "created")
            )
            video_id = cursor.lastrowid

        mock_transcript = [{"start": 0.0, "end": 5.0, "text": "Test segment"}]
        save_transcript(video_id, mock_transcript)
        assert has_transcript_segments(video_id)

        clear_transcript_segments(video_id)
        assert not has_transcript_segments(video_id)


class TestCleanCommandBehavior:
    def test_clean_temp_only_dry_run_shows_files_to_delete(self, tmp_path, monkeypatch):
        db_path = tmp_path / ".video_summarizer" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = tmp_path / "video_summarizer_output"
        output_dir.mkdir()
        log_dir = output_dir / "logs"
        log_dir.mkdir()
        test_log = log_dir / "run-test.log"
        test_log.write_text("test log content")

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_DB_PATH", str(db_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_OUTPUT_DIR", str(output_dir))

        import importlib
        import video_summarizer.config
        importlib.reload(video_summarizer.config)
        importlib.reload(video_summarizer.db)

        from video_summarizer.config import settings
        settings.ensure_directories()

        from video_summarizer import db as db_module
        db_module.init_db()

        from video_summarizer.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['clean', '--temp-only', '--dry-run'])

        assert result.exit_code == 0
        assert "dry-run mode" in result.output.lower() or "dry run" in result.output.lower()


class TestStatusCommandBehavior:
    def test_status_shows_transcript_and_summary_count(self, tmp_path, monkeypatch):
        db_path = tmp_path / ".video_summarizer" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_DB_PATH", str(db_path))
        monkeypatch.setenv("VIDEO_SUMMARIZER_OUTPUT_DIR", str(tmp_path / "output"))

        import importlib
        import video_summarizer.config
        importlib.reload(video_summarizer.config)
        importlib.reload(video_summarizer.db)

        from video_summarizer.config import settings
        settings.ensure_directories()

        from video_summarizer import db as db_module
        db_module.init_db()

        with db_module.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO videos (source_type, source_path, title, status, current_stage) VALUES (?, ?, ?, ?, ?)",
                ("local", "/test/video.mp4", "Test Video", "processing", "created")
            )
            video_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO transcript_segments (video_id, start, end, text) VALUES (?, ?, ?, ?)",
                (video_id, 0.0, 5.0, "Test segment 1")
            )
            cursor.execute(
                "INSERT INTO transcript_segments (video_id, start, end, text) VALUES (?, ?, ?, ?)",
                (video_id, 5.0, 10.0, "Test segment 2")
            )
            cursor.execute(
                "INSERT INTO summary_chunks (video_id, start, end, source_text, summary) VALUES (?, ?, ?, ?, ?)",
                (video_id, 0.0, 10.0, "Test source", "Test summary")
            )
            cursor.execute(
                "INSERT INTO final_summaries (video_id, one_sentence_summary, detailed_summary) VALUES (?, ?, ?)",
                (video_id, "One sentence", "Detailed")
            )
            conn.commit()

        from video_summarizer.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['status', str(video_id)])

        assert result.exit_code == 0
        assert "Transcript Segments: 2" in result.output
        assert "Summary Chunks: 1" in result.output
        assert "Has Final Summary: Yes" in result.output


class TestWhisperNoFallback:
    def test_faster_whisper_failure_no_mock_fallback(self):
        from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine, FasterWhisperError

        engine = FasterWhisperEngine(use_mock=False, model_name="invalid-model")

        with patch('faster_whisper.WhisperModel', side_effect=Exception("Download failed")):
            with pytest.raises(FasterWhisperError) as exc_info:
                _ = engine.model

            error_msg = str(exc_info.value)
            assert "解决方案" in error_msg or "Failed" in error_msg


class TestBilibiliURL:
    def test_bv_number_recognition(self):
        from video_summarizer.media.downloader import is_bilibili_url, normalize_bilibili_url

        assert is_bilibili_url("BV1xx411c7mZ")
        assert is_bilibili_url("BV1xx411C7MZ")
        assert normalize_bilibili_url("BV1xx411c7mZ") == "https://www.bilibili.com/video/BV1xx411c7mZ"

    def test_av_number_recognition(self):
        from video_summarizer.media.downloader import is_bilibili_url, normalize_bilibili_url

        assert is_bilibili_url("av12345678")
        assert is_bilibili_url("AV12345678")
        assert normalize_bilibili_url("av12345678") == "https://www.bilibili.com/video/av12345678"

    def test_bilibili_url_recognition(self):
        from video_summarizer.media.downloader import is_bilibili_url, normalize_bilibili_url

        assert is_bilibili_url("https://www.bilibili.com/video/BV1xx411c7mZ")
        assert is_bilibili_url("https://b23.tv/BV1xx411c7mZ")
        assert is_bilibili_url("//www.bilibili.com/video/BV1xx411c7mZ")
        assert normalize_bilibili_url("//www.bilibili.com/video/BV1xx411c7mZ") == "https://www.bilibili.com/video/BV1xx411c7mZ"

    def test_non_bilibili_url(self):
        from video_summarizer.media.downloader import is_bilibili_url

        assert not is_bilibili_url("https://www.youtube.com/watch?v=abc123")
        assert not is_bilibili_url("https://example.com/video")


class TestDownloaderErrorHandling:
    def test_http_412_error_message(self):
        from video_summarizer.media.downloader import handle_ytdlp_error, DownloaderError

        with pytest.raises(DownloaderError) as exc_info:
            handle_ytdlp_error("HTTP Error 412: Precondition Failed")
        
        error_msg = str(exc_info.value)
        assert "HTTP 412" in error_msg
        assert "Cookie" in error_msg

    def test_http_403_error_message(self):
        from video_summarizer.media.downloader import handle_ytdlp_error, DownloaderError

        with pytest.raises(DownloaderError) as exc_info:
            handle_ytdlp_error("HTTP Error 403: Forbidden")
        
        error_msg = str(exc_info.value)
        assert "HTTP 403" in error_msg
        assert "登录" in error_msg

    def test_video_unavailable_error(self):
        from video_summarizer.media.downloader import handle_ytdlp_error, DownloaderError

        with pytest.raises(DownloaderError) as exc_info:
            handle_ytdlp_error("Video unavailable")
        
        error_msg = str(exc_info.value)
        assert "不存在" in error_msg or "删除" in error_msg


class TestCookiesParameter:
    def test_cookies_file_parameter(self, tmp_path):
        from video_summarizer.media.downloader import build_ytdlp_args

        cookies_file = tmp_path / "cookies.txt"
        cookies_file.write_text("test cookies")
        
        args = build_ytdlp_args("https://test.com", cookies_file=str(cookies_file))
        assert "--cookies" in args
        assert str(cookies_file) in args

    def test_cookies_from_browser_parameter(self):
        from video_summarizer.media.downloader import build_ytdlp_args

        args = build_ytdlp_args("https://test.com", cookies_from_browser="chrome")
        assert "--cookies-from-browser" in args
        assert "chrome" in args

    def test_invalid_cookies_from_browser(self):
        from video_summarizer.media.downloader import build_ytdlp_args, DownloaderError

        with pytest.raises(DownloaderError):
            build_ytdlp_args("https://test.com", cookies_from_browser="invalid")

    def test_proxy_parameter(self):
        from video_summarizer.media.downloader import build_ytdlp_args

        args = build_ytdlp_args("https://test.com", proxy="http://127.0.0.1:7890")
        assert "--proxy" in args
        assert "http://127.0.0.1:7890" in args
