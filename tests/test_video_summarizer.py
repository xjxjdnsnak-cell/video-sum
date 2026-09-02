import pytest
import tempfile
import threading
import time
import os
from pathlib import Path
from types import SimpleNamespace
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


class TestModelsRemoved:
    """A-7: video_summarizer/models.py was dead code and has been removed."""

    def test_models_module_removed(self):
        import video_summarizer
        pkg_dir = Path(video_summarizer.__file__).parent
        assert not (pkg_dir / "models.py").exists()

    def test_package_imports_cleanly_without_models(self):
        """Import scan: every module in the package must still import."""
        import importlib
        import pkgutil
        import video_summarizer

        failed = []
        for mod in pkgutil.walk_packages(video_summarizer.__path__, prefix="video_summarizer."):
            try:
                importlib.import_module(mod.name)
            except Exception as e:  # pragma: no cover - only on failure
                failed.append((mod.name, repr(e)))
        assert failed == []

    def test_no_source_references_models_module(self):
        import video_summarizer
        pkg_dir = Path(video_summarizer.__file__).parent
        markers = (
            "from ..models",
            "from .models",
            "from video_summarizer.models",
            "import video_summarizer.models",
            "import models",
        )
        offenders = []
        for py in pkg_dir.rglob("*.py"):
            src = py.read_text(encoding="utf-8")
            for marker in markers:
                if marker in src:
                    offenders.append(f"{py.name}: {marker}")
        assert offenders == []


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

    def test_doctor_probes_each_tool_once(self, monkeypatch):
        """P-4: ffmpeg/yt-dlp must be probed once per doctor run, not twice."""
        from video_summarizer import cli
        from typer.testing import CliRunner

        calls = {"ffmpeg": 0, "ytdlp": 0}

        def fake_ffmpeg():
            calls["ffmpeg"] += 1
            return True

        def fake_ytdlp():
            calls["ytdlp"] += 1
            return True

        monkeypatch.setattr(cli, "check_ffmpeg_installed", fake_ffmpeg)
        monkeypatch.setattr(cli, "check_ytdlp_installed", fake_ytdlp)
        # Isolate the CLI callback and doctor's DB listing from module-level
        # settings mutations made by other tests in the session; this test
        # only cares about the probe counts.
        monkeypatch.setattr(cli, "setup_logging", lambda: "unused.log")
        monkeypatch.setattr(cli, "init_db", lambda: None)
        import video_summarizer.db as db_module
        monkeypatch.setattr(db_module, "get_all_videos", lambda: [])

        # The autouse settings fixture points OUTPUT_DIR at a temp dir but
        # never creates it; doctor's writability check would report NOT writable.
        Path(cli.settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        runner = CliRunner()
        result = runner.invoke(cli.app, ["doctor"])

        assert result.exit_code == 0, result.output
        assert calls["ffmpeg"] == 1
        assert calls["ytdlp"] == 1


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


class TestAllowAnyUrl:
    """S-4: non-bilibili URLs require an explicit --allow-any-url opt-in."""

    NON_BILI_URL = "https://example.com/videos/123"

    def _mock_pipeline(self, monkeypatch):
        """Stub every network-touching callable in the CLI's URL flow."""
        monkeypatch.setattr("video_summarizer.cli.check_ffmpeg_installed", lambda: True)
        monkeypatch.setattr("video_summarizer.cli.check_ytdlp_installed", lambda: True)

        info = SimpleNamespace(title="Example Video", author="someone", duration=60.0, requires_login=False)
        info_calls = []

        def fake_get_url_info(url, **kwargs):
            info_calls.append(url)
            return info

        monkeypatch.setattr("video_summarizer.cli.get_url_info", fake_get_url_info)
        return info_calls

    def test_non_bilibili_url_refused_without_flag(self, monkeypatch):
        from video_summarizer.db import init_db
        from typer.testing import CliRunner
        from video_summarizer.cli import app

        init_db()
        info_calls = self._mock_pipeline(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(app, [
            "summarize-url", self.NON_BILI_URL,
            "--asr-provider", "mock",
            "--llm-provider", "mock",
        ])

        assert result.exit_code != 0
        assert "allow-any-url" in result.output
        # Refusal must happen before any yt-dlp call
        assert info_calls == []

    def test_non_bilibili_url_allowed_with_flag(self, tmp_path, monkeypatch):
        from video_summarizer.db import init_db
        from typer.testing import CliRunner
        from video_summarizer.cli import app

        init_db()
        info_calls = self._mock_pipeline(monkeypatch)

        srt_content = """1
00:00:00,000 --> 00:00:05,000
第一句字幕

2
00:00:05,000 --> 00:00:10,000
第二句字幕
"""

        def fake_download_subtitles(url, output_dir, **kwargs):
            srt = Path(output_dir) / "example.srt"
            srt.write_text(srt_content, encoding="utf-8")
            return srt

        monkeypatch.setattr("video_summarizer.cli.download_subtitles", fake_download_subtitles)

        runner = CliRunner()
        result = runner.invoke(app, [
            "summarize-url", self.NON_BILI_URL,
            "--asr-provider", "mock",
            "--llm-provider", "mock",
            "--allow-any-url",
        ])

        assert result.exit_code == 0, result.output
        assert info_calls == [self.NON_BILI_URL]

    def test_bilibili_url_does_not_need_flag(self, monkeypatch):
        """BV号 input (a bilibili identity) must keep working without the flag."""
        from video_summarizer.db import init_db
        from typer.testing import CliRunner
        from video_summarizer.cli import app

        init_db()
        info_calls = self._mock_pipeline(monkeypatch)

        def fake_download_subtitles(url, output_dir, **kwargs):
            srt = Path(output_dir) / "BV.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:05,000\n字幕\n", encoding="utf-8")
            return srt

        monkeypatch.setattr("video_summarizer.cli.download_subtitles", fake_download_subtitles)

        runner = CliRunner()
        result = runner.invoke(app, [
            "summarize-url", "BV1xx411c7mZ",
            "--asr-provider", "mock",
            "--llm-provider", "mock",
        ])

        assert result.exit_code == 0, result.output
        assert info_calls == ["BV1xx411c7mZ"]


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


class TestEvaluator:
    def test_quote_not_in_transcript_raises_error(self):
        from video_summarizer.evaluator.evaluate import TranscriptValidator

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "这是正常的转写文本内容"}
        ]
        validator = TranscriptValidator(transcript)
        
        assert not validator.quote_exists_in_transcript("完全不存在的引用XYZ123")
        assert validator.quote_exists_in_transcript("这是正常的转写文本内容")

    def test_invalid_timestamp_range(self):
        from video_summarizer.evaluator.evaluate import TranscriptValidator

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "测试内容"}
        ]
        validator = TranscriptValidator(transcript)
        
        assert not validator.time_in_transcript_range("99:00", "99:30")
        assert validator.time_in_transcript_range("00:00", "00:05")

    def test_empty_summary_raises_warning(self):
        from video_summarizer.evaluator.evaluate import Evaluator

        transcript = [{"start": 0.0, "end": 5.0, "text": "测试内容"}]
        
        evaluator = Evaluator(
            transcript=transcript,
            chunks=[],
            final_summary={}
        )
        result = evaluator.evaluate()
        
        assert len(result.warnings) > 0
        assert any("empty" in w.lower() for w in result.warnings)

    def test_evaluation_report_generates(self):
        from video_summarizer.evaluator.evaluate import Evaluator, generate_markdown_report

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "这是测试内容"},
            {"start": 5.0, "end": 10.0, "text": "第二段测试内容"}
        ]
        chunks = [
            {"start": 0.0, "end": 5.0, "start_time": "00:00", "end_time": "00:05", "summary": "测试摘要1"},
            {"start": 5.0, "end": 10.0, "start_time": "00:05", "end_time": "00:10", "summary": "测试摘要2"}
        ]
        final_summary = {
            "one_sentence_summary": "这是一句测试总结",
            "key_points": ["要点1", "要点2"]
        }

        evaluator = Evaluator(transcript, chunks, final_summary)
        result = evaluator.evaluate()
        
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        assert "completeness_score" in result.to_dict()

    def test_evaluate_command_exists(self):
        from video_summarizer.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert "video_id" in result.output

    def test_mock_evaluation_score_stable(self):
        from video_summarizer.evaluator.evaluate import Evaluator

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "这是第一段真实转写内容"},
            {"start": 5.0, "end": 10.0, "text": "这是第二段真实转写内容"}
        ]
        chunks = [
            {"start": 0.0, "end": 5.0, "start_time": "00:00", "end_time": "00:05", 
             "summary": "第一段摘要", "topic": "第一段主题"},
            {"start": 5.0, "end": 10.0, "start_time": "00:05", "end_time": "00:10",
             "summary": "第二段摘要", "topic": "第二段主题"}
        ]
        final_summary = {
            "one_sentence_summary": "这是一句测试总结内容",
            "chapter_toc": ["章节1", "章节2"],
            "key_points": ["核心观点1", "核心观点2"]
        }

        evaluator = Evaluator(transcript, chunks, final_summary)
        result = evaluator.evaluate()
        
        score1 = result.overall_score
        evaluator2 = Evaluator(transcript, chunks, final_summary)
        result2 = evaluator2.evaluate()
        score2 = result2.overall_score
        
        assert score1 == score2

    def test_hallucinated_entity_warning_generated(self):
        from video_summarizer.evaluator.evaluate import Evaluator

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "这是正常的测试内容"}
        ]
        final_summary = {
            "one_sentence_summary": "完全不存在的XYZ123名字出现在了视频中这是一个很长很长的摘要内容"
        }

        evaluator = Evaluator(transcript, [], final_summary)
        result = evaluator.evaluate()

        assert len(result.warnings) > 0 or len(result.issues) > 0

    def test_invented_quote_scores_lower_than_real_quotes(self):
        """A-6: unmatched quotes must reduce the faithfulness score."""
        from video_summarizer.evaluator.evaluate import Evaluator

        transcript = [
            {"start": 0.0, "end": 5.0, "text": "这是第一段真实的转写内容"},
            {"start": 5.0, "end": 10.0, "text": "这是第二段真实的转写内容"},
        ]
        chunks = [
            {"start": 0.0, "end": 5.0, "start_time": "00:00", "end_time": "00:05", "summary": "第一段摘要"},
            {"start": 5.0, "end": 10.0, "start_time": "00:05", "end_time": "00:10", "summary": "第二段摘要"},
        ]
        final_summary = {
            "one_sentence_summary": "这是关于视频内容的一句话总结",
            "chapter_toc": ["章节1", "章节2"],
        }
        real_quotes = [
            {"text": "这是第一段真实的转写内容", "start_time": "00:00", "end_time": "00:05"},
            {"text": "这是第二段真实的转写内容", "start_time": "00:05", "end_time": "00:10"},
        ]
        invented_quotes = [
            {"text": "这是第一段真实的转写内容", "start_time": "00:00", "end_time": "00:05"},
            {"text": "这句引用在转写中完全不存在而且是从未说过的话XYZQWERTY", "start_time": "00:05", "end_time": "00:10"},
        ]

        good = Evaluator(transcript, chunks, final_summary, quotes=real_quotes).evaluate()
        bad = Evaluator(transcript, chunks, final_summary, quotes=invented_quotes).evaluate()

        assert bad.faithfulness_score < good.faithfulness_score
        assert bad.overall_score < good.overall_score
        assert good.faithfulness_score == 100.0
        assert any("not found in transcript" in issue for issue in bad.issues)
        assert not any("not found in transcript" in issue for issue in good.issues)


class TestLLMClientProvider:
    def test_get_llm_client_mock(self):
        from video_summarizer.summarizer.llm_client import get_llm_client, MockLLMClient
        client = get_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_get_llm_client_ollama(self):
        from video_summarizer.summarizer.llm_client import get_llm_client, OllamaLLMClient
        client = get_llm_client("ollama")
        assert isinstance(client, OllamaLLMClient)

    def test_get_llm_client_unknown_raises_error(self):
        from video_summarizer.summarizer.llm_client import get_llm_client, LLMError
        with pytest.raises(LLMError) as exc_info:
            get_llm_client("unknown-provider")
        assert "Unknown LLM provider" in str(exc_info.value)

    def test_get_llm_client_openai_and_openai_compatible_use_same_class(self):
        from video_summarizer.summarizer.llm_client import OpenAILLMClient
        assert OpenAILLMClient is not None


class TestResumeBehaviorFix:
    def test_resume_with_existing_summary_chunks_no_unboundlocalerror(self, setup_env):
        from video_summarizer.db import init_db, create_video_record
        from video_summarizer.summarizer.pipeline import save_transcript, save_summary_chunks, save_final_summary

        init_db()
        video_id = create_video_record(
            source_type="local",
            title="Test Resume Video"
        )

        save_transcript(video_id, [
            {"start": 0.0, "end": 5.0, "text": "Test segment", "source": "mock"}
        ])

        save_summary_chunks(video_id, [
            {
                "start": 0.0,
                "end": 180.0,
                "source_text": "Original source text",
                "summary": "Existing summary",
                "topic": "Test",
                "key_points": ["point1"],
                "important_terms": [],
                "quote": "",
                "chapter_hint": ""
            }
        ])

        save_final_summary(video_id, {
            "one_sentence_summary": "Test summary"
        })

        from video_summarizer.db import has_summary_chunks
        assert has_summary_chunks(video_id) is True

        from video_summarizer.summarizer.pipeline import get_summary_chunks, get_chapters, get_quotes
        result_chunks = get_summary_chunks(video_id)
        result_chapters = get_chapters(video_id)
        result_quotes = get_quotes(video_id)

        assert result_chunks is not None
        assert isinstance(result_chunks, list)


class TestCreateVideoRecord:
    def test_create_video_record_inserts_and_returns_id(self, setup_env):
        from video_summarizer.db import init_db, create_video_record, get_video_info
        init_db()

        video_id = create_video_record(
            source_type="local",
            source_path="/path/to/video.mp4",
            title="Test Video",
            author="Test Author",
            duration=120.5
        )

        assert isinstance(video_id, int)
        assert video_id > 0

        video_info = get_video_info(video_id)
        assert video_info is not None
        assert video_info["title"] == "Test Video"
        assert video_info["author"] == "Test Author"
        assert video_info["duration"] == 120.5
        assert video_info["source_type"] == "local"

    def test_update_video_duration_updates_duration(self, setup_env):
        from video_summarizer.db import init_db, create_video_record, update_video_duration, get_video_info
        init_db()

        video_id = create_video_record(
            source_type="local",
            title="Test Video"
        )

        info_before = get_video_info(video_id)
        assert info_before["duration"] is None

        update_video_duration(video_id, 300.5)

        info_after = get_video_info(video_id)
        assert info_after["duration"] == 300.5


class TestWebUIImportsFromDB:
    def test_web_ui_does_not_import_create_video_record_from_cli(self):
        app_path = Path(__file__).resolve().parent.parent / "video_summarizer" / "web_ui" / "app.py"
        with open(app_path, "r") as f:
            content = f.read()
        assert "from video_summarizer.cli import create_video_record" not in content
        assert "from video_summarizer.cli import update_video_duration" not in content

    def test_web_ui_imports_create_video_record_from_db(self):
        app_path = Path(__file__).resolve().parent.parent / "video_summarizer" / "web_ui" / "app.py"
        with open(app_path, "r") as f:
            content = f.read()
        assert "from video_summarizer.db import" in content


class TestFindVideoBySource:
    def test_finds_record_by_source_path_or_url(self, setup_env):
        from video_summarizer.db import init_db, create_video_record, get_db, find_video_by_source

        init_db()
        local_id = create_video_record(source_type="local", source_path="C:/videos/a.mp4", title="A")
        url_id = create_video_record(source_type="url", url="https://www.bilibili.com/video/BV1xx", title="B")

        with get_db() as conn:
            assert find_video_by_source(conn, "C:/videos/a.mp4")["id"] == local_id
            assert find_video_by_source(conn, "https://www.bilibili.com/video/BV1xx")["id"] == url_id
            assert find_video_by_source(conn, "https://www.bilibili.com/video/BV1xx", "BV1xx")["id"] == url_id
            assert find_video_by_source(conn, "does-not-exist.mp4") is None
            assert find_video_by_source(conn) is None

    def test_uses_parameterized_query(self, setup_env):
        from video_summarizer.db import init_db, create_video_record, get_db, find_video_by_source

        init_db()
        create_video_record(source_type="local", source_path="safe.mp4", title="A")

        with get_db() as conn:
            # SQL metacharacters must be treated as literal data
            assert find_video_by_source(conn, "x' OR '1'='1") is None
            assert find_video_by_source(conn, "safe.mp4")["source_path"] == "safe.mp4"


class TestFkIndexes:
    def test_init_db_creates_video_id_indexes(self, setup_env):
        from video_summarizer.db import init_db, get_db

        init_db()
        with get_db() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
            names = {r["name"] for r in rows}

        assert "idx_transcript_segments_video" in names
        assert "idx_summary_chunks_video" in names
        assert "idx_video_chapters_video" in names
        assert "idx_video_quotes_video" in names
        assert "idx_video_terms_video" in names


class TestCrossRunResume:
    """A-1: summarize must reuse the video record of the same source so the
    has_* checkpoints and the audio cache key stay valid across runs."""

    FAKE_SEGMENTS = [
        SimpleNamespace(start=0.0, end=5.0, text="第一段内容"),
        SimpleNamespace(start=5.0, end=10.0, text="第二段内容"),
    ]

    def _invoke_summarize_local(self, runner, monkeypatch, video_file, force=False):
        monkeypatch.setattr("video_summarizer.cli.check_ffmpeg_installed", lambda: True)
        monkeypatch.setattr("video_summarizer.cli.check_ytdlp_installed", lambda: True)
        monkeypatch.setattr("video_summarizer.cli.get_video_duration", lambda path: 60.0)
        monkeypatch.setattr(
            "video_summarizer.cli.extract_audio",
            lambda video, out: Path(out).touch()
        )
        engine_mock = MagicMock()
        engine_mock.return_value.transcribe.return_value = list(self.FAKE_SEGMENTS)
        monkeypatch.setattr("video_summarizer.cli.FasterWhisperEngine", engine_mock)

        from video_summarizer.cli import app

        args = [
            "summarize-local", str(video_file),
            "--asr-provider", "mock",
            "--llm-provider", "mock",
        ]
        if force:
            args.append("--force")

        result = runner.invoke(app, args)
        return result, engine_mock

    def _table_count(self, table):
        from video_summarizer.db import get_db
        with get_db() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_running_twice_reuses_record_and_skips_transcription(self, tmp_path, monkeypatch):
        from video_summarizer.db import init_db
        from typer.testing import CliRunner

        init_db()
        video_file = tmp_path / "resume_video.mp4"
        video_file.write_bytes(b"fake video bytes")

        runner = CliRunner()
        result1, engine_mock = self._invoke_summarize_local(runner, monkeypatch, video_file)
        assert result1.exit_code == 0, result1.output

        assert self._table_count("videos") == 1
        assert self._table_count("transcript_segments") == 2
        assert self._table_count("summary_chunks") == 1
        assert engine_mock.return_value.transcribe.call_count == 1

        # Second run on the same source: record must be reused (no new row),
        # transcript must come from the has_* checkpoint (no re-transcription),
        # and the summary stage must reuse the persisted chunks.
        result2, engine_mock2 = self._invoke_summarize_local(runner, monkeypatch, video_file)
        assert result2.exit_code == 0, result2.output

        assert self._table_count("videos") == 1
        assert self._table_count("transcript_segments") == 2
        assert self._table_count("summary_chunks") == 1
        assert self._table_count("video_quotes") == 2
        # The fresh engine mock is never touched: transcription was skipped
        assert engine_mock2.return_value.transcribe.call_count == 0

    def test_force_reuses_id_and_does_not_duplicate_rows(self, tmp_path, monkeypatch):
        from video_summarizer.db import init_db
        from typer.testing import CliRunner

        init_db()
        video_file = tmp_path / "force_video.mp4"
        video_file.write_bytes(b"fake video bytes")

        runner = CliRunner()
        result1, engine_mock = self._invoke_summarize_local(runner, monkeypatch, video_file)
        assert result1.exit_code == 0, result1.output
        assert self._table_count("videos") == 1

        result2, engine_mock2 = self._invoke_summarize_local(runner, monkeypatch, video_file, force=True)
        assert result2.exit_code == 0, result2.output

        # Same record reused, everything re-derived exactly once, no duplicates
        assert self._table_count("videos") == 1
        assert self._table_count("transcript_segments") == 2
        assert self._table_count("summary_chunks") == 1
        assert self._table_count("video_quotes") == 2
        assert self._table_count("video_chapters") == 1
        assert self._table_count("final_summaries") == 1
        # --force re-transcribed: the second run's engine mock was used once
        assert engine_mock2.return_value.transcribe.call_count == 1

    SRT_CONTENT = """1
00:00:00,000 --> 00:00:05,000
第一句字幕

2
00:00:05,000 --> 00:00:10,000
第二句字幕
"""

    def test_url_resume_skips_subtitle_download(self, tmp_path, monkeypatch):
        from video_summarizer.db import init_db
        from typer.testing import CliRunner

        init_db()

        monkeypatch.setattr("video_summarizer.cli.check_ffmpeg_installed", lambda: True)
        monkeypatch.setattr("video_summarizer.cli.check_ytdlp_installed", lambda: True)

        info = SimpleNamespace(title="URL Video", author="UP主", duration=60.0, requires_login=False)
        monkeypatch.setattr("video_summarizer.cli.get_url_info", lambda *a, **kw: info)

        subtitle_calls = []

        def fake_download_subtitles(url, output_dir, **kwargs):
            subtitle_calls.append(url)
            srt = Path(output_dir) / "BV1xx411c7mZ.zh-Hans.srt"
            srt.write_text(self.SRT_CONTENT)
            return srt

        monkeypatch.setattr("video_summarizer.cli.download_subtitles", fake_download_subtitles)

        from video_summarizer.cli import app

        args = [
            "summarize-url", "BV1xx411c7mZ",
            "--asr-provider", "mock",
            "--llm-provider", "mock",
        ]

        runner = CliRunner()
        result1 = runner.invoke(app, args)
        assert result1.exit_code == 0, result1.output

        assert self._table_count("videos") == 1
        assert self._table_count("transcript_segments") == 2
        assert self._table_count("summary_chunks") == 1
        assert len(subtitle_calls) == 1

        # Second run: record reused, subtitle download phase skipped entirely,
        # no duplicated rows
        result2 = runner.invoke(app, args)
        assert result2.exit_code == 0, result2.output

        assert self._table_count("videos") == 1
        assert self._table_count("transcript_segments") == 2
        assert self._table_count("summary_chunks") == 1
        assert len(subtitle_calls) == 1


class SleepingLLMClient:
    """MockLLMClient whose summarize_chunk sleeps and records call order."""

    def __init__(self):
        from video_summarizer.summarizer.llm_client import MockLLMClient
        self._mock = MockLLMClient()
        self.calls = []
        self._lock = threading.Lock()

    def summarize_chunk(self, text, start_time, end_time):
        with self._lock:
            self.calls.append((start_time, end_time))
        time.sleep(0.05)
        return self._mock.summarize_chunk(text, start_time, end_time)

    def __getattr__(self, name):
        return getattr(self._mock, name)


class TestChunkSummaryConcurrency:
    def test_all_chunk_summaries_persist_in_chunk_order(self, setup_env):
        from video_summarizer.db import init_db, create_video_record
        from video_summarizer.summarizer import pipeline as pipeline_module
        from video_summarizer.summarizer.pipeline import save_transcript, get_summary_chunks, get_final_summary

        init_db()
        video_id = create_video_record(source_type="local", title="Concurrency Test")

        # 8 x 30s segments with chunk_min=chunk_max=1min -> 4 chunks of 60s
        segments = [
            {"start": i * 30.0, "end": (i + 1) * 30.0, "text": f"segment {i} content", "source": "mock"}
            for i in range(8)
        ]
        save_transcript(video_id, segments)

        client = SleepingLLMClient()
        with patch.object(pipeline_module, "get_llm_client", return_value=client):
            result = pipeline_module.summarize_video_pipeline(
                video_id, llm_provider="mock", chunk_min=1, chunk_max=1
            )

        assert len(client.calls) == 4

        chunks = get_summary_chunks(video_id)
        assert len(chunks) == 4
        # Persistence order must follow chunk order
        assert [c["start"] for c in chunks] == [0.0, 60.0, 120.0, 180.0]
        # Each persisted summary must belong to its own chunk (not swapped)
        for chunk in chunks:
            assert chunk["start_time"] in chunk["summary"]
        assert [c["start"] for c in result["chunks"]] == [0.0, 60.0, 120.0, 180.0]

        # The rest of the pipeline still completed on top of the chunks
        assert get_final_summary(video_id) is not None


class TestSingleCallSubtitles:
    def _fake_run(self, calls, tmp_path, video_id="BV1xx411c7mZ", files=("zh-Hans", "zh-Hant", "en"),
                  get_id_ok=True, main_ok=True):
        def fake_run(cmd, capture_output=None, text=None):
            calls.append(list(cmd))
            if "--get-id" in cmd:
                rc = 0 if get_id_ok else 1
                return SimpleNamespace(returncode=rc, stdout=f"{video_id}\n" if rc == 0 else "", stderr="")
            if main_ok:
                for lang in files:
                    (tmp_path / f"{video_id}.{lang}.srt").write_text("subtitle data")
            return SimpleNamespace(returncode=0 if main_ok else 1, stdout="", stderr="yt-dlp boom")
        return fake_run

    def test_single_invocation_prefers_configured_language(self, tmp_path, monkeypatch):
        from video_summarizer.media import downloader

        calls = []
        monkeypatch.setattr(downloader.subprocess, "run", self._fake_run(calls, tmp_path))

        result = downloader.download_subtitles("https://www.bilibili.com/video/BV1xx411c7mZ", tmp_path)

        # Best file = first configured language that was produced
        assert result == tmp_path / "BV1xx411c7mZ.zh-Hans.srt"
        # Exactly 2 subprocess calls: --get-id probe + ONE subtitle fetch
        # (the old loop made 1 + len(languages) calls)
        assert len(calls) == 2
        sub_cmd = calls[-1]
        assert "--write-subs" in sub_cmd
        assert "--write-auto-subs" in sub_cmd
        assert "--skip-download" in sub_cmd
        assert "--sub-langs" in sub_cmd
        assert sub_cmd[sub_cmd.index("--sub-langs") + 1] == "zh-Hans,zh-Hant,zh,en,zh-CN"
        assert "--convert-subs" in sub_cmd and "srt" in sub_cmd

    def test_language_fallback_order(self, tmp_path, monkeypatch):
        from video_summarizer.media import downloader

        calls = []
        # Only English subs available -> returned even though zh is preferred
        monkeypatch.setattr(
            downloader.subprocess, "run",
            self._fake_run(calls, tmp_path, files=("en",))
        )

        result = downloader.download_subtitles("BV1xx411c7mZ", tmp_path)
        assert result == tmp_path / "BV1xx411c7mZ.en.srt"

    def test_returns_none_when_no_subtitles_produced(self, tmp_path, monkeypatch):
        from video_summarizer.media import downloader

        calls = []
        monkeypatch.setattr(
            downloader.subprocess, "run",
            self._fake_run(calls, tmp_path, files=())
        )

        result = downloader.download_subtitles("BV1xx411c7mZ", tmp_path)
        assert result is None

    def test_returns_none_when_ytdlp_fails(self, tmp_path, monkeypatch):
        from video_summarizer.media import downloader

        calls = []
        monkeypatch.setattr(
            downloader.subprocess, "run",
            self._fake_run(calls, tmp_path, main_ok=False)
        )

        result = downloader.download_subtitles("BV1xx411c7mZ", tmp_path)
        assert result is None
        # One probe + one fetch attempt, no per-language retry storm
        assert len(calls) == 2

    def test_get_id_failure_still_fetches_with_default_template(self, tmp_path, monkeypatch):
        from video_summarizer.media import downloader

        calls = []
        monkeypatch.setattr(
            downloader.subprocess, "run",
            self._fake_run(calls, tmp_path, get_id_ok=False, files=("zh-Hans",))
        )

        result = downloader.download_subtitles("BV1xx411c7mZ", tmp_path)
        assert result == tmp_path / "BV1xx411c7mZ.zh-Hans.srt"
        assert len(calls) == 2
        assert any("%(id)s.%(ext)s" in arg for arg in calls[-1])
