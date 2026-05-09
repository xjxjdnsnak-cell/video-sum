import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        assert "[Mock]" in result
        assert "00:00:00" in result

        final = client.generate_final_summary("Test Video", [
            {"start_time": "00:00", "end_time": "05:00", "summary": "Test summary"}
        ])
        assert "one_sentence_summary" in final
        assert "detailed_summary" in final
        assert "key_points" in final
        assert "questions" in final

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
        assert "视频总结：Test Video" in content
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
                )
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
            output_dir=tmp_path
        )

        content = Path(output_path).read_text()

        assert "视频总结：Test Video" in content
        assert "## 一句话总结" in content
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
