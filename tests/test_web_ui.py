import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWebUIImports:
    def test_web_module_importable(self):
        from video_summarizer.web_ui import main
        assert main is not None

    def test_app_module_importable(self):
        from video_summarizer.web_ui import app
        assert app is not None


class TestWebUIFunctions:
    def test_check_environment_no_ffmpeg(self):
        with patch('video_summarizer.web_ui.app.check_ffmpeg_installed', return_value=False):
            with patch('video_summarizer.web_ui.app.check_ytdlp_installed', return_value=True):
                with patch.dict('sys.modules', {'faster_whisper': MagicMock()}):
                    from video_summarizer.web_ui.app import check_environment
                    errors = check_environment()
                    assert any("FFmpeg" in e for e in errors)

    def test_check_environment_no_ytdlp(self):
        with patch('video_summarizer.web_ui.app.check_ffmpeg_installed', return_value=True):
            with patch('video_summarizer.web_ui.app.check_ytdlp_installed', return_value=False):
                with patch.dict('sys.modules', {'faster_whisper': MagicMock()}):
                    from video_summarizer.web_ui.app import check_environment
                    errors = check_environment()
                    assert any("yt-dlp" in e for e in errors)

    def test_check_environment_no_faster_whisper(self):
        with patch('video_summarizer.web_ui.app.check_ffmpeg_installed', return_value=True):
            with patch('video_summarizer.web_ui.app.check_ytdlp_installed', return_value=True):
                with patch.dict('sys.modules', {'faster_whisper': None}):
                    import importlib
                    import video_summarizer.web_ui.app as app_module
                    with patch.object(app_module, 'check_environment', return_value=["faster-whisper missing"]):
                        from video_summarizer.web_ui.app import check_environment
                        errors = check_environment()
                        assert any("faster-whisper" in e for e in errors)

    def test_get_video_list(self):
        with patch('video_summarizer.web_ui.app.get_all_videos') as mock_get:
            mock_get.return_value = [
                {"id": 1, "title": "Test Video 1", "status": "completed"},
                {"id": 2, "title": "Test Video 2", "status": "failed"}
            ]
            from video_summarizer.web_ui.app import get_video_list
            videos = get_video_list()
            assert len(videos) == 2
            assert videos[0]["title"] == "Test Video 1"

    def test_get_video_details_not_found(self):
        with patch('video_summarizer.web_ui.app.get_video_info', return_value=None):
            from video_summarizer.web_ui.app import get_video_details
            result = get_video_details(999)
            assert result is None


class TestWebUIWithDatabase:
    def test_get_video_list_empty(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.web_ui.app import get_video_list
        videos = get_video_list()
        assert isinstance(videos, list)

    def test_get_video_details_with_mock_data(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        
        from video_summarizer.cli import create_video_record
        
        video_id = create_video_record(
            source_type="local",
            title="Test Video",
            duration=120.0
        )
        
        with patch('video_summarizer.web_ui.app.get_transcript', return_value=[]):
            with patch('video_summarizer.web_ui.app.get_summary_chunks', return_value=[]):
                with patch('video_summarizer.web_ui.app.get_final_summary', return_value=None):
                    with patch('video_summarizer.web_ui.app.get_chapters', return_value=[]):
                        with patch('video_summarizer.web_ui.app.get_quotes', return_value=[]):
                            with patch('video_summarizer.web_ui.app.get_terms', return_value=[]):
                                from video_summarizer.web_ui.app import get_video_details
                                details = get_video_details(video_id)
                                assert details is not None
                                assert details["info"]["title"] == "Test Video"


class TestCLIWebCommand:
    def test_cli_has_web_command(self):
        from video_summarizer.cli import app
        commands = list(app.registered_commands)
        command_names = [c.name for c in commands]
        assert "web" in command_names

    def test_web_command_function_exists(self):
        from video_summarizer.cli import web
        assert callable(web)


class TestEvaluateSummary:
    def test_evaluate_summary_returns_result(self):
        with patch('video_summarizer.web_ui.app.evaluate_video') as mock_evaluate:
            mock_result = MagicMock()
            mock_result.overall_score = 85
            mock_result.warnings = []
            mock_result.suggestions = []
            mock_evaluate.return_value = ("# Report", mock_result)
            
            from video_summarizer.web_ui.app import evaluate_summary
            result, report = evaluate_summary(
                video_id=1,
                transcript=[],
                chunks=[],
                final_summary=None,
                quotes=[],
                terms=[],
                chapters=[]
            )
            assert result.overall_score == 85
            assert "# Report" in report


class TestMockPipeline:
    def test_process_local_video_with_mock_asr(self, setup_env):
        from video_summarizer.db import init_db
        init_db()

        with patch('video_summarizer.media.ffmpeg.get_video_duration') as mock_duration:
            with patch('video_summarizer.media.ffmpeg.extract_audio') as mock_extract:
                with patch('video_summarizer.summarizer.pipeline.save_transcript') as mock_save:
                    with patch('video_summarizer.summarizer.pipeline.summarize_video_pipeline') as mock_summarize:
                        with patch('video_summarizer.exporters.markdown.export_markdown') as mock_export:
                            mock_duration.return_value = 60.0
                            mock_extract.return_value = None
                            mock_save.return_value = None
                            mock_summarize.return_value = {
                                "chunks": [],
                                "final_summary": {},
                                "chapters": [],
                                "quotes": []
                            }
                            mock_export.return_value = "/tmp/test.md"
                            
                            from video_summarizer.web_ui.app import process_local_video
                            
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                                f.write(b'fake video data')
                                temp_path = f.name
                            
                            try:
                                video_id = process_local_video(
                                    video_path=temp_path,
                                    asr_provider="mock",
                                    llm_provider="mock",
                                    whisper_model="tiny",
                                    device="cpu",
                                    language="auto",
                                    note_style="brief",
                                    chunk_min=3,
                                    chunk_max=5,
                                    keep_audio=False
                                )
                                assert video_id is not None
                            finally:
                                os.unlink(temp_path)
