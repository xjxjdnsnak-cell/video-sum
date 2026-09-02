import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSearchImports:
    def test_search_module_importable(self):
        from video_summarizer.search import (
            SearchResult, Evidence, check_fts5_support,
            init_fts_tables, rebuild_all_indexes,
            search_fts, search_like, get_evidence,
            get_transcript_for_qa, get_summary_for_qa,
            get_final_summary_for_qa, generate_qa_prompt, parse_qa_response
        )
        assert SearchResult is not None
        assert Evidence is not None

    def test_fts5_support_check(self):
        from video_summarizer.search import check_fts5_support
        result = check_fts5_support()
        assert isinstance(result, bool)


class TestSearchFunctions:
    def test_bm25_like_scoring(self):
        from video_summarizer.search.evidence_retriever import _bm25_score
        score1 = _bm25_score("hello world test", ["hello", "world"])
        score2 = _bm25_score("hello python", ["hello", "world"])
        assert score1 > score2

    def test_prepare_query(self):
        from video_summarizer.search.evidence_retriever import _prepare_query
        result = _prepare_query("hello world")
        assert "hello" in result
        assert "world" in result


class TestQAPrompt:
    def test_generate_qa_prompt(self):
        from video_summarizer.search import generate_qa_prompt
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "This is a test"}
        ]
        summaries = [{"start": 0.0, "end": 10.0, "text": "Summary"}]
        
        prompt = generate_qa_prompt("What is this about?", transcript, summaries)
        assert "What is this about?" in prompt
        assert "Hello world" in prompt
        assert "Summary" in prompt

    def test_generate_qa_prompt_wraps_untrusted_data(self):
        """S-5: transcript/summary text must sit inside explicit data markers."""
        from video_summarizer.search import generate_qa_prompt
        from video_summarizer.summarizer.prompts import UNTRUSTED_DATA_BEGIN, UNTRUSTED_DATA_END, UNTRUSTED_DATA_RULE

        transcript = [{"start": 0.0, "end": 5.0, "text": "Hello world"}]
        summaries = [{"start": 0.0, "end": 5.0, "text": "Summary"}]
        final_summary = {"one_sentence_summary": "一句话总结"}

        prompt = generate_qa_prompt("What is this about?", transcript, summaries, final_summary)

        assert UNTRUSTED_DATA_BEGIN in prompt
        assert UNTRUSTED_DATA_END in prompt
        assert UNTRUSTED_DATA_RULE in prompt
        # One delimited block per untrusted section: transcript, summaries,
        # final summary. (The rule sentence itself also names the markers, so
        # anchor on the newline that starts each delimited block.)
        assert prompt.count(f"{UNTRUSTED_DATA_BEGIN}\n") == 3
        assert prompt.count(f"\n{UNTRUSTED_DATA_END}") == 3

    def test_parse_qa_response_with_json(self):
        from video_summarizer.search import parse_qa_response
        response = '''
        {
            "answer": "Test answer",
            "evidence": [{"timestamp": 1.5, "text": "evidence"}],
            "cited_timestamps": [1.5],
            "uncertainty": false
        }
        '''
        result = parse_qa_response(response)
        assert result["answer"] == "Test answer"
        assert result["uncertainty"] is False
        assert 1.5 in result["cited_timestamps"]

    def test_parse_qa_response_uncertainty(self):
        from video_summarizer.search import parse_qa_response
        response = "不确定，当前转写中没有足够证据"
        result = parse_qa_response(response)
        assert result["uncertainty"] is True
        assert "证据" in result["answer"]


class TestSearchWithDatabase:
    def test_init_fts_tables(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import init_fts_tables
        init_fts_tables()

    def test_search_like_empty(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import search_like
        results = search_like("test", None, 10)
        assert isinstance(results, list)

    def test_search_fts_empty(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import check_fts5_support, search_like
        if check_fts5_support():
            from video_summarizer.search import init_fts_tables, search_fts
            init_fts_tables()
            results = search_fts("test", None, 10)
            assert isinstance(results, list)
        else:
            results = search_like("test", None, 10)
            assert isinstance(results, list)


class TestEvidenceRetrieval:
    def test_get_transcript_for_qa(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import get_transcript_for_qa
        result = get_transcript_for_qa(999)
        assert isinstance(result, list)

    def test_get_summary_for_qa(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import get_summary_for_qa
        result = get_summary_for_qa(999)
        assert isinstance(result, list)

    def test_get_final_summary_for_qa(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.search import get_final_summary_for_qa
        result = get_final_summary_for_qa(999)
        assert result is None


class TestCLISearchCommand:
    def test_cli_has_search_command(self):
        from video_summarizer.cli import app
        commands = list(app.registered_commands)
        command_names = [c.name for c in commands]
        assert "search" in command_names

    def test_cli_has_ask_command(self):
        from video_summarizer.cli import app
        commands = list(app.registered_commands)
        command_names = [c.name for c in commands]
        assert "ask" in command_names

    def test_cli_has_rebuild_index_command(self):
        from video_summarizer.cli import app
        commands = list(app.registered_commands)
        command_names = [c.name for c in commands]
        assert "rebuild-index" in command_names


class TestSearchWithMockData:
    def test_search_with_video_id_filter(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.cli import create_video_record
        from video_summarizer.summarizer.pipeline import save_transcript
        from video_summarizer.search import init_fts_tables, search_like
        
        video_id = create_video_record(
            source_type="local",
            title="Test Video"
        )
        
        save_transcript(video_id, [
            {"start": 0.0, "end": 5.0, "text": "Hello world test content", "source": "mock"},
            {"start": 5.0, "end": 10.0, "text": "Another segment", "source": "mock"}
        ])
        
        init_fts_tables()
        
        results = search_like("test", video_id, 10)
        assert len(results) >= 1
        assert results[0].video_id == video_id

    def test_ask_returns_timestamps_from_transcript(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.cli import create_video_record
        from video_summarizer.summarizer.pipeline import save_transcript
        from video_summarizer.search import (
            get_transcript_for_qa, generate_qa_prompt, parse_qa_response
        )
        
        video_id = create_video_record(
            source_type="local",
            title="Test Video"
        )
        
        transcript_data = [
            {"start": 0.0, "end": 5.0, "text": "Hello world", "source": "mock"},
            {"start": 5.0, "end": 10.0, "text": "This is a test video", "source": "mock"}
        ]
        save_transcript(video_id, transcript_data)
        
        transcript = get_transcript_for_qa(video_id)
        assert len(transcript) == 2
        assert transcript[0]["start"] == 0.0
        
        prompt = generate_qa_prompt("What is this?", transcript, [])
        assert "Hello world" in prompt
        
        mock_response = '{"answer": "Test", "evidence": [{"timestamp": 5.0, "text": "test"}], "cited_timestamps": [5.0], "uncertainty": false}'
        result = parse_qa_response(mock_response)
        assert 5.0 in result["cited_timestamps"]

    def test_ask_uncertainty_response(self, setup_env):
        from video_summarizer.search import parse_qa_response
        
        response = "这个问题无法确定答案，证据不足"
        result = parse_qa_response(response)
        assert result["uncertainty"] is True
        assert len(result["cited_timestamps"]) == 0


class TestRebuildIndex:
    def test_rebuild_all_indexes(self, setup_env):
        from video_summarizer.db import init_db
        init_db()
        from video_summarizer.cli import create_video_record
        from video_summarizer.summarizer.pipeline import save_transcript
        from video_summarizer.search import (
            init_fts_tables, rebuild_all_indexes,
            check_fts5_support
        )
        
        video_id = create_video_record(
            source_type="local",
            title="Test Video"
        )
        save_transcript(video_id, [
            {"start": 0.0, "end": 5.0, "text": "Test content", "source": "mock"}
        ])
        
        if check_fts5_support():
            init_fts_tables()
            rebuild_all_indexes()
        else:
            pytest.skip("FTS5 not supported")


class TestWebUISearchModule:
    def test_web_ui_search_module_import(self):
        from video_summarizer.web_ui import app
        assert hasattr(app, 'render_search_page')
        assert hasattr(app, 'render_qa_page')

    def test_web_ui_can_import_search_functions(self):
        from video_summarizer.web_ui.app import (
            check_fts5_support, init_fts_tables, search_fts,
            search_like, get_evidence, generate_qa_prompt, parse_qa_response
        )
        assert callable(check_fts5_support)
        assert callable(search_fts)
        assert callable(search_like)
