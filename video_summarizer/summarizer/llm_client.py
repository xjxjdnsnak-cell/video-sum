import json
import os
import re
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from openai import OpenAI

from ..config import settings
from .prompts import (
    CHUNK_SUMMARY_PROMPT,
    FINAL_SUMMARY_PROMPTS,
    NoteStyle,
    QUOTE_EXTRACTION_PROMPT,
    CHAPTER_AGGREGATION_PROMPT,
    TERM_EXTRACTION_PROMPT,
    UNTRUSTED_DATA_RULE,
    wrap_untrusted_text,
)


class LLMError(Exception):
    pass


# Hard cap for a single LLM round-trip, in seconds. Without it the OpenAI SDK
# default (600s) can hang a whole chunk (or the whole pipeline, since chunk
# summaries now share a small thread pool) on a stalled connection. Timeouts
# surface as exceptions and keep the existing LLMError error behavior.
LLM_CALL_TIMEOUT_SECONDS = 120

# extract_quotes stuffs the transcript into a single prompt. Capped to the
# last ~12000 characters (roughly 4-6k tokens of English, fewer for CJK) so a
# long video cannot exceed the model context window and fail the whole
# pipeline. We keep the tail because closing/summary statements - the usual
# quote material - live near the end, and we only ever drop whole
# timestamp-prefixed lines from the front, so remaining lines stay intact.
QUOTE_CONTEXT_MAX_CHARS = 12000


def bounded_transcript_text(transcript: List[Dict]) -> str:
    """Format transcript segments as timestamped lines, keeping only as many
    trailing lines as fit within QUOTE_CONTEXT_MAX_CHARS."""
    lines = [
        f"[{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}] {seg['text']}"
        for seg in transcript
    ]
    selected: List[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line) + 1  # +1 for the joining newline
        if selected and total + cost > QUOTE_CONTEXT_MAX_CHARS:
            break
        selected.append(line)
        total += cost
    return "\n".join(reversed(selected))


class BaseLLMClient(ABC):
    @abstractmethod
    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> Dict:
        pass

    @abstractmethod
    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: List[Dict],
        note_style: NoteStyle = NoteStyle.DETAILED
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_quotes(self, transcript: List[Dict]) -> List[Dict]:
        pass

    @abstractmethod
    def aggregate_chapters(
        self,
        chunk_summaries: List[Dict]
    ) -> List[Dict]:
        pass

    @abstractmethod
    def extract_terms(
        self,
        video_title: str,
        transcript: List[Dict],
        chapter_summaries: List[Dict]
    ) -> List[Dict]:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Single-turn completion: send the prompt as the user message and return the content string."""
        pass


class MockLLMClient(BaseLLMClient):
    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> Dict:
        word_count = len(text.split())
        text_snippet = text[:100] + "..." if len(text) > 100 else text
        
        return {
            "topic": f"时间段 {start_time} - {end_time} 的内容主题",
            "key_points": [
                "核心要点1",
                "核心要点2",
                "核心要点3"
            ],
            "important_terms": [
                "术语1: 相关解释",
                "术语2: 相关解释"
            ],
            "quote": text_snippet,
            "chapter_hint": "主要内容章节",
            "summary": f"[Mock] 这是一段约 {word_count} 字的视频内容摘要，时间范围 {start_time} - {end_time}。内容主要讨论了视频中的核心要点。"
        }

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: List[Dict],
        note_style: NoteStyle = NoteStyle.DETAILED
    ) -> Dict[str, Any]:
        if note_style == NoteStyle.BRIEF:
            return {
                "one_sentence_summary": f"[Mock] 这是一个关于「{video_title}」的视频总结。",
                "key_points": [
                    "重点列表项1",
                    "重点列表项2",
                    "重点列表项3"
                ]
            }
        elif note_style == NoteStyle.STUDY:
            return {
                "one_sentence_summary": f"[Mock] 这是一个关于「{video_title}」的学习型总结。",
                "chapter_toc": ["第一章：基础知识", "第二章：核心概念", "第三章：实践应用"],
                "key_knowledge": ["关键知识点1", "关键知识点2"],
                "terms": [
                    {"term": "术语1", "explanation": "术语1的解释"},
                    {"term": "术语2", "explanation": "术语2的解释"}
                ],
                "review_questions": [
                    "复习问题1",
                    "复习问题2",
                    "复习问题3"
                ],
                "common_mistakes": [
                    "学习时容易犯的错误1",
                    "学习时容易犯的错误2"
                ],
                "action_items": [
                    "可执行的复习行动1",
                    "可执行的复习行动2"
                ]
            }
        elif note_style == NoteStyle.MEETING:
            return {
                "one_sentence_summary": f"[Mock] 这是关于「{video_title}」的会议记录。",
                "topics": ["议题1", "议题2", "议题3"],
                "decisions": ["做出的决定1", "做出的决定2"],
                "action_items": [
                    {"task": "待办事项1", "owner": "责任人1"},
                    {"task": "待办事项2", "owner": "责任人2"}
                ],
                "timeline_summary": "会议各阶段的讨论内容"
            }
        elif note_style == NoteStyle.TUTORIAL:
            return {
                "one_sentence_summary": f"[Mock] 这是关于「{video_title}」的教程总结。",
                "chapter_toc": ["步骤1", "步骤2", "步骤3"],
                "steps": [
                    {
                        "step": "步骤1名称",
                        "description": "步骤1的详细描述",
                        "commands": ["相关命令或代码1", "相关命令或代码2"]
                    },
                    {
                        "step": "步骤2名称",
                        "description": "步骤2的详细描述",
                        "commands": ["相关命令或代码1"]
                    }
                ],
                "key_points": ["关键要点1", "关键要点2"],
                "notes": ["注意事项1", "注意事项2"],
                "prerequisites": ["前置要求1", "前置要求2"]
            }
        else:
            return {
                "one_sentence_summary": f"[Mock] 这是一个关于「{video_title}」的详细总结。",
                "chapter_toc": ["章节1", "章节2", "章节3"],
                "timeline_summary": "\n".join([f"- {s['start_time']} - {s['end_time']}: {s['summary']}" for s in chunk_summaries]),
                "key_points": ["核心观点1", "核心观点2", "核心观点3"],
                "key_knowledge": ["关键知识点1", "关键知识点2"],
                "action_items": ["行动项1", "行动项2"]
            }

    def extract_quotes(self, transcript: List[Dict]) -> List[Dict]:
        quotes = []
        for i, seg in enumerate(transcript[:5]):
            quotes.append({
                "text": seg["text"][:100] if len(seg["text"]) > 100 else seg["text"],
                "start_time": format_timestamp(seg["start"]),
                "end_time": format_timestamp(seg["end"])
            })
        return quotes

    def aggregate_chapters(self, chunk_summaries: List[Dict]) -> List[Dict]:
        if not chunk_summaries:
            return []
        
        chapters = []
        chunk_count = len(chunk_summaries)
        chapter_size = max(1, chunk_count // 3)
        
        for i in range(0, chunk_count, chapter_size):
            end_idx = min(i + chapter_size, chunk_count)
            chapter_chunks = chunk_summaries[i:end_idx]
            
            chapters.append({
                "title": f"章节 {len(chapters) + 1}",
                "start_time": chapter_chunks[0]["start_time"],
                "end_time": chapter_chunks[-1]["end_time"],
                "chunks": list(range(i, end_idx)),
                "summary": f"章节 {len(chapters) + 1} 的简要总结"
            })
        
        return chapters

    def extract_terms(
        self,
        video_title: str,
        transcript: List[Dict],
        chapter_summaries: List[Dict]
    ) -> List[Dict]:
        all_text = " ".join([seg["text"] for seg in transcript[:50]])
        words = all_text.split()[:100]
        
        return [
            {
                "term": "术语1",
                "explanation": "基于视频内容的术语解释",
                "first_seen_time": format_timestamp(transcript[0]["start"]) if transcript else "00:00"
            },
            {
                "term": "术语2",
                "explanation": "基于视频内容的术语解释",
                "first_seen_time": format_timestamp(transcript[5]["start"]) if len(transcript) > 5 else "00:00"
            }
        ]

    def generate(self, prompt: str) -> str:
        return (
            '{"answer": "[Mock] 这是基于视频内容的模拟回答（当前为 Mock 模式，未调用真实 LLM）。", '
            '"evidence": [], "cited_timestamps": [], "uncertainty": false}'
        )


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_json_response(response: str) -> Optional[Dict]:
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or settings.OPENAI_BASE_URL
        model = model or settings.OPENAI_MODEL

        if not api_key:
            raise LLMError("OpenAI API key is required. Set OPENAI_API_KEY in .env or pass it directly.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=LLM_CALL_TIMEOUT_SECONDS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}")

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}")

    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> Dict:
        user_prompt = CHUNK_SUMMARY_PROMPT.format(
            start_time=start_time,
            end_time=end_time,
            text=wrap_untrusted_text(text)
        )
        response = self._call_llm(
            "你是一个专业的视频内容摘要助手。请严格按照JSON格式输出。" + UNTRUSTED_DATA_RULE,
            user_prompt
        )
        
        result = parse_json_response(response)
        if result:
            return result
        
        return {
            "topic": f"{start_time} - {end_time} 的内容",
            "key_points": ["关键观点1", "关键观点2"],
            "important_terms": [],
            "quote": text[:100] if len(text) > 100 else text,
            "chapter_hint": "主要内容",
            "summary": response[:200] if response else "摘要生成失败"
        }

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: List[Dict],
        note_style: NoteStyle = NoteStyle.DETAILED
    ) -> Dict[str, Any]:
        prompt_template = FINAL_SUMMARY_PROMPTS.get(note_style, FINAL_SUMMARY_PROMPTS[NoteStyle.DETAILED])
        
        summaries_text = "\n\n".join([
            f"时间段 {s['start_time']} - {s['end_time']}:\n{s['summary']}"
            for s in chunk_summaries
        ])

        user_prompt = prompt_template.format(
            video_title=video_title,
            chunk_summaries=summaries_text
        )

        response = self._call_llm(
            "你是一个专业的视频内容摘要助手，请严格按照JSON格式输出。",
            user_prompt
        )

        result = parse_json_response(response)
        if result:
            return result
        
        return {
            "one_sentence_summary": f"关于「{video_title}」的总结",
            "key_points": ["要点1", "要点2"],
            "error": "JSON解析失败"
        }

    def extract_quotes(self, transcript: List[Dict]) -> List[Dict]:
        transcript_text = bounded_transcript_text(transcript)

        user_prompt = QUOTE_EXTRACTION_PROMPT.format(transcript=wrap_untrusted_text(transcript_text))
        
        response = self._call_llm(
            "你是一个专业的视频内容分析师，请严格按照JSON格式输出。" + UNTRUSTED_DATA_RULE,
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "quotes" in result:
            return result["quotes"]
        
        return []

    def aggregate_chapters(self, chunk_summaries: List[Dict]) -> List[Dict]:
        summaries_text = "\n".join([
            f"[{i}] {s['start_time']} - {s['end_time']}: {s.get('summary', s.get('topic', ''))}"
            for i, s in enumerate(chunk_summaries)
        ])
        
        user_prompt = CHAPTER_AGGREGATION_PROMPT.format(chunk_summaries=summaries_text)
        
        response = self._call_llm(
            "你是一个专业的视频内容分析师，请严格按照JSON格式输出。",
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "chapters" in result:
            return result["chapters"]
        
        return []

    def extract_terms(
        self,
        video_title: str,
        transcript: List[Dict],
        chapter_summaries: List[Dict]
    ) -> List[Dict]:
        transcript_text = "\n".join([seg["text"] for seg in transcript[:100]])
        chapter_text = "\n".join([s.get("summary", "") for s in chapter_summaries])
        
        user_prompt = TERM_EXTRACTION_PROMPT.format(
            video_title=video_title,
            transcript=transcript_text,
            chapter_summaries=chapter_text
        )
        
        response = self._call_llm(
            "你是一个专业的术语提取助手，请严格按照JSON格式输出。",
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "terms" in result:
            return result["terms"]
        
        return []


class OllamaLLMClient(BaseLLMClient):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        base_url = base_url or settings.OLLAMA_BASE_URL
        model = model or settings.OLLAMA_MODEL

        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama"
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=LLM_CALL_TIMEOUT_SECONDS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Ollama API call failed: {e}")

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Ollama API call failed: {e}")

    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> Dict:
        user_prompt = CHUNK_SUMMARY_PROMPT.format(
            start_time=start_time,
            end_time=end_time,
            text=wrap_untrusted_text(text)
        )
        response = self._call_llm(
            "你是一个专业的视频内容摘要助手。" + UNTRUSTED_DATA_RULE,
            user_prompt
        )
        
        result = parse_json_response(response)
        if result:
            return result
        
        return {
            "topic": f"{start_time} - {end_time} 的内容",
            "key_points": ["关键观点1", "关键观点2"],
            "important_terms": [],
            "quote": text[:100] if len(text) > 100 else text,
            "chapter_hint": "主要内容",
            "summary": response[:200] if response else "摘要生成失败"
        }

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: List[Dict],
        note_style: NoteStyle = NoteStyle.DETAILED
    ) -> Dict[str, Any]:
        prompt_template = FINAL_SUMMARY_PROMPTS.get(note_style, FINAL_SUMMARY_PROMPTS[NoteStyle.DETAILED])
        
        summaries_text = "\n\n".join([
            f"时间段 {s['start_time']} - {s['end_time']}:\n{s['summary']}"
            for s in chunk_summaries
        ])

        user_prompt = prompt_template.format(
            video_title=video_title,
            chunk_summaries=summaries_text
        )

        response = self._call_llm(
            "你是一个专业的视频内容摘要助手。",
            user_prompt
        )

        result = parse_json_response(response)
        if result:
            return result
        
        return {
            "one_sentence_summary": f"关于「{video_title}」的总结",
            "key_points": ["要点1", "要点2"]
        }

    def extract_quotes(self, transcript: List[Dict]) -> List[Dict]:
        transcript_text = bounded_transcript_text(transcript)

        user_prompt = QUOTE_EXTRACTION_PROMPT.format(transcript=wrap_untrusted_text(transcript_text))
        
        response = self._call_llm(
            "你是一个专业的视频内容分析师。" + UNTRUSTED_DATA_RULE,
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "quotes" in result:
            return result["quotes"]
        
        return []

    def aggregate_chapters(self, chunk_summaries: List[Dict]) -> List[Dict]:
        summaries_text = "\n".join([
            f"[{i}] {s['start_time']} - {s['end_time']}: {s.get('summary', s.get('topic', ''))}"
            for i, s in enumerate(chunk_summaries)
        ])
        
        user_prompt = CHAPTER_AGGREGATION_PROMPT.format(chunk_summaries=summaries_text)
        
        response = self._call_llm(
            "你是一个专业的视频内容分析师。",
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "chapters" in result:
            return result["chapters"]
        
        return []

    def extract_terms(
        self,
        video_title: str,
        transcript: List[Dict],
        chapter_summaries: List[Dict]
    ) -> List[Dict]:
        transcript_text = "\n".join([seg["text"] for seg in transcript[:100]])
        chapter_text = "\n".join([s.get("summary", "") for s in chapter_summaries])
        
        user_prompt = TERM_EXTRACTION_PROMPT.format(
            video_title=video_title,
            transcript=transcript_text,
            chapter_summaries=chapter_text
        )
        
        response = self._call_llm(
            "你是一个专业的术语提取助手。",
            user_prompt
        )
        
        result = parse_json_response(response)
        if result and "terms" in result:
            return result["terms"]
        
        return []


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    provider = provider or settings.LLM_PROVIDER

    if provider == "mock":
        return MockLLMClient()
    elif provider in ("openai", "openai-compatible"):
        return OpenAILLMClient()
    elif provider == "ollama":
        return OllamaLLMClient()
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")
