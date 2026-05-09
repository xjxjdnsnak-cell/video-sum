import json
import os
import re
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from openai import OpenAI

from ..config import settings
from .prompts import (
    CHUNK_SUMMARY_PROMPT,
    FINAL_SUMMARY_PROMPT,
)


class LLMError(Exception):
    pass


class BaseLLMClient(ABC):
    @abstractmethod
    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> str:
        pass

    @abstractmethod
    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: list[dict]
    ) -> Dict[str, str]:
        pass


class MockLLMClient(BaseLLMClient):
    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> str:
        word_count = len(text.split())
        return f"[Mock] 这是一段约 {word_count} 字的视频内容摘要，时间范围 {start_time} - {end_time}。内容主要讨论了视频中的核心要点。"

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: list[dict]
    ) -> Dict[str, str]:
        return {
            "one_sentence_summary": f"[Mock] 这是一个关于「{video_title}」的视频总结。",
            "detailed_summary": "[Mock] 详细总结：视频涵盖了多个重要主题，进行了深入的讨论和分析。\n\n" + "\n\n".join(
                [f"第{i+1}部分: {s['summary']}" for i, s in enumerate(chunk_summaries)]
            ),
            "key_points": "- 要点1: 视频的核心主题\n- 要点2: 重要的讨论内容\n- 要点3: 实用的建议和方法\n- 要点4: 关键结论和总结",
            "questions": "1. 视频的主要观点是什么？\n2. 有哪些重要的细节需要注意？\n3. 如何将视频内容应用到实践中？"
        }


def parse_summary_response(response: str) -> Dict[str, str]:
    json_match = re.search(r'\{[^{}]*"[^"]*":\s*"[^"]*"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "one_sentence_summary": data.get("one_sentence_summary", ""),
                "detailed_summary": data.get("detailed_summary", ""),
                "key_points": data.get("key_points", ""),
                "questions": data.get("questions", "")
            }
        except json.JSONDecodeError:
            pass

    return {
        "one_sentence_summary": response[:100] if len(response) > 100 else response,
        "detailed_summary": response,
        "key_points": "- 关键信息已提取",
        "questions": "1. 视频的核心内容是什么？"
    }


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
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}")

    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> str:
        user_prompt = CHUNK_SUMMARY_PROMPT.format(
            start_time=start_time,
            end_time=end_time,
            text=text
        )
        return self._call_llm(
            "你是一个专业的视频内容摘要助手。",
            user_prompt
        )

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: list[dict]
    ) -> Dict[str, str]:
        summaries_text = "\n\n".join([
            f"时间段 {s['start_time']} - {s['end_time']}:\n{s['summary']}"
            for s in chunk_summaries
        ])

        user_prompt = FINAL_SUMMARY_PROMPT.format(
            video_title=video_title,
            chunk_summaries=summaries_text
        )

        response = self._call_llm(
            "你是一个专业的视频内容摘要助手，擅长提取关键信息和知识点。",
            user_prompt
        )

        return parse_summary_response(response)


class OllamaLLMClient(BaseLLMClient):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        base_url = base_url or settings.OLLAMA_BASE_URL
        model = model or settings.OLLAMA_MODEL

        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            base_url=f"{base_url}/api",
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
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Ollama API call failed: {e}")

    def summarize_chunk(self, text: str, start_time: str, end_time: str) -> str:
        user_prompt = CHUNK_SUMMARY_PROMPT.format(
            start_time=start_time,
            end_time=end_time,
            text=text
        )
        return self._call_llm(
            "你是一个专业的视频内容摘要助手。",
            user_prompt
        )

    def generate_final_summary(
        self,
        video_title: str,
        chunk_summaries: list[dict]
    ) -> Dict[str, str]:
        summaries_text = "\n\n".join([
            f"时间段 {s['start_time']} - {s['end_time']}:\n{s['summary']}"
            for s in chunk_summaries
        ])

        user_prompt = FINAL_SUMMARY_PROMPT.format(
            video_title=video_title,
            chunk_summaries=summaries_text
        )

        response = self._call_llm(
            "你是一个专业的视频内容摘要助手，擅长提取关键信息和知识点。",
            user_prompt
        )

        return parse_summary_response(response)


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    provider = provider or settings.LLM_PROVIDER

    if provider == "mock":
        return MockLLMClient()
    elif provider == "openai":
        return OpenAILLMClient()
    elif provider == "ollama":
        return OllamaLLMClient()
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")
