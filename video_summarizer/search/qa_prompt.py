import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


def generate_qa_prompt(
    question: str,
    transcript: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    final_summary: Optional[Dict[str, Any]] = None
) -> str:
    transcript_text = "\n".join(
        f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
        for s in transcript[:50]
    )
    
    summary_text = "\n".join(
        f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
        for s in summaries
    )
    
    prompt = f"""你是一个视频内容问答助手。你的任务是根据提供的转写和摘要内容，回答用户的问题。

重要规则：
1. 你必须基于提供的转写文本和摘要内容来回答问题
2. 回答时必须引用具体的时间戳，格式为 [X.Xs]
3. 如果转写和摘要中没有相关内容，必须明确说"不确定，当前转写中没有足够证据"
4. 禁止凭空编造答案，所有回答必须有转写或摘要作为依据
5. 如果证据不足，可以说明"基于现有转写，只能确认..."而不是猜测

转写内容（按时间顺序，最多50段）：
{transcript_text}

摘要内容：
{summary_text}
"""
    
    if final_summary:
        prompt += f"""
视频总摘要：
一句话总结：{final_summary.get('one_sentence_summary', 'N/A')}
详细总结：{final_summary.get('detailed_summary', 'N/A')}
关键要点：{final_summary.get('key_points', 'N/A')}
常见问题：{final_summary.get('questions', 'N/A')}
"""

    prompt += f"""
用户问题：{question}

请按以下JSON格式回答：
{{
    "answer": "你的回答（必须包含时间戳引用）",
    "evidence": [
        {{"timestamp": 12.5, "text": "引用的原文片段"}},
        {{"timestamp": 45.3, "text": "另一个引用片段"}}
    ],
    "cited_timestamps": [12.5, 45.3],
    "uncertainty": false
}}

如果证据不足：
{{
    "answer": "不确定，当前转写中没有足够证据回答这个问题",
    "evidence": [],
    "cited_timestamps": [],
    "uncertainty": true
}}

请只输出JSON，不要包含其他内容：
"""
    return prompt


def parse_qa_response(response_text: str) -> Dict[str, Any]:
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    
    uncertainty_pattern = r'不确定|没有.*证据|证据不足|无法确定'
    if re.search(uncertainty_pattern, response_text):
        return {
            "answer": "不确定，当前转写中没有足够证据回答这个问题",
            "evidence": [],
            "cited_timestamps": [],
            "uncertainty": True
        }
    
    timestamps = re.findall(r'\[(\d+\.?\d*)s\]', response_text)
    cited_timestamps = [float(t) for t in timestamps[:10]]
    
    return {
        "answer": response_text.strip(),
        "evidence": [],
        "cited_timestamps": cited_timestamps,
        "uncertainty": False
    }


@dataclass
class QAResult:
    answer: str
    evidence: List[Dict[str, Any]]
    cited_timestamps: List[float]
    uncertainty: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "cited_timestamps": self.cited_timestamps,
            "uncertainty": self.uncertainty
        }
