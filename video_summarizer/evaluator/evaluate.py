import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class EvaluationResult:
    overall_score: float
    completeness_score: float
    faithfulness_score: float
    timestamp_accuracy_score: float
    structure_quality_score: float
    note_usefulness_score: float
    hallucination_risk_score: float
    
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "completeness_score": self.completeness_score,
            "faithfulness_score": self.faithfulness_score,
            "timestamp_accuracy_score": self.timestamp_accuracy_score,
            "structure_quality_score": self.structure_quality_score,
            "note_usefulness_score": self.note_usefulness_score,
            "hallucination_risk_score": self.hallucination_risk_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationError:
    rule: str
    message: str
    severity: str  # "error" or "warning"


class TranscriptValidator:
    def __init__(self, transcript: List[Dict]):
        self.transcript = transcript
        self.all_text = " ".join([seg.get("text", "") for seg in transcript])
        self.transcript_by_time: Dict[Tuple[float, float], str] = {
            (seg.get("start", 0), seg.get("end", 0)): seg.get("text", "")
            for seg in transcript
        }

    def quote_exists_in_transcript(self, quote: str, tolerance: float = 0.1) -> bool:
        normalized_quote = self._normalize_text(quote)
        if not normalized_quote:
            return False
        normalized_all = self._normalize_text(self.all_text)
        return normalized_quote in normalized_all

    def time_in_transcript_range(self, start_time: str, end_time: str) -> bool:
        try:
            start_seconds = self._parse_time(start_time)
            end_seconds = self._parse_time(end_time)
            
            for seg in self.transcript:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                if seg_start <= start_seconds <= seg_end and seg_start <= end_seconds <= seg_end:
                    return True
            return False
        except ValueError:
            return False

    def find_text_in_time_range(self, start_time: str, end_time: str) -> Optional[str]:
        try:
            start_seconds = self._parse_time(start_time)
            end_seconds = self._parse_time(end_time)
            
            for seg in self.transcript:
                if abs(seg.get("start", 0) - start_seconds) < 1 and abs(seg.get("end", 0) - end_seconds) < 1:
                    return seg.get("text", "")
            return None
        except ValueError:
            return None

    def extract_entities(self, text: str) -> set:
        entity_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(entity_pattern, text)
        return set(matches)

    def check_entity_in_transcript(self, entity: str) -> bool:
        normalized_entity = self._normalize_text(entity)
        normalized_all = self._normalize_text(self.all_text)
        words = normalized_entity.split()
        if len(words) <= 2:
            return normalized_entity in normalized_all
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in normalized_all:
                return True
        return False

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', '', text.lower())

    def _parse_time(self, time_str: str) -> float:
        parts = time_str.replace(',', ':').split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])


class Evaluator:
    def __init__(self, transcript: List[Dict], chunks: List[Dict], final_summary: Dict, 
                 quotes: List[Dict] = None, terms: List[Dict] = None, chapters: List[Dict] = None):
        self.transcript = transcript
        self.chunks = chunks
        self.final_summary = final_summary
        self.quotes = quotes or []
        self.terms = terms or []
        self.chapters = chapters or []
        
        self.validator = TranscriptValidator(transcript)
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []

    def evaluate(self) -> EvaluationResult:
        completeness = self._evaluate_completeness()
        faithfulness = self._evaluate_faithfulness()
        timestamp_accuracy = self._evaluate_timestamp_accuracy()
        structure_quality = self._evaluate_structure_quality()
        note_usefulness = self._evaluate_note_usefulness()
        hallucination_risk = self._evaluate_hallucination_risk()
        
        self.completeness_score = completeness
        self.faithfulness_score = faithfulness
        self.timestamp_accuracy_score = timestamp_accuracy
        self.structure_quality_score = structure_quality
        self.note_usefulness_score = note_usefulness
        self.hallucination_risk_score = hallucination_risk
        
        overall = (completeness * 0.2 + faithfulness * 0.25 + timestamp_accuracy * 0.15 + 
                  structure_quality * 0.15 + note_usefulness * 0.1 + hallucination_risk * 0.15)
        
        issues = [e.message for e in self.errors]
        warnings = self.warnings + [e.message for e in self.errors if e.severity == "warning"]
        
        suggestions = self._generate_suggestions()
        
        return EvaluationResult(
            overall_score=round(overall, 1),
            completeness_score=round(completeness, 1),
            faithfulness_score=round(faithfulness, 1),
            timestamp_accuracy_score=round(timestamp_accuracy, 1),
            structure_quality_score=round(structure_quality, 1),
            note_usefulness_score=round(note_usefulness, 1),
            hallucination_risk_score=round(hallucination_risk, 1),
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )

    def _evaluate_completeness(self) -> float:
        score = 100.0
        
        if not self.final_summary:
            self.warnings.append("Final summary is empty")
            return 30.0
        
        required_fields = ["one_sentence_summary"]
        missing_fields = []
        for field in required_fields:
            if not self.final_summary.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            score -= 20 * len(missing_fields)
            self.warnings.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        summary_text = self._get_summary_text()
        if len(summary_text) < 50:
            self.warnings.append("Summary is very short (less than 50 characters)")
            score -= 15
        
        if not self.chunks and not self.final_summary.get("chapter_toc"):
            self.warnings.append("No chapter structure found")
            score -= 10
        
        return max(0, min(100, score))

    def _evaluate_faithfulness(self) -> float:
        score = 100.0
        
        if self.errors:
            quote_errors = [e for e in self.errors if "quote" in e.message.lower()]
            score -= 15 * len(quote_errors)
        
        return max(0, min(100, score))

    def _evaluate_timestamp_accuracy(self) -> float:
        score = 100.0
        
        if not self.chunks:
            return 80.0
        
        invalid_timestamps = 0
        for i, chunk in enumerate(self.chunks):
            start_time = chunk.get("start_time", "")
            end_time = chunk.get("end_time", "")
            
            if start_time and end_time:
                if not self.validator.time_in_transcript_range(start_time, end_time):
                    invalid_timestamps += 1
                    self.errors.append(ValidationError(
                        rule="timestamp_range",
                        message=f"Chunk {i+1} timestamp {start_time}-{end_time} does not match transcript range",
                        severity="error"
                    ))
        
        if invalid_timestamps > 0:
            score -= 20 * (invalid_timestamps / len(self.chunks))
        
        return max(0, min(100, score))

    def _evaluate_structure_quality(self) -> float:
        score = 80.0
        
        has_structure = False
        if self.final_summary:
            structure_indicators = ["chapter_toc", "timeline_summary", "key_points", 
                                   "topics", "steps", "decisions"]
            for indicator in structure_indicators:
                if self.final_summary.get(indicator):
                    has_structure = True
                    break
        
        if not has_structure:
            self.warnings.append("Summary lacks clear structure")
            score -= 15
        
        if self._has_duplicate_content():
            self.warnings.append("Summary contains duplicate content")
            score -= 10
        
        return max(0, min(100, score))

    def _evaluate_note_usefulness(self) -> float:
        score = 75.0
        
        summary_text = self._get_summary_text()
        
        has_key_points = bool(self.final_summary.get("key_points"))
        if has_key_points:
            score += 10
        
        has_action_items = bool(self.final_summary.get("action_items"))
        if has_action_items:
            score += 5
        
        has_quotes = len(self.quotes) > 0
        if has_quotes:
            score += 5
        
        has_terms = len(self.terms) > 0
        if has_terms:
            score += 5
        
        return max(0, min(100, score))

    def _evaluate_hallucination_risk(self) -> float:
        score = 100.0
        
        if not self.final_summary and not self.chunks:
            return 50.0
        
        summary_text = self._get_summary_text()
        
        potential_entities = self.validator.extract_entities(summary_text)
        suspicious_entities = []
        
        for entity in potential_entities:
            if len(entity) > 3 and not self.validator.check_entity_in_transcript(entity):
                suspicious_entities.append(entity)
        
        if suspicious_entities:
            risk_ratio = len(suspicious_entities) / max(1, len(potential_entities))
            score -= risk_ratio * 40
            self.warnings.append(
                f"Potential hallucinated entities found: {', '.join(suspicious_entities[:5])}"
            )
            self.errors.append(ValidationError(
                rule="entity_verification",
                message=f"Found {len(suspicious_entities)} entities not present in transcript",
                severity="warning"
            ))
        
        vague_patterns = [
            r'据说', r'据说', r'据说', r'可能', r'大概是',
            r'好像是', r'似乎', r'传闻', r'传言'
        ]
        for pattern in vague_patterns:
            if re.search(pattern, summary_text):
                score -= 10
                break
        
        return max(0, min(100, score))

    def _has_duplicate_content(self) -> bool:
        summary_text = self._get_summary_text()
        sentences = re.split(r'[。\n]', summary_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return False
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if self._similarity(sentences[i], sentences[j]) > 0.8:
                    return True
        return False

    def _similarity(self, s1: str, s2: str) -> float:
        s1_words = set(s1.split())
        s2_words = set(s2.split())
        if not s1_words or not s2_words:
            return 0.0
        intersection = len(s1_words & s2_words)
        union = len(s1_words | s2_words)
        return intersection / union if union > 0 else 0.0

    def _get_summary_text(self) -> str:
        parts = []
        if self.final_summary:
            for key in ["one_sentence_summary", "detailed_summary", "timeline_summary", 
                       "summary", "topics", "decisions", "notes"]:
                val = self.final_summary.get(key)
                if val:
                    if isinstance(val, list):
                        parts.extend([str(v) for v in val])
                    else:
                        parts.append(str(val))
        
        for chunk in self.chunks:
            summary = chunk.get("summary", "")
            if summary:
                parts.append(summary)
        
        return " ".join(parts)

    def _generate_suggestions(self) -> List[str]:
        suggestions = []
        
        if self.completeness_score < 80:
            suggestions.append("Add more detailed content to improve completeness")
        
        if self.faithfulness_score < 80:
            suggestions.append("Review and verify all quotes against original transcript")
        
        if self.timestamp_accuracy_score < 90:
            suggestions.append("Review and correct all timestamps to match transcript ranges")
        
        if self.structure_quality_score < 80:
            suggestions.append("Improve summary structure with clear chapter divisions")
        
        if self.note_usefulness_score < 80:
            suggestions.append("Add key points, action items, or relevant quotes")
        
        if self.hallucination_risk_score < 80:
            suggestions.append("Verify entity names and factual claims against transcript")
        
        if not suggestions:
            suggestions.append("Summary quality is good. Consider adding more quotes for better reference.")
        
        return suggestions


def evaluate_video(
    video_id: int,
    transcript: List[Dict],
    chunks: List[Dict],
    final_summary: Dict,
    quotes: List[Dict] = None,
    terms: List[Dict] = None,
    chapters: List[Dict] = None,
    format: str = "markdown"
) -> Tuple[str, EvaluationResult]:
    evaluator = Evaluator(transcript, chunks, final_summary, quotes, terms, chapters)
    result = evaluator.evaluate()
    
    if format == "json":
        output = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    else:
        output = generate_markdown_report(video_id, result)
    
    return output, result


def generate_markdown_report(video_id: int, result: EvaluationResult) -> str:
    score_label = _get_score_label(result.overall_score)
    
    lines = [
        f"# 摘要质量评估",
        "",
        f"## 总分",
        "",
        f"**{result.overall_score} / 100** ({score_label})",
        "",
        "## 各项评分",
        "",
        f"- 完整性：{result.completeness_score}/100",
        f"- 忠实度：{result.faithfulness_score}/100",
        f"- 时间戳准确性：{result.timestamp_accuracy_score}/100",
        f"- 结构质量：{result.structure_quality_score}/100",
        f"- 笔记实用性：{result.note_usefulness_score}/100",
        f"- 幻觉风险：{result.hallucination_risk_score}/100",
        "",
    ]
    
    if result.warnings:
        lines.extend([
            "## 发现的问题",
            "",
        ])
        for warning in result.warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    if result.issues:
        lines.extend([
            "## 严重问题",
            "",
        ])
        for issue in result.issues:
            lines.append(f"- ❌ {issue}")
        lines.append("")
    
    if result.suggestions:
        lines.extend([
            "## 修改建议",
            "",
        ])
        for suggestion in result.suggestions:
            lines.append(f"- 💡 {suggestion}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        f"> 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> Video ID: {video_id}",
        "",
        "> 本评估报告由规则校验和 LLM 分析生成，仅供参考。",
    ])
    
    return "\n".join(lines)


def _get_score_label(score: float) -> str:
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "一般"
    elif score >= 60:
        return "及格"
    else:
        return "需改进"
