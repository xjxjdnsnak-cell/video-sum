from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime

from ..config import settings
from ..utils.timefmt import format_timestamp
from ..summarizer.prompts import NoteStyle


def export_markdown(
    video_id: int,
    video_title: str,
    transcript: List[Dict],
    chunk_summaries: List[Dict],
    final_summary: Dict,
    chapters: List[Dict] = None,
    quotes: List[Dict] = None,
    note_style: NoteStyle = NoteStyle.DETAILED,
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> str:
    if chapters is None:
        chapters = []
    if quotes is None:
        quotes = []

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)[:50]
        output_path = output_dir / f"{safe_title}.md"
    else:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)[:50]
        output_path = settings.OUTPUT_DIR / f"{safe_title}.md"

    note_style_labels = {
        NoteStyle.BRIEF: "简短笔记",
        NoteStyle.DETAILED: "详细笔记",
        NoteStyle.STUDY: "学习笔记",
        NoteStyle.MEETING: "会议记录",
        NoteStyle.TUTORIAL: "教程笔记"
    }
    style_label = note_style_labels.get(note_style, "详细笔记")

    lines = [
        f"# {style_label}：{video_title}",
        "",
        f"> 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 模板类型: {style_label}",
        "",
        "---",
        "",
    ]

    if final_summary:
        lines.extend([
            "## 一句话总结",
            "",
            final_summary.get("one_sentence_summary", ""),
            "",
            "---",
            "",
        ])

    if note_style == NoteStyle.BRIEF:
        _add_brief_style(lines, final_summary, chunk_summaries, quotes)
    elif note_style == NoteStyle.STUDY:
        _add_study_style(lines, final_summary, chunk_summaries, chapters, quotes)
    elif note_style == NoteStyle.MEETING:
        _add_meeting_style(lines, final_summary, chunk_summaries, chapters)
    elif note_style == NoteStyle.TUTORIAL:
        _add_tutorial_style(lines, final_summary, chunk_summaries, chapters)
    else:
        _add_detailed_style(lines, final_summary, chunk_summaries, chapters, quotes)

    if transcript:
        lines.extend([
            "---",
            "",
            "## 完整转写",
            "",
        ])
        for seg in transcript:
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            lines.extend([
                f"### [{start_str} - {end_str}]",
                "",
                seg["text"],
                "",
            ])

    lines.extend([
        "",
        "---",
        "",
        f"> 📎 完整转写入口: Video ID {video_id}",
        "",
        "> ⚠️ 本笔记由 AI 自动生成，如有疑问请参考原始视频。",
    ])

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)


def _add_brief_style(lines: List, final_summary: Dict, chunk_summaries: List[Dict], quotes: List[Dict]):
    key_points = final_summary.get("key_points", [])
    if key_points:
        lines.extend([
            "## 重点列表",
            "",
        ])
        if isinstance(key_points, list):
            for point in key_points:
                lines.append(f"- {point}")
        else:
            for point in key_points.split("\n"):
                point = point.strip()
                if point:
                    if point.startswith("-") or point.startswith("*"):
                        lines.append(point)
                    else:
                        lines.append(f"- {point}")
        lines.extend(["", "---", ""])

    if quotes:
        lines.extend([
            "## 精选引用",
            "",
        ])
        for quote in quotes[:5]:
            lines.append(f"> {quote.get('text', '')}")
            lines.append(f"> — [{quote.get('start_time', '')} - {quote.get('end_time', '')}]")
            lines.append("")
        lines.extend(["---", ""])


def _add_study_style(lines: List, final_summary: Dict, chunk_summaries: List[Dict], 
                     chapters: List[Dict], quotes: List[Dict]):
    chapter_toc = final_summary.get("chapter_toc", [])
    if chapter_toc:
        lines.extend([
            "## 章节目录",
            "",
        ])
        if isinstance(chapter_toc, list):
            for i, chapter in enumerate(chapter_toc, 1):
                lines.append(f"{i}. {chapter}")
        lines.extend(["", "---", ""])

    key_knowledge = final_summary.get("key_knowledge", [])
    if key_knowledge:
        lines.extend([
            "## 关键知识点",
            "",
        ])
        if isinstance(key_knowledge, list):
            for knowledge in key_knowledge:
                lines.append(f"- {knowledge}")
        else:
            for k in key_knowledge.split("\n"):
                k = k.strip()
                if k:
                    lines.append(f"- {k}")
        lines.extend(["", "---", ""])

    terms = final_summary.get("terms", [])
    if terms:
        lines.extend([
            "## 术语解释",
            "",
        ])
        if isinstance(terms, list):
            for term in terms:
                if isinstance(term, dict):
                    lines.append(f"### {term.get('term', '')}")
                    lines.append(f"{term.get('explanation', '')}")
                else:
                    lines.append(f"- {term}")
                lines.append("")
        lines.extend(["", "---", ""])

    review_questions = final_summary.get("review_questions", [])
    if review_questions:
        lines.extend([
            "## 复习问题",
            "",
        ])
        if isinstance(review_questions, list):
            for i, q in enumerate(review_questions, 1):
                lines.append(f"{i}. {q}")
        else:
            for q in review_questions.split("\n"):
                q = q.strip().lstrip("0123456789.、 ")
                if q:
                    lines.append(f"- {q}")
        lines.extend(["", "---", ""])

    common_mistakes = final_summary.get("common_mistakes", [])
    if common_mistakes:
        lines.extend([
            "## 易错点",
            "",
        ])
        if isinstance(common_mistakes, list):
            for mistake in common_mistakes:
                lines.append(f"- {mistake}")
        else:
            for m in common_mistakes.split("\n"):
                m = m.strip()
                if m:
                    lines.append(f"- {m}")
        lines.extend(["", "---", ""])

    if quotes:
        lines.extend([
            "## 精选引用",
            "",
        ])
        for quote in quotes[:5]:
            lines.append(f"> {quote.get('text', '')}")
            lines.append(f"> — [{quote.get('start_time', '')} - {quote.get('end_time', '')}]")
            lines.append("")
        lines.extend(["---", ""])


def _add_meeting_style(lines: List, final_summary: Dict, chunk_summaries: List[Dict], chapters: List[Dict]):
    topics = final_summary.get("topics", [])
    if topics:
        lines.extend([
            "## 议题",
            "",
        ])
        if isinstance(topics, list):
            for topic in topics:
                lines.append(f"- {topic}")
        else:
            for t in topics.split("\n"):
                t = t.strip()
                if t:
                    lines.append(f"- {t}")
        lines.extend(["", "---", ""])

    decisions = final_summary.get("decisions", [])
    if decisions:
        lines.extend([
            "## 决策",
            "",
        ])
        if isinstance(decisions, list):
            for decision in decisions:
                lines.append(f"- {decision}")
        else:
            for d in decisions.split("\n"):
                d = d.strip()
                if d:
                    lines.append(f"- {d}")
        lines.extend(["", "---", ""])

    action_items = final_summary.get("action_items", [])
    if action_items:
        lines.extend([
            "## 待办事项",
            "",
        ])
        if isinstance(action_items, list):
            for item in action_items:
                if isinstance(item, dict):
                    lines.append(f"- [ ] {item.get('task', '')} - {item.get('owner', '未指定')}")
                else:
                    lines.append(f"- [ ] {item}")
        lines.extend(["", "---", ""])

    timeline = final_summary.get("timeline_summary", "")
    if timeline:
        lines.extend([
            "## 时间线",
            "",
            timeline,
            "",
            "---",
            "",
        ])


def _add_tutorial_style(lines: List, final_summary: Dict, chunk_summaries: List[Dict], chapters: List[Dict]):
    chapter_toc = final_summary.get("chapter_toc", [])
    if chapter_toc:
        lines.extend([
            "## 步骤目录",
            "",
        ])
        if isinstance(chapter_toc, list):
            for i, chapter in enumerate(chapter_toc, 1):
                lines.append(f"{i}. {chapter}")
        lines.extend(["", "---", ""])

    prerequisites = final_summary.get("prerequisites", [])
    if prerequisites:
        lines.extend([
            "## 前置要求",
            "",
        ])
        if isinstance(prerequisites, list):
            for prereq in prerequisites:
                lines.append(f"- {prereq}")
        else:
            for p in prerequisites.split("\n"):
                p = p.strip()
                if p:
                    lines.append(f"- {p}")
        lines.extend(["", "---", ""])

    steps = final_summary.get("steps", [])
    if steps:
        lines.extend([
            "## 操作步骤",
            "",
        ])
        if isinstance(steps, list):
            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    lines.append(f"### 步骤 {i}: {step.get('step', '')}")
                    lines.append("")
                    lines.append(step.get('description', ''))
                    lines.append("")
                    commands = step.get('commands', [])
                    if commands:
                        lines.append("```bash")
                        for cmd in commands:
                            lines.append(cmd)
                        lines.append("```")
                        lines.append("")
                else:
                    lines.append(f"### 步骤 {i}")
                    lines.append(step)
                    lines.append("")
        lines.extend(["---", ""])

    notes = final_summary.get("notes", [])
    if notes:
        lines.extend([
            "## 注意事项",
            "",
        ])
        if isinstance(notes, list):
            for note in notes:
                lines.append(f"- {note}")
        else:
            for n in notes.split("\n"):
                n = n.strip()
                if n:
                    lines.append(f"- {n}")
        lines.extend(["", "---", ""])


def _add_detailed_style(lines: List, final_summary: Dict, chunk_summaries: List[Dict],
                        chapters: List[Dict], quotes: List[Dict]):
    chapter_toc = final_summary.get("chapter_toc", [])
    if chapter_toc:
        lines.extend([
            "## 章节目录",
            "",
        ])
        if isinstance(chapter_toc, list):
            for i, chapter in enumerate(chapter_toc, 1):
                lines.append(f"{i}. {chapter}")
        lines.extend(["", "---", ""])

    detailed = final_summary.get("detailed_summary", final_summary.get("timeline_summary", ""))
    if detailed:
        lines.extend([
            "## 详细总结",
            "",
            detailed,
            "",
            "---",
            "",
        ])

    if chapters:
        lines.extend([
            "## 时间轴章节",
            "",
        ])
        for chapter in chapters:
            lines.append(f"### {chapter.get('title', '章节')}")
            lines.append(f"[{chapter.get('start_time', '')} - {chapter.get('end_time', '')}]")
            lines.append("")
            lines.append(chapter.get('summary', ''))
            lines.append("")
        lines.extend(["---", ""])
    elif chunk_summaries:
        lines.extend([
            "## 时间轴摘要",
            "",
        ])
        for chunk in chunk_summaries:
            topic = chunk.get("topic", "")
            if topic:
                lines.append(f"### {topic}")
            else:
                lines.append(f"### {chunk['start_time']} - {chunk['end_time']}")
            lines.append("")
            lines.append(chunk.get("summary", ""))
            lines.append("")

            key_points = chunk.get("key_points", [])
            if key_points:
                lines.append("**关键观点:**")
                for point in key_points:
                    lines.append(f"- {point}")
                lines.append("")
            lines.append("")

    key_points = final_summary.get("key_points", [])
    if key_points:
        lines.extend([
            "## 核心观点",
            "",
        ])
        if isinstance(key_points, list):
            for point in key_points:
                lines.append(f"- {point}")
        else:
            for point in key_points.split("\n"):
                point = point.strip()
                if point:
                    if point.startswith("-") or point.startswith("*"):
                        lines.append(point)
                    else:
                        lines.append(f"- {point}")
        lines.extend(["", "---", ""])

    key_knowledge = final_summary.get("key_knowledge", [])
    if key_knowledge:
        lines.extend([
            "## 关键知识点",
            "",
        ])
        if isinstance(key_knowledge, list):
            for knowledge in key_knowledge:
                lines.append(f"- {knowledge}")
        else:
            for k in key_knowledge.split("\n"):
                k = k.strip()
                if k:
                    lines.append(f"- {k}")
        lines.extend(["", "---", ""])

    action_items = final_summary.get("action_items", [])
    if action_items:
        lines.extend([
            "## 行动项",
            "",
        ])
        if isinstance(action_items, list):
            for item in action_items:
                lines.append(f"- {item}")
        else:
            for item in action_items.split("\n"):
                item = item.strip()
                if item:
                    lines.append(f"- {item}")
        lines.extend(["", "---", ""])

    if quotes:
        lines.extend([
            "## 精选引用",
            "",
        ])
        for quote in quotes[:5]:
            lines.append(f"> {quote.get('text', '')}")
            lines.append(f"> — [{quote.get('start_time', '')} - {quote.get('end_time', '')}]")
            lines.append("")
        lines.extend(["---", ""])
