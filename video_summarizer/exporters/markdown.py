from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime

from ..config import settings
from ..utils.timefmt import format_timestamp


def export_markdown(
    video_id: int,
    video_title: str,
    transcript: List[Dict],
    chunk_summaries: List[Dict],
    final_summary: Dict,
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> str:
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

    lines = [
        f"# 视频总结：{video_title}",
        "",
        f"> 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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

        detailed = final_summary.get("detailed_summary", "")
        if detailed:
            lines.extend([
                "## 详细总结",
                "",
                detailed,
                "",
                "---",
                "",
            ])

    if chunk_summaries:
        lines.extend([
            "## 时间轴摘要",
            "",
        ])
        for chunk in chunk_summaries:
            lines.extend([
                f"### {chunk['start_time']} - {chunk['end_time']}",
                "",
                chunk.get("summary", ""),
                "",
            ])

    if final_summary:
        key_points = final_summary.get("key_points", "")
        if key_points:
            lines.extend([
                "## 关键知识点",
                "",
            ])
            for point in key_points.split("\n"):
                point = point.strip()
                if point:
                    if point.startswith("-") or point.startswith("*"):
                        lines.append(point)
                    else:
                        lines.append(f"- {point}")
            lines.extend(["", "---", ""])

        questions = final_summary.get("questions", "")
        if questions:
            lines.extend([
                "## 可复习问题",
                "",
            ])
            for i, q in enumerate(questions.split("\n"), 1):
                q = q.strip().lstrip("0123456789.、 ")
                if q:
                    lines.append(f"{i}. {q}")
            lines.extend(["", "---", ""])

    if transcript:
        lines.extend([
            "## 完整转写",
            "",
        ])
        for seg in transcript:
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            lines.extend([
                f"### {start_str} - {end_str}",
                "",
                seg["text"],
                "",
            ])

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)
