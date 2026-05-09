import srt
from pathlib import Path
from typing import List, Optional
from datetime import timedelta


def export_srt(
    segments: List[dict],
    output_path: str
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subtitles = []
    for i, seg in enumerate(segments, start=1):
        start_td = timedelta(seconds=seg["start"])
        end_td = timedelta(seconds=seg["end"])
        subtitles.append(srt.Subtitle(
            index=i,
            start=start_td,
            end=end_td,
            content=seg["text"]
        ))

    content = srt.compose(subtitles)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(output_path)
