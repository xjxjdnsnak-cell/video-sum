from enum import Enum


class NoteStyle(str, Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    STUDY = "study"
    MEETING = "meeting"
    TUTORIAL = "tutorial"


CHUNK_SUMMARY_PROMPT = """请为以下视频片段生成详细的中文摘要。

时间段: {start_time} - {end_time}
视频内容:
{text}

请按照以下JSON格式输出摘要（只输出JSON，不要其他内容）：
{{
    "topic": "本段主题（一句话概括）",
    "key_points": ["关键观点1", "关键观点2", "关键观点3"],
    "important_terms": ["术语1: 解释", "术语2: 解释"],
    "quote": "从原文中最具代表性的一句话（必须完全来自原文，不允许编造）",
    "chapter_hint": "该段内容适合归入哪个章节",
    "summary": "2-3句话的详细摘要"
}}
"""

FINAL_SUMMARY_PROMPT_BRIEF = """你是一个专业的视频内容摘要助手。请根据以下视频各时间段的摘要，生成简洁的总结。

视频标题: {video_title}
各时间段摘要:
{chunk_summaries}

请按照以下JSON格式输出最终总结（只输出JSON，不要其他内容）：
{{
    "one_sentence_summary": "一句话总结（不超过50字）",
    "key_points": ["重点列表项1", "重点列表项2", "重点列表项3"]
}}
"""

FINAL_SUMMARY_PROMPT_DETAILED = """你是一个专业的视频内容摘要助手。请根据以下视频各时间段的摘要，生成详细的总结。

视频标题: {video_title}
各时间段摘要:
{chunk_summaries}

请按照以下JSON格式输出最终总结（只输出JSON，不要其他内容）：
{{
    "one_sentence_summary": "一句话总结（不超过50字）",
    "chapter_toc": ["章节1", "章节2", "章节3"],
    "timeline_summary": "时间轴摘要，描述各时间段的要点",
    "key_points": ["核心观点1", "核心观点2", "核心观点3"],
    "key_knowledge": ["关键知识点1", "关键知识点2"],
    "action_items": ["可执行的行动项1", "可执行的行动项2"]
}}
"""

FINAL_SUMMARY_PROMPT_STUDY = """你是一个专业的教育视频摘要助手。请根据以下视频各时间段的摘要，生成适合学习总结。

视频标题: {video_title}
各时间段摘要:
{chunk_summaries}

请按照以下JSON格式输出最终总结（只输出JSON，不要其他内容）：
{{
    "one_sentence_summary": "一句话总结（不超过50字）",
    "chapter_toc": ["章节1", "章节2", "章节3"],
    "key_knowledge": ["关键知识点1", "关键知识点2"],
    "terms": [
        {{"term": "术语1", "explanation": "术语1的解释"}},
        {{"term": "术语2", "explanation": "术语2的解释"}}
    ],
    "review_questions": ["复习问题1", "复习问题2", "复习问题3"],
    "common_mistakes": ["学习时容易犯的错误1", "学习时容易犯的错误2"],
    "action_items": ["可执行的复习行动1", "可执行的复习行动2"]
}}
"""

FINAL_SUMMARY_PROMPT_MEETING = """你是一个专业的会议记录助手。请根据以下视频各时间段的摘要，生成会议记录风格的总结。

视频标题: {video_title}
各时间段摘要:
{chunk_summaries}

请按照以下JSON格式输出最终总结（只输出JSON，不要其他内容）：
{{
    "one_sentence_summary": "一句话总结（不超过50字）",
    "topics": ["议题1", "议题2", "议题3"],
    "decisions": ["做出的决定1", "做出的决定2"],
    "action_items": [
        {{"task": "待办事项1", "owner": "责任人1"}},
        {{"task": "待办事项2", "owner": "责任人2"}}
    ],
    "timeline_summary": "时间线上的主要讨论内容"
}}
"""

FINAL_SUMMARY_PROMPT_TUTORIAL = """你是一个专业的教程视频摘要助手。请根据以下视频各时间段的摘要，生成教程风格的总结。

视频标题: {video_title}
各时间段摘要:
{chunk_summaries}

请按照以下JSON格式输出最终总结（只输出JSON，不要其他内容）：
{{
    "one_sentence_summary": "教程的一句话总结（不超过50字）",
    "chapter_toc": ["步骤章节1", "步骤章节2", "步骤章节3"],
    "steps": [
        {{"step": "步骤1名称", "description": "步骤1的详细描述", "commands": ["相关命令或代码1", "相关命令或代码2"]}},
        {{"step": "步骤2名称", "description": "步骤2的详细描述", "commands": ["相关命令或代码1"]}}
    ],
    "key_points": ["关键要点1", "关键要点2"],
    "notes": ["注意事项1", "注意事项2", "注意事项3"],
    "prerequisites": ["前置要求1", "前置要求2"]
}}
"""

FINAL_SUMMARY_PROMPTS = {
    NoteStyle.BRIEF: FINAL_SUMMARY_PROMPT_BRIEF,
    NoteStyle.DETAILED: FINAL_SUMMARY_PROMPT_DETAILED,
    NoteStyle.STUDY: FINAL_SUMMARY_PROMPT_STUDY,
    NoteStyle.MEETING: FINAL_SUMMARY_PROMPT_MEETING,
    NoteStyle.TUTORIAL: FINAL_SUMMARY_PROMPT_TUTORIAL,
}

VIDEO_ANALYSIS_PROMPT = """你是一个专业的视频内容分析师。请分析以下视频内容并回答问题。

视频标题: {title}
视频内容:
{content}

请回答以下问题：
1. 视频的主要观点是什么？
2. 有哪些重要的细节需要注意？
3. 如何将视频内容应用到实践中？

回答要求：
- 使用中文
- 简洁有条理
- 每个问题回答100-200字
"""

QUOTE_EXTRACTION_PROMPT = """你是一个专业的视频内容分析师。请从以下转写文本中提取最多5条最具代表性的原文引用。

转写内容:
{transcript}

要求：
1. 每条引用必须完全来自原文，不允许编造或修改
2. 每条引用必须带有时间戳
3. 选择最能代表视频核心内容的引用
4. 引用长度适中（10-50字）

请按照以下JSON格式输出（只输出JSON）：
{{
    "quotes": [
        {{"text": "引用原文1", "start_time": "00:00", "end_time": "00:10"}},
        {{"text": "引用原文2", "start_time": "00:30", "end_time": "00:40"}}
    ]
}}
"""

CHAPTER_AGGREGATION_PROMPT = """你是一个专业的视频内容分析师。请分析以下视频分段摘要，将相似主题的段落聚合为章节。

各分段摘要:
{chunk_summaries}

要求：
1. 根据主题相似度合并相邻的分段
2. 每个章节必须包含 start_time 和 end_time（取该章节内所有分段的时间范围）
3. 每个章节需要给出章节名称
4. 章节数量适中（通常3-8个）

请按照以下JSON格式输出（只输出JSON）：
{{
    "chapters": [
        {{
            "title": "章节1名称",
            "start_time": "00:00",
            "end_time": "05:30",
            "chunks": [0, 1, 2],
            "summary": "该章节的简要总结"
        }},
        {{
            "title": "章节2名称",
            "start_time": "05:30",
            "end_time": "10:00",
            "chunks": [3, 4],
            "summary": "该章节的简要总结"
        }}
    ]
}}
"""

TERM_EXTRACTION_PROMPT = """你是一个专业的术语提取助手。请从以下视频内容中提取专业术语及其解释。

视频标题: {video_title}

转写内容:
{transcript}

章节摘要:
{chapter_summaries}

要求：
1. 只提取真正在视频中出现过的术语
2. 术语解释必须基于视频内容，不允许编造
3. 标注每个术语首次出现的时间

请按照以下JSON格式输出（只输出JSON）：
{{
    "terms": [
        {{"term": "术语1", "explanation": "术语1的解释", "first_seen_time": "00:00"}},
        {{"term": "术语2", "explanation": "术语2的解释", "first_seen_time": "02:30"}}
    ]
}}
"""
