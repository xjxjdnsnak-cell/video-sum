import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer

from video_summarizer.config import settings
from video_summarizer.db import get_all_videos, get_video_info
from video_summarizer.summarizer.pipeline import (
    get_transcript, get_summary_chunks, get_final_summary,
    get_chapters, get_quotes, get_terms
)
from video_summarizer.evaluator.evaluate import evaluate_video, generate_markdown_report
from video_summarizer.media.ffmpeg import check_ffmpeg_installed
from video_summarizer.media.downloader import check_ytdlp_installed, is_bilibili_url
from video_summarizer.exporters.markdown import export_markdown
from video_summarizer.exporters.json_exporter import export_json
from video_summarizer.exporters.srt import export_srt
from video_summarizer.utils.filename import sanitize_filename
from video_summarizer.summarizer.prompts import NoteStyle
from video_summarizer.search.evidence_retriever import (
    check_fts5_support, init_fts_tables, rebuild_all_indexes,
    search_fts, search_like, get_evidence,
    get_transcript_for_qa, get_summary_for_qa, get_final_summary_for_qa
)
from video_summarizer.search.qa_prompt import generate_qa_prompt, parse_qa_response


st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_environment():
    errors = []
    
    if not check_ffmpeg_installed():
        errors.append("❌ FFmpeg 未安装。请先安装 FFmpeg。")
    
    if not check_ytdlp_installed():
        errors.append("❌ yt-dlp 未安装。请运行: pip install yt-dlp")
    
    try:
        import faster_whisper
    except ImportError:
        errors.append("❌ faster-whisper 未安装。请运行: pip install faster-whisper")
    
    return errors


def get_video_list():
    return get_all_videos()


def get_video_details(video_id):
    info = get_video_info(video_id)
    if not info:
        return None
    
    transcript = get_transcript(video_id)
    chunks = get_summary_chunks(video_id)
    final_summary = get_final_summary(video_id)
    chapters = get_chapters(video_id)
    quotes = get_quotes(video_id)
    terms = get_terms(video_id)
    
    return {
        "info": info,
        "transcript": transcript,
        "chunks": chunks,
        "final_summary": final_summary,
        "chapters": chapters,
        "quotes": quotes,
        "terms": terms
    }


def evaluate_summary(video_id, transcript, chunks, final_summary, quotes, terms, chapters):
    output, result = evaluate_video(
        video_id=video_id,
        transcript=transcript,
        chunks=chunks,
        final_summary=final_summary,
        quotes=quotes,
        terms=terms,
        chapters=chapters,
        format="markdown"
    )
    return result, output


def render_home_page():
    st.title("🎬 Video Summarizer")
    st.markdown("### B站/本地视频总结器")
    
    errors = check_environment()
    if errors:
        st.error("### 环境检查失败")
        for error in errors:
            st.error(error)
        st.stop()
    
    st.success("✅ 环境检查通过")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 新建任务", "📋 历史记录", "🔍 搜索内容", "💬 视频问答"])
    
    with tab1:
        render_new_task_form()
    
    with tab2:
        render_history()
    
    with tab3:
        render_search_page()
    
    with tab4:
        render_qa_page()


def render_new_task_form():
    st.subheader("创建新任务")
    
    input_type = st.radio("选择输入类型", ["📁 本地视频", "🔗 B站链接"], horizontal=True)
    
    video_file = None
    video_url = None
    
    if input_type == "📁 本地视频":
        video_file = st.file_uploader(
            "上传视频文件",
            type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
            help="支持 MP4, AVI, MOV, MKV, FLV, WMV 格式"
        )
        if video_file:
            st.info(f"已选择: {video_file.name} ({video_file.size / 1024 / 1024:.2f} MB)")
    else:
        video_url = st.text_input(
            "B站视频链接或BV号",
            placeholder="例如: https://www.bilibili.com/video/BV1xx411c7mZ 或 BV1xx411c7mZ",
            help="支持 BV号、av号、完整链接和短链接"
        )
        cookies_file = st.text_input(
            "Cookie文件路径 (可选)",
            placeholder="/path/to/cookies.txt",
            help="需要登录才能访问的视频，请提供Cookie文件"
        )
        if cookies_file:
            st.info(f"使用 Cookie: {cookies_file}")
    
    st.markdown("#### 处理参数")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asr_provider = st.selectbox(
            "ASR Provider",
            ["mock", "faster-whisper"],
            index=0,
            help="faster-whisper 需要安装模型，mock 用于测试"
        )
        
        whisper_model = st.selectbox(
            "Whisper 模型",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="越大的模型越准确，但需要更多显存"
        )
        
        device = st.selectbox(
            "设备",
            ["cpu", "cuda", "auto"],
            index=0,
            help="cuda 需要 NVIDIA 显卡"
        )
    
    with col2:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["mock", "openai-compatible", "ollama"],
            index=0,
            help="mock 用于测试，openai-compatible 和 ollama 需要配置"
        )
        
        language = st.selectbox(
            "语言",
            ["auto", "zh", "en"],
            index=0,
            help="auto 自动检测，zh 中文，en 英文"
        )
        
        note_style = st.selectbox(
            "笔记模板",
            ["brief", "detailed", "study", "meeting", "tutorial"],
            index=1,
            format_func=lambda x: {
                "brief": "📝 简短笔记",
                "detailed": "📄 详细笔记",
                "study": "📚 学习笔记",
                "meeting": "📅 会议记录",
                "tutorial": "🎓 教程笔记"
            }[x]
        )
    
    with col3:
        chunk_min = st.slider("最小分段 (分钟)", 1, 10, 3)
        chunk_max = st.slider("最大分段 (分钟)", 2, 15, 5)

        keep_audio = st.checkbox("保留音频文件")
        force_rerun = st.checkbox(
            "强制重跑（忽略已有结果）",
            help="默认断点续跑：同一视频已完成的转写/摘要阶段会直接复用"
        )
    
    st.markdown("---")
    
    if st.button("🚀 开始处理", type="primary", use_container_width=True):
        if input_type == "📁 本地视频" and not video_file:
            st.error("请先上传视频文件")
            return
        if input_type == "🔗 B站链接" and not video_url:
            st.error("请先输入B站链接")
            return
        # S-4: the Web UI must not hand arbitrary domains (and any cookies the
        # user configured) to yt-dlp. Non-bilibili URLs are refused outright.
        if input_type == "🔗 B站链接" and not is_bilibili_url(video_url):
            st.error(
                "仅支持B站内容：bilibili.com 链接、b23.tv 短链接或 BV/av 号。"
                "Web UI 不允许处理其他域名的 URL（CLI 可用 --allow-any-url 显式放行）。"
            )
            return
        
        with st.spinner("正在处理..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if input_type == "📁 本地视频":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as f:
                        f.write(video_file.getvalue())
                        temp_path = f.name
                    
                    try:
                        progress_bar.progress(20)
                        status_text.text("正在处理本地视频...")
                        
                        video_id = process_local_video(
                            temp_path,
                            asr_provider=asr_provider,
                            llm_provider=llm_provider,
                            whisper_model=whisper_model,
                            device=device,
                            language=language,
                            note_style=note_style,
                            chunk_min=chunk_min,
                            chunk_max=chunk_max,
                            keep_audio=keep_audio,
                            force_rerun=force_rerun
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("处理完成！")
                        
                        time.sleep(1)
                        st.success(f"✅ 处理完成！Video ID: {video_id}")
                        
                        st.session_state.selected_video_id = video_id
                        st.rerun()
                    
                    finally:
                        if not keep_audio and os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                else:
                    progress_bar.progress(20)
                    status_text.text("正在处理B站链接...")
                    
                    video_id = process_bilibili_url(
                        video_url,
                        cookies_file=cookies_file if cookies_file else None,
                        asr_provider=asr_provider,
                        llm_provider=llm_provider,
                        whisper_model=whisper_model,
                        device=device,
                        language=language,
                        note_style=note_style,
                        chunk_min=chunk_min,
                        chunk_max=chunk_max,
                        force_rerun=force_rerun
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("处理完成！")
                    
                    time.sleep(1)
                    st.success(f"✅ 处理完成！Video ID: {video_id}")
                    
                    st.session_state.selected_video_id = video_id
                    st.rerun()
            
            except Exception as e:
                progress_bar.progress(0)
                st.error(f"处理失败: {str(e)}")


def _run_shared_pipeline(
    video_path_or_url: str,
    is_url: bool,
    *,
    asr_provider: str,
    llm_provider: str,
    whisper_model: str,
    device: str,
    language: str,
    note_style: str,
    chunk_min: int,
    chunk_max: int,
    keep_audio: bool,
    cookies_file: str | None = None,
    force_rerun: bool = False,
) -> int:
    """Delegate to the CLI's shared orchestration (audit A-5).

    The web UI used to carry a second, drifting copy of the pipeline. There
    is now a single implementation; the web layer only collects parameters
    and translates the result back into a video id. Resume semantics come
    for free (same source -> same video record -> completed stages skipped).
    """
    import typer

    from video_summarizer.cli import _run_summarize
    from video_summarizer.db import find_video_by_source, get_video_info, get_db
    from video_summarizer.summarizer.prompts import NoteStyle as NS

    _run_summarize(
        video_path_or_url=video_path_or_url,
        is_url=is_url,
        llm_provider=llm_provider,
        asr_provider=asr_provider,
        chunk_min=chunk_min,
        chunk_max=chunk_max,
        output=None,
        model=whisper_model,
        device=device,
        language=language,
        model_dir=None,
        keep_audio=keep_audio,
        force=force_rerun,
        resume=not force_rerun,
        cookies=cookies_file if is_url else None,
        note_style=NS[note_style.upper()],
    )

    with get_db() as conn:
        row = find_video_by_source(conn, video_path_or_url, str(Path(video_path_or_url).resolve()))
    if not row:
        raise Exception("处理完成，但未找到视频记录，请查看历史页面。")
    return row["id"]


def process_local_video(
    video_path: str,
    asr_provider: str,
    llm_provider: str,
    whisper_model: str,
    device: str,
    language: str,
    note_style: str,
    chunk_min: int,
    chunk_max: int,
    keep_audio: bool,
    force_rerun: bool = False
) -> int:
    return _run_shared_pipeline(
        str(video_path),
        False,
        asr_provider=asr_provider,
        llm_provider=llm_provider,
        whisper_model=whisper_model,
        device=device,
        language=language,
        note_style=note_style,
        chunk_min=chunk_min,
        chunk_max=chunk_max,
        keep_audio=keep_audio,
        force_rerun=force_rerun,
    )


def process_bilibili_url(
    url: str,
    cookies_file: str = None,
    asr_provider: str = "mock",
    llm_provider: str = "mock",
    whisper_model: str = "base",
    device: str = "cpu",
    language: str = "auto",
    note_style: str = "detailed",
    chunk_min: int = 3,
    chunk_max: int = 5,
    force_rerun: bool = False
) -> int:
    try:
        return _run_shared_pipeline(
            url,
            True,
            asr_provider=asr_provider,
            llm_provider=llm_provider,
            whisper_model=whisper_model,
            device=device,
            language=language,
            note_style=note_style,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            keep_audio=False,
            cookies_file=cookies_file,
            force_rerun=force_rerun,
        )
    except typer.Exit:
        # The shared runner already printed the reason and recorded
        # last_error on the video record; surface it in the UI.
        from video_summarizer.db import find_video_by_source, get_db

        with get_db() as conn:
            row = find_video_by_source(conn, url, normalize_bilibili_url(url))
        message = (row["last_error"] if row else None) or "处理失败，详情见服务端日志。"
        raise Exception(f"处理失败: {message}")


def render_history():
    st.subheader("历史记录")
    
    videos = get_video_list()
    
    if not videos:
        st.info("暂无处理记录")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("搜索", placeholder="输入标题或ID搜索...")
    with col2:
        status_filter = st.selectbox(
            "状态筛选",
            ["全部", "completed", "failed", "processing"],
            index=0
        )
    
    filtered_videos = videos
    if search:
        filtered_videos = [v for v in filtered_videos if 
                         search.lower() in str(v.get('id', '')).lower() or 
                         search.lower() in v.get('title', '').lower()]
    if status_filter != "全部":
        filtered_videos = [v for v in filtered_videos if v.get('status') == status_filter]
    
    st.markdown(f"共 {len(filtered_videos)} 条记录")
    
    for video in filtered_videos[:20]:
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**ID: {video['id']}**")
                status_color = {
                    "completed": "🟢",
                    "failed": "🔴",
                    "processing": "🟡"
                }.get(video.get('status', ''), "⚪")
                st.markdown(f"{status_color} {video.get('status', 'unknown')}")
            
            with col2:
                st.markdown(f"**{video.get('title', 'Untitled')}**")
                st.caption(f"来源: {video.get('source_type', 'unknown')}")
                st.caption(f"创建: {video.get('created_at', '')[:10]}")
            
            with col3:
                if st.button("查看", key=f"view_{video['id']}"):
                    st.session_state.selected_video_id = video['id']
                    st.rerun()
            
            st.markdown("---")


def render_video_detail(video_id):
    details = get_video_details(video_id)
    
    if not details:
        st.error(f"未找到 Video ID: {video_id}")
        if st.button("返回历史记录"):
            st.session_state.pop('selected_video_id', None)
            st.rerun()
        return
    
    info = details["info"]
    transcript = details["transcript"]
    chunks = details["chunks"]
    final_summary = details["final_summary"]
    chapters = details["chapters"]
    quotes = details["quotes"]
    terms = details["terms"]
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title(f"📄 {info.get('title', 'Untitled')}")
    with col2:
        if st.button("← 返回历史记录"):
            st.session_state.pop('selected_video_id', None)
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 摘要", "📋 详情", "💾 导出", "⭐ 评估"])
    
    with tab1:
        if final_summary:
            st.markdown("### 一句话总结")
            st.info(final_summary.get("one_sentence_summary", ""))
            
            if chunks:
                st.markdown("### 时间轴摘要")
                for chunk in chunks:
                    with st.expander(f"[{chunk.get('start_time', '00:00')} - {chunk.get('end_time', '00:00')}] {chunk.get('topic', '片段')}"):
                        st.markdown(chunk.get("summary", ""))
                        
                        key_points = chunk.get("key_points", [])
                        if key_points:
                            st.markdown("**关键观点:**")
                            for point in key_points:
                                st.markdown(f"- {point}")
            
            if chapters:
                st.markdown("### 章节结构")
                for i, chapter in enumerate(chapters, 1):
                    st.markdown(f"**{i}. {chapter.get('title', '章节')}** [{chapter.get('start_time', '')} - {chapter.get('end_time', '')}]")
                    st.markdown(chapter.get("summary", ""))
            
            if quotes:
                st.markdown("### 精选引用")
                for quote in quotes:
                    st.markdown(f"> {quote.get('text', '')}")
                    st.caption(f"— [{quote.get('start_time', '')} - {quote.get('end_time', '')}]")
        else:
            st.warning("暂无摘要")
    
    with tab2:
        st.markdown("### 基本信息")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Video ID", info.get('id'))
        with col2:
            st.metric("来源", info.get('source_type'))
        with col3:
            st.metric("状态", info.get('status'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**创建时间:** {info.get('created_at', '')}")
            st.markdown(f"**当前阶段:** {info.get('current_stage', '')}")
        with col2:
            st.markdown(f"**转写片段:** {info.get('transcript_count', 0)}")
            st.markdown(f"**摘要章节:** {info.get('chunk_count', 0)}")
        
        if info.get('last_error'):
            st.error(f"**错误:** {info.get('last_error')}")
        
        if transcript:
            st.markdown("### 转写预览")
            st.text_area(
                "转写内容",
                "\n\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}" for s in transcript[:20]]),
                height=300,
                disabled=True
            )
            if len(transcript) > 20:
                st.info(f"还有 {len(transcript) - 20} 段转写...")
    
    with tab3:
        st.markdown("### 导出选项")
        
        export_format = st.selectbox("选择格式", ["Markdown", "JSON", "SRT"])
        
        if st.button("📥 生成导出文件"):
            with st.spinner("正在导出..."):
                try:
                    if export_format == "Markdown":
                        output_path = export_markdown(
                            video_id=video_id,
                            video_title=info.get('title', 'Untitled'),
                            transcript=transcript,
                            chunk_summaries=chunks,
                            final_summary=final_summary,
                            chapters=chapters,
                            quotes=quotes
                        )
                    elif export_format == "JSON":
                        output_path = export_json(
                            video_id=video_id,
                            video_title=info.get('title', 'Untitled'),
                            video_url=info.get('url'),
                            video_author=info.get('author'),
                            duration=info.get('duration'),
                            transcript=transcript,
                            chunk_summaries=chunks,
                            final_summary=final_summary,
                            chapters=chapters,
                            quotes=quotes,
                            terms=terms
                        )
                    else:
                        safe_title = sanitize_filename(info.get('title', 'video'))
                        output_path = export_srt(transcript, str(settings.OUTPUT_DIR / f"{safe_title}.srt"))
                    
                    st.success(f"已导出到: {output_path}")
                    
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    st.download_button(
                        f"下载 {export_format}",
                        content,
                        file_name=Path(output_path).name,
                        mime="text/plain" if export_format != "JSON" else "application/json"
                    )
                
                except Exception as e:
                    st.error(f"导出失败: {str(e)}")
    
    with tab4:
        st.markdown("### 质量评估")
        
        if st.button("🔍 运行评估"):
            with st.spinner("正在评估..."):
                try:
                    result, report = evaluate_summary(
                        video_id,
                        transcript,
                        chunks,
                        final_summary,
                        quotes,
                        terms,
                        chapters
                    )
                    
                    st.markdown("### 评估结果")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总分", f"{result.overall_score}/100")
                    with col2:
                        score_emoji = "🌟" if result.overall_score >= 90 else "✅" if result.overall_score >= 80 else "⚠️" if result.overall_score >= 60 else "❌"
                        st.markdown(f"**等级:** {score_emoji}")
                    
                    st.markdown("#### 各项评分")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("完整性", f"{result.completeness_score}/100")
                        st.metric("忠实度", f"{result.faithfulness_score}/100")
                    with col2:
                        st.metric("时间戳准确", f"{result.timestamp_accuracy_score}/100")
                        st.metric("结构质量", f"{result.structure_quality_score}/100")
                    with col3:
                        st.metric("笔记实用", f"{result.note_usefulness_score}/100")
                        st.metric("幻觉风险", f"{result.hallucination_risk_score}/100")
                    
                    if result.warnings:
                        st.markdown("#### 发现的问题")
                        for warning in result.warnings:
                            st.warning(warning)
                    
                    if result.suggestions:
                        st.markdown("#### 修改建议")
                        for suggestion in result.suggestions:
                            st.info(f"💡 {suggestion}")
                    
                    st.download_button(
                        "📥 下载评估报告",
                        report,
                        file_name=f"evaluation_{video_id}.md",
                        mime="text/markdown"
                    )
                
                except Exception as e:
                    st.error(f"评估失败: {str(e)}")


def render_search_page():
    st.subheader("🔍 搜索视频内容")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input("搜索关键词", placeholder="输入要搜索的关键词...")
    with col2:
        video_id = st.number_input("限定视频ID（可选）", min_value=1, step=1, value=0)
        if video_id == 0:
            video_id = None
    
    limit = st.slider("返回结果数量", 5, 50, 20)
    
    if st.button("🔍 搜索", type="primary"):
        if not keyword:
            st.error("请输入搜索关键词")
            return
        
        with st.spinner("正在搜索..."):
            try:
                if not check_fts5_support():
                    st.warning("FTS5 不支持，将使用 LIKE 搜索")
                    results = search_like(keyword, video_id, limit)
                else:
                    try:
                        init_fts_tables()
                        results = search_fts(keyword, video_id, limit)
                    except Exception as e:
                        st.warning(f"FTS 搜索失败，fallback: {e}")
                        results = search_like(keyword, video_id, limit)
                
                if not results:
                    st.info("没有找到匹配结果")
                    return
                
                st.success(f"找到 {len(results)} 个结果")
                
                for i, result in enumerate(results):
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown(f"**Video {result.video_id}**")
                            st.caption(result.title[:30] if result.title else "Untitled")
                        with col2:
                            if result.source in ["transcript", "chunk_summary"]:
                                st.markdown(f"[{result.start:.1f}s - {result.end:.1f}s] {result.text[:200]}...")
                            else:
                                st.markdown(result.text[:200] + "...")
                            
                            source_color = {
                                "transcript": "🟢",
                                "chunk_summary": "🔵",
                                "final_summary": "🟡"
                            }.get(result.source, "⚪")
                            st.caption(f"{source_color} 来源: {result.source} | 得分: {result.score:.2f}")
                        
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"搜索失败: {str(e)}")


def render_qa_page():
    st.subheader("💬 视频问答")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        video_id = st.number_input("视频ID", min_value=1, step=1, value=1)
        
        videos = get_all_videos()
        video_options = {v['id']: v.get('title', 'Untitled') for v in videos}
        if video_id in video_options:
            st.info(f"当前视频: {video_options[video_id]}")
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["mock", "openai-compatible", "ollama"],
            index=0
        )
    
    with col2:
        question = st.text_area("输入问题", placeholder="例如：这个视频主要讲了什么？")
        
        if st.button("💬 提问", type="primary"):
            if not question:
                st.error("请输入问题")
                return
            
            info = get_video_info(video_id)
            if not info:
                st.error(f"视频 {video_id} 不存在")
                return
            
            if not info.get('transcript_count', 0) > 0:
                st.error(f"视频 {video_id} 没有转写数据")
                return
            
            with st.spinner("正在分析..."):
                try:
                    transcript = get_transcript_for_qa(video_id)
                    summaries = get_summary_for_qa(video_id)
                    final_summary = get_final_summary_for_qa(video_id)
                    
                    if not transcript and not summaries:
                        st.error("视频没有可用的转写或摘要数据")
                        return
                    
                    if llm_provider == "mock":
                        st.warning("使用 Mock LLM（模拟回答）")
                        mock_response = f'''{{
    "answer": "根据视频内容...（Mock模式，未使用真实LLM）",
    "evidence": [
        {{"timestamp": {transcript[0]['start'] if transcript else 0}, "text": "{transcript[0]['text'][:50] if transcript else 'N/A'}..."}},
        {{"timestamp": {transcript[1]['start'] if len(transcript) > 1 else 0}, "text": "{transcript[1]['text'][:50] if len(transcript) > 1 else 'N/A'}..."}}
    ],
    "cited_timestamps": [{transcript[0]['start'] if transcript else 0}, {transcript[1]['start'] if len(transcript) > 1 else 0}],
    "uncertainty": false
}}'''
                        result = parse_qa_response(mock_response)
                    else:
                        prompt = generate_qa_prompt(question, transcript, summaries, final_summary)
                        try:
                            from video_summarizer.summarizer.llm_client import get_llm_client
                            client = get_llm_client(llm_provider)
                            response = client.generate(prompt)
                            result = parse_qa_response(response)
                        except Exception as e:
                            st.error(f"LLM 调用失败: {str(e)}")
                            return
                    
                    st.markdown("### 回答")
                    
                    if result.get("uncertainty"):
                        st.warning(result["answer"])
                    else:
                        st.markdown(result["answer"])
                    
                    if result.get("evidence"):
                        st.markdown("#### 引用来源")
                        for ev in result["evidence"][:5]:
                            st.markdown(f"**[{ev['timestamp']:.1f}s]** {ev['text'][:100]}...")
                    
                    if result.get("cited_timestamps"):
                        st.caption(f"引用时间戳: {result['cited_timestamps']}")
                
                except Exception as e:
                    st.error(f"问答失败: {str(e)}")


def main():
    if 'selected_video_id' in st.session_state:
        render_video_detail(st.session_state.selected_video_id)
    else:
        render_home_page()


if __name__ == "__main__":
    main()
