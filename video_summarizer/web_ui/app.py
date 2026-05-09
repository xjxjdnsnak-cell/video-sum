import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_summarizer.config import settings
from video_summarizer.db import get_all_videos, get_video_info
from video_summarizer.summarizer.pipeline import (
    get_transcript, get_summary_chunks, get_final_summary,
    get_chapters, get_quotes, get_terms
)
from video_summarizer.evaluator.evaluate import evaluate_video, generate_markdown_report
from video_summarizer.media.ffmpeg import check_ffmpeg_installed
from video_summarizer.media.downloader import check_ytdlp_installed
from video_summarizer.exporters.markdown import export_markdown
from video_summarizer.exporters.json_exporter import export_json
from video_summarizer.exporters.srt import export_srt
from video_summarizer.summarizer.prompts import NoteStyle


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
    
    tab1, tab2 = st.tabs(["📤 新建任务", "📋 历史记录"])
    
    with tab1:
        render_new_task_form()
    
    with tab2:
        render_history()


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
    
    st.markdown("---")
    
    if st.button("🚀 开始处理", type="primary", use_container_width=True):
        if input_type == "📁 本地视频" and not video_file:
            st.error("请先上传视频文件")
            return
        if input_type == "🔗 B站链接" and not video_url:
            st.error("请先输入B站链接")
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
                            keep_audio=keep_audio
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
                        chunk_max=chunk_max
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
    keep_audio: bool
) -> int:
    from video_summarizer.cli import create_video_record, update_video_duration, update_video_status, update_video_stage
    from video_summarizer.media.ffmpeg import extract_audio, get_video_duration, FFmpegError
    from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine, FasterWhisperError
    from video_summarizer.summarizer.pipeline import save_transcript, summarize_video_pipeline
    from video_summarizer.exporters.markdown import export_markdown
    from video_summarizer.exporters.json_exporter import export_json
    from video_summarizer.exporters.srt import export_srt
    from video_summarizer.summarizer.prompts import NoteStyle as NS
    import tempfile
    
    video_path = Path(video_path)
    video_title = video_path.stem
    
    video_id = create_video_record(
        source_type="local",
        source_path=str(video_path),
        title=video_title
    )
    
    update_video_status(video_id, "processing", "created")
    
    try:
        duration = get_video_duration(str(video_path))
        update_video_duration(video_id, duration)
        
        update_video_stage(video_id, "audio_extracted")
        
        audio_path = Path(tempfile.gettempdir()) / f"video_summarizer_{video_id}.wav"
        extract_audio(str(video_path), str(audio_path))
        
        update_video_stage(video_id, "transcribed")
        
        use_mock = asr_provider == "mock"
        
        if use_mock:
            transcript_data = [
                {"start": 0.0, "end": 5.0, "text": "[Mock转写] 这是第一段转写内容", "source": "mock"},
                {"start": 5.0, "end": 10.0, "text": "[Mock转写] 这是第二段转写内容", "source": "mock"},
            ]
        else:
            engine = FasterWhisperEngine(
                model_name=whisper_model,
                device=device,
                compute_type="float16" if device == "cuda" else "int8",
                use_mock=False,
                language=language if language != "auto" else None
            )
            segments = engine.transcribe(str(audio_path), language=language if language != "auto" else None)
            transcript_data = [
                {"start": s.start, "end": s.end, "text": s.text, "source": "asr"}
                for s in segments
            ]
        
        save_transcript(video_id, transcript_data)
        
        if not keep_audio and audio_path.exists():
            audio_path.unlink()
        
        note_style_enum = NS[note_style.upper()]
        result = summarize_video_pipeline(
            video_id,
            llm_provider=llm_provider,
            chunk_min=chunk_min,
            chunk_max=chunk_max,
            note_style=note_style_enum
        )
        
        update_video_stage(video_id, "exported")
        
        export_markdown(
            video_id=video_id,
            video_title=video_title,
            transcript=transcript_data,
            chunk_summaries=result["chunks"],
            final_summary=result["final_summary"],
            chapters=result.get("chapters", []),
            quotes=result.get("quotes", []),
            note_style=note_style_enum
        )
        
        update_video_status(video_id, "completed", "exported")
        
        return video_id
    
    except Exception as e:
        update_video_status(video_id, "failed")
        raise


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
    chunk_max: int = 5
) -> int:
    from video_summarizer.cli import create_video_record, update_video_status, update_video_stage
    from video_summarizer.media.downloader import download_audio, download_subtitles, get_video_info, DownloaderError
    from video_summarizer.asr.subtitle_parser import parse_srt
    from video_summarizer.summarizer.pipeline import save_transcript, summarize_video_pipeline
    from video_summarizer.summarizer.prompts import NoteStyle as NS
    from video_summarizer.exporters.markdown import export_markdown
    import tempfile
    
    video_id = create_video_record(
        source_type="url",
        url=url,
        title=f"B站视频 {url}"
    )
    
    update_video_status(video_id, "processing", "created")
    
    try:
        info = get_video_info(
            url,
            cookies_file=cookies_file
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            srt_path = download_subtitles(url, temp_path, cookies_file=cookies_file)
            
            if srt_path and srt_path.exists():
                transcript_data = parse_srt(str(srt_path))
                transcript_data = [
                    {"start": s.start, "end": s.end, "text": s.text, "source": "subtitle"}
                    for s in transcript_data
                ]
                save_transcript(video_id, transcript_data)
                update_video_stage(video_id, "transcribed")
            else:
                audio_path = temp_path / "audio.wav"
                download_audio(url, audio_path, cookies_file=cookies_file)
                update_video_stage(video_id, "audio_extracted")
                
                use_mock = asr_provider == "mock"
                if use_mock:
                    transcript_data = [
                        {"start": 0.0, "end": 5.0, "text": "[Mock转写] 这是第一段转写内容", "source": "mock"},
                        {"start": 5.0, "end": 10.0, "text": "[Mock转写] 这是第二段转写内容", "source": "mock"},
                    ]
                else:
                    from video_summarizer.asr.faster_whisper_engine import FasterWhisperEngine
                    engine = FasterWhisperEngine(
                        model_name=whisper_model,
                        device=device,
                        use_mock=False,
                        language=language if language != "auto" else None
                    )
                    segments = engine.transcribe(str(audio_path), language=language if language != "auto" else None)
                    transcript_data = [
                        {"start": s.start, "end": s.end, "text": s.text, "source": "asr"}
                        for s in segments
                    ]
                
                save_transcript(video_id, transcript_data)
                update_video_stage(video_id, "transcribed")
            
            note_style_enum = NS[note_style.upper()]
            result = summarize_video_pipeline(
                video_id,
                llm_provider=llm_provider,
                chunk_min=chunk_min,
                chunk_max=chunk_max,
                note_style=note_style_enum
            )
            
            update_video_stage(video_id, "exported")
            
            export_markdown(
                video_id=video_id,
                video_title=info.title,
                transcript=transcript_data,
                chunk_summaries=result["chunks"],
                final_summary=result["final_summary"],
                chapters=result.get("chapters", []),
                quotes=result.get("quotes", []),
                note_style=note_style_enum
            )
            
            update_video_status(video_id, "completed", "exported")
            
            return video_id
    
    except DownloaderError as e:
        update_video_status(video_id, "failed")
        raise Exception(f"B站下载失败: {str(e)}")
    except Exception as e:
        update_video_status(video_id, "failed")
        raise


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
                        output_path = export_srt(transcript, f"{info.get('title', 'video')}.srt")
                    
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


def main():
    if 'selected_video_id' in st.session_state:
        render_video_detail(st.session_state.selected_video_id)
    else:
        render_home_page()


if __name__ == "__main__":
    main()
