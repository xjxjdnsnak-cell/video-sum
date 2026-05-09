# Video Summarizer

B站/本地视频总结器 - 自动提取音频/字幕，转成文字，按时间戳分段，调用 LLM 生成摘要，最后导出 Markdown 笔记。

## 安装

```bash
pip install -e .
```

## 依赖

- Python 3.10+
- FFmpeg
- yt-dlp

## 使用方法

### 本地视频总结

```bash
video-summarizer summarize-local path/to/video.mp4 --llm-provider mock
```

### B站链接总结

```bash
video-summarizer summarize-url "https://www.bilibili.com/video/BVxxx"
```

### 仅转写

```bash
video-summarizer transcribe path/to/video.mp4
```

### 导出

```bash
video-summarizer export VIDEO_ID
video-summarizer list
```

## 配置

复制 `.env.example` 为 `.env` 并配置：

- `LLM_PROVIDER`: mock/openai/ollama
- `OPENAI_API_KEY`: OpenAI API 密钥
- `WHISPER_MODEL`: whisper 模型大小

## 命令

- `summarize-local`: 总结本地视频
- `summarize-url`: 总结B站视频
- `transcribe`: 仅转写视频
- `export`: 导出已有视频
- `list`: 列出所有视频
