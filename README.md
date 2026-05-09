# Video Summarizer

> **版本**: v0.2.0-real-asr
> **目标**: 让 faster-whisper 在真实视频上稳定转写

B站/本地视频总结器 - 自动提取音频/字幕，转成文字，按时间戳分段，调用 LLM 生成摘要，最后导出 Markdown 笔记。

## 安装

```bash
pip install -e .
```

## 依赖

- Python 3.10+
- FFmpeg
- yt-dlp

---

## 快速开始

### 1. Mock 流程测试（无需网络）

用于测试完整流程，不需要真实 ASR 模型：

```bash
# 完整总结（Mock ASR + Mock LLM）
video-summarizer summarize-local ./test.mp4 \
  --asr-provider mock \
  --llm-provider mock \
  --output ./outputs

# 仅转写（Mock ASR）
video-summarizer transcribe ./test.mp4 \
  --asr-provider mock
```

### 2. 真实 ASR 转写验收

在有网络的环境中，依次执行：

```bash
# Step 1: 预下载模型（可选，但推荐）
video-summarizer download-model --model tiny

# 或指定本地模型缓存目录
video-summarizer download-model --model tiny --model-dir ./models

# Step 2: 转写视频
video-summarizer transcribe ./test.mp4 \
  --asr-provider faster-whisper \
  --model tiny \
  --device cpu

# 成功标准：输出 JSON 和 SRT
# [
#   {
#     "start": 0.0,
#     "end": 3.2,
#     "text": "真实识别出来的内容"
#   }
# ]
```

### 3. B站链接测试

```bash
video-summarizer summarize-url "https://www.bilibili.com/video/BVxxx" \
  --asr-provider mock \
  --llm-provider mock
```

如果遇到 HTTP 412/403，需要设置 Cookie：
```
解决方案：
1. 在浏览器中登录B站
2. 导出Cookie为Netscape格式
3. 保存到 ~/.video_summarizer/cookies.txt
```

---

## 参数说明

### ASR 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-provider` | faster-whisper | ASR 引擎：faster-whisper, mock |
| `--model` | base | Whisper 模型：tiny, base, small, medium, large |
| `--device` | cpu | 设备：cpu, cuda, auto |
| `--language` | zh | 语言：zh, en, auto |
| `--model-dir` | None | 本地模型缓存目录 |

### LLM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm-provider` | mock | LLM 引擎：mock, openai, ollama |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | ~ | 输出目录 |
| `--keep-audio` | False | 保留中间音频文件 |
| `--force` | False | 强制重新转写 |
| `--chunk-min` | 3 | 摘要分块最小分钟数 |
| `--chunk-max` | 5 | 摘要分块最大分钟数 |

---

## 命令

```bash
# 总结本地视频
video-summarizer summarize-local <视频路径> [参数]

# 总结 B站链接
video-summarizer summarize-url <链接> [参数]

# 仅转写
video-summarizer transcribe <视频路径> [参数]

# 导出已有视频
video-summarizer export <视频ID>

# 列出所有视频
video-summarizer list

# 下载 Whisper 模型
video-summarizer download-model [参数]
```

---

## 配置

复制 `.env.example` 为 `.env` 并配置：

```bash
# LLM 配置
LLM_PROVIDER=mock
OPENAI_API_KEY=your-api-key

# Whisper 配置
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# 分块设置
CHUNK_DURATION_MIN=3
CHUNK_DURATION_MAX=5
```

---

## 测试

```bash
pytest tests/ -v
```

### 测试覆盖

- `timefmt` - 时间戳格式化
- `chunker` - 文本分块
- `subtitle_parser` - SRT 字幕解析
- `markdown_exporter` - Markdown 导出
- `mock pipeline` - Mock ASR + Mock LLM 完整流程
- `ASR 隔离` - Mock ASR 不影响 Faster-Whisper
- `fallback 保护` - Faster-Whisper 失败时不会自动 fallback
- `model_dir` - 本地模型目录参数传递

---

## 错误处理

如果 Whisper 模型下载失败：

```
解决方案:
1. 检查网络连接
2. 预下载模型: video-summarizer download-model --model tiny
3. 尝试更小的模型: --model tiny
4. 或使用 Mock ASR 测试流程: --asr-provider mock
```

---

## 项目结构

```
video_summarizer/
├── cli.py                    # CLI 主入口
├── config.py                 # 配置管理
├── db.py                     # SQLite 数据库
├── models.py                 # 数据模型
├── media/
│   ├── ffmpeg.py           # FFmpeg 音频提取
│   └── downloader.py       # yt-dlp 下载器
├── asr/
│   ├── faster_whisper_engine.py  # Whisper 引擎
│   └── subtitle_parser.py       # SRT 字幕解析
├── summarizer/
│   ├── chunker.py          # 文本分块
│   ├── llm_client.py       # LLM 客户端
│   ├── pipeline.py         # 摘要生成管道
│   └── prompts.py          # 提示词模板
├── exporters/
│   ├── markdown.py          # Markdown 导出
│   ├── srt.py             # SRT 字幕导出
│   └── json_exporter.py   # JSON 导出
└── utils/
    └── timefmt.py         # 时间戳格式化
```
