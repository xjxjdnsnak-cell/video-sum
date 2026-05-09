# Video Summarizer

> **版本**: v0.2.1-local-robustness
> **目标**: 提升本地视频总结器的稳定性、断点续跑能力和可诊断性

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

### 1. 环境诊断

```bash
video-summarizer doctor
```

检查 Python 版本、FFmpeg、yt-dlp、faster-whisper、数据库路径、输出目录等。

### 2. Mock 流程测试（无需网络）

```bash
# 完整总结
video-summarizer summarize-local ./test.mp4 \
  --asr-provider mock \
  --llm-provider mock \
  --output ./outputs
```

### 3. 断点续跑

```bash
# 默认开启 --resume，已有的转写不会重复生成
video-summarizer summarize-local ./test.mp4 \
  --asr-provider mock \
  --llm-provider mock

# 第二次运行会自动跳过已完成的阶段
# 只会重新执行未完成的阶段

# 强制重跑全部阶段
video-summarizer summarize-local ./test.mp4 \
  --asr-provider mock \
  --llm-provider mock \
  --force
```

### 4. 查看任务状态

```bash
video-summarizer status VIDEO_ID
```

显示：视频来源、标题、当前阶段、转写片段数量、摘要状态、输出文件路径等。

### 5. 清理缓存

```bash
# 查看将要删除的文件（dry-run）
video-summarizer clean --temp-only

# 真正删除
video-summarizer clean --temp-only --yes

# 删除所有缓存（包括模型）
video-summarizer clean --all-cache --yes
```

---

## 真实 ASR 转写（需网络环境）

```bash
# 预下载模型
video-summarizer download-model --model tiny

# 转写视频
video-summarizer transcribe ./test.mp4 \
  --asr-provider faster-whisper \
  --model tiny \
  --device cpu
```

---

## 参数说明

### 断点续跑参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume` | True | 从现有检查点继续 |
| `--force` | False | 强制重跑全部阶段 |

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
| `--chunk-min` | 3 | 摘要分块最小分钟数 |
| `--chunk-max` | 5 | 摘要分块最大分钟数 |

---

## 命令

```bash
# 诊断
video-summarizer doctor                    # 环境检查

# 任务管理
video-summarizer status <VIDEO_ID>        # 查看任务状态
video-summarizer list                     # 列出所有视频
video-summarizer clean [选项]              # 清理缓存

# 处理
video-summarizer summarize-local <路径>    # 总结本地视频
video-summarizer summarize-url <URL>       # 总结 B站链接
video-summarizer transcribe <路径>         # 仅转写
video-summarizer export <VIDEO_ID>         # 导出已有视频

# 模型
video-summarizer download-model           # 下载 Whisper 模型
```

---

## Pipeline 阶段

```
created → audio_extracted → transcribed → chunked → summarized → exported
                                    ↓
                                failed
```

### 断点续跑规则

- 已有音频则不重复提取，除非 `--force`
- 已有 transcript_segments 则不重复 ASR，除非 `--force`
- 已有 summary_chunks 则不重复分段摘要，除非 `--force`
- 已有 final_summary 则不重复总摘要，除非 `--force`
- 导出可以重复执行

---

## 日志

每次运行生成 `logs/run-时间戳.log`，记录：
- 命令参数
- 每个阶段开始/结束
- 外部命令 stderr
- 错误堆栈
- 输出文件路径

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
- `doctor` - 环境诊断命令
- `status` - 任务状态查看
- `clean` - 清理命令
- `resume/force` - 断点续跑
- `ASR 隔离` - Mock ASR 不影响 Faster-Whisper
- `fallback 保护` - Faster-Whisper 失败时不会自动 fallback

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

## 当前版本限制

- **真实 ASR**: 需要在有网络的环境中验证 Whisper 模型下载
- **B站链接**: 部分视频需要登录 Cookie 才能访问

---

## 项目结构

```
video_summarizer/
├── cli.py                    # CLI 主入口
├── config.py                 # 配置管理
├── db.py                     # SQLite 数据库 + 断点续跑
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
