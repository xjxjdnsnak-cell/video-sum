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

## 验收测试

### 1. Mock 流程验收（无需网络和模型下载）

用于测试完整流程，不需要真实 ASR 模型：

```bash
# 完整总结（Mock ASR + Mock LLM）
video-summarizer summarize-local ./test.mp4 \
  --asr-provider mock \
  --llm-provider mock \
  --output ./outputs

# 仅转写（Mock ASR）
video-summarizer transcribe ./test.mp4 \
  --asr-provider mock \
  --output ./outputs/transcript.json
```

预期输出：
- `outputs/test.md` - Markdown 总结（包含：一句话总结、详细总结、时间轴摘要、关键知识点、完整转写）
- `outputs/test.json` - JSON 结构数据
- `outputs/test.srt` - SRT 字幕

### 2. 真实 ASR 验收（需要网络下载 Whisper 模型）

```bash
# 先预下载模型（推荐）
video-summarizer download-model --model base

# 使用真实 Whisper ASR + Mock LLM
video-summarizer summarize-local ./test.mp4 \
  --asr-provider faster-whisper \
  --llm-provider mock \
  --output ./outputs

# 或使用真实 Whisper ASR + 真实 LLM
video-summarizer summarize-local ./test.mp4 \
  --asr-provider faster-whisper \
  --llm-provider openai \
  --model tiny \
  --output ./outputs
```

如果 Whisper 模型下载失败，会提示：
```
解决方案:
1. 检查网络连接
2. 预下载模型: video-summarizer download-model --model tiny
3. 尝试更小的模型: --model tiny
4. 或使用 Mock ASR 测试流程: --asr-provider mock
```

### 3. B站链接验收（需要网络）

```bash
# 总结 B站视频
video-summarizer summarize-url "https://www.bilibili.com/video/BVxxx" \
  --asr-provider faster-whisper \
  --llm-provider mock \
  --output ./outputs
```

如果 B站访问受限（HTTP 412/403），会提示需要设置 Cookie：
```
B站访问受限 (HTTP 412)。可能需要登录或设置Cookie。
解决方案：
1. 在浏览器中登录B站
2. 导出Cookie为Netscape格式
3. 保存到 ~/.video_summarizer/cookies.txt
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-provider` | faster-whisper | ASR 引擎：faster-whisper, mock |
| `--llm-provider` | mock | LLM 引擎：mock, openai, ollama |
| `--model` | base | Whisper 模型：tiny, base, small, medium, large |
| `--device` | cpu | 设备：cpu, cuda |
| `--language` | zh | 语言：zh, en, auto |
| `--output` | ~ | 输出目录 |
| `--keep-audio` | False | 保留中间音频文件 |
| `--force` | False | 强制重新转写 |

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
- `fallback 保护` - faster-whisper 失败时不会自动 fallback
