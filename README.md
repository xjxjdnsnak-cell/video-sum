# Video Summarizer

> **版本**: v0.4.1-quality-evaluation
> **目标**: 评估摘要质量，支持回归测试机制

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

## B站视频处理

### 1. 检查视频信息

```bash
video-summarizer inspect-url "BV1xx411c7mZ"
video-summarizer inspect-url "https://www.bilibili.com/video/BV1xx411c7mZ"
```

显示：标题、UP主、时长、视频ID、分P数量、是否有字幕、可用字幕语言、是否需要登录等。

### 2. 支持的链接格式

- BV号: `BV1xx411c7mZ`
- av号: `av12345678`
- 完整链接: `https://www.bilibili.com/video/BV1xx411c7mZ`
- 短链接: `https://b23.tv/BV1xx411c7mZ`

### 3. 使用 Cookie 登录

部分视频需要登录才能观看，可使用以下方式：

```bash
# 使用 Cookie 文件
video-summarizer summarize-url "BV1xx411c7mZ" \
  --cookies /path/to/cookies.txt \
  --llm-provider mock

# 从浏览器导入 Cookie
video-summarizer summarize-url "BV1xx411c7mZ" \
  --cookies-from-browser chrome \
  --llm-provider mock
```

### 4. 字幕优先策略

默认情况下，工具会：
1. 先检查是否有官方字幕或自动字幕
2. 如果有字幕，直接解析为文本（无需 ASR）
3. 如果没有字幕，再下载音频并进行语音转写

```bash
# 只下载字幕（不进行 ASR）
video-summarizer summarize-url "BV1xx411c7mZ" \
  --download-subtitle-only \
  --llm-provider mock

# 只下载音频（不检查字幕）
video-summarizer summarize-url "BV1xx411c7mZ" \
  --download-audio-only \
  --llm-provider mock
```

### 5. 代理支持

```bash
video-summarizer summarize-url "BV1xx411c7mZ" \
  --proxy http://127.0.0.1:7890 \
  --llm-provider mock
```

### 6. 如何导出 Cookie

#### 方法一：使用浏览器扩展

1. 安装 Cookie-Editor 扩展（Chrome/Firefox）
2. 登录 B站
3. 打开 Cookie-Editor
4. 点击"导出" -> 选择"Netscape"格式
5. 保存为 `cookies.txt`

#### 方法二：手动导出

```bash
# 使用 yt-dlp 导出
yt-dlp --cookies-from-browser chrome --dump-json "https://www.bilibili.com" > /dev/null
# Cookie 会自动保存到 ~/.cache/yt-dlp/cookies/chrome.sqlite
```

### 7. B站链接失败的常见原因

| 错误码 | 原因 | 解决方案 |
|--------|------|----------|
| HTTP 412 | 需要登录 | 使用 `--cookies` 或 `--cookies-from-browser` |
| HTTP 403 | 访问被拒绝 | 检查视频权限，可能需要登录 |
| 视频不存在 | 视频已删除或私密 | 检查链接是否正确 |
| 地区限制 | 仅限特定地区观看 | 使用 `--proxy` 参数 |

### 8. 重要声明

本工具**不承诺**绕过版权和付费限制：
- 仅用于个人学习和研究目的
- 请遵守 B站用户协议和相关法律法规
- 付费视频需要购买后才能观看
- 版权保护的内容请获得授权后使用

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

# B站工具
video-summarizer inspect-url <URL/BV号>    # 检查视频信息
video-summarizer summarize-url <URL/BV号>  # 总结 B站视频

# 任务管理
video-summarizer status <VIDEO_ID>        # 查看任务状态
video-summarizer list                     # 列出所有视频
video-summarizer clean [选项]              # 清理缓存

# 处理
video-summarizer summarize-local <路径>    # 总结本地视频
video-summarizer transcribe <路径>         # 仅转写
video-summarizer export <VIDEO_ID>         # 导出已有视频

# 模型
video-summarizer download-model           # 下载 Whisper 模型
```

---

## B站参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cookies` | None | Cookie文件路径 (Netscape格式) |
| `--cookies-from-browser` | None | 从浏览器导入Cookie: chrome/firefox/edge/brave/safari |
| `--proxy` | None | 代理服务器地址 |
| `--user-agent` | 默认UA | 自定义User-Agent |
| `--download-subtitle-only` | False | 只下载字幕，不进行ASR |
| `--download-audio-only` | False | 只下载音频，不检查字幕 |

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

## 笔记模板

使用 `--note-style` 参数选择不同的笔记输出模板：

| 模板 | 说明 |
|------|------|
| `brief` | 一句话总结 + 重点列表 + 精选引用 |
| `detailed` | 详细总结 + 章节目录 + 时间轴摘要 + 核心观点 |
| `study` | 知识点 + 术语解释 + 复习问题 + 易错点 |
| `meeting` | 议题 + 决策 + 待办事项 + 责任人 |
| `tutorial` | 步骤拆解 + 命令/代码块 + 注意事项 + 前置要求 |

```bash
# 生成学习笔记
video-summarizer summarize-local video.mp4 --note-style study --llm-provider mock

# 生成简短笔记
video-summarizer summarize-local video.mp4 --note-style brief --llm-provider mock
```

## 评估与回归测试

使用 `evaluate` 命令评估已有总结的质量：

```bash
video-summarizer evaluate VIDEO_ID
video-summarizer evaluate VIDEO_ID --format json
```

### 评估维度

| 维度 | 说明 |
|------|------|
| completeness | 是否覆盖主要内容 |
| faithfulness | 是否忠于原始转写 |
| timestamp_accuracy | 时间戳是否来自真实 transcript/chunk |
| structure_quality | 结构是否清晰 |
| note_usefulness | 是否适合复习/学习 |
| hallucination_risk | 是否有编造风险 |

### 规则校验

评估器会进行以下规则校验：
- Quote 必须能在 transcript 中找到
- 时间戳必须落在 transcript/chunk 范围内
- Final summary 不得出现完全不存在的高风险实体
- 空摘要、过短摘要、重复摘要会报 warning

---

## 当前版本限制

- **真实 ASR**: 需要在有网络的环境中验证 Whisper 模型下载
- **B站链接**: 部分视频需要登录 Cookie 才能访问，建议使用 `--cookies` 或 `--cookies-from-browser` 参数
- **付费视频**: 本工具不承诺绕过付费限制，付费内容需要购买后才能观看

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
