# shujige - 通用本地文档混合检索系统 (Local Hybrid Document Search)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**shujige** 是一个轻量级、完全离线、高精度的本地 Markdown / 知识库混合检索系统。

通过结合 **SQLite FTS5 (BM25 词法检索，支持中文 n-gram 切词)** 与 **Dense Vector Embedding (语义向量检索)**，并使用 **RRF (Reciprocal Rank Fusion)** 进行混合重排，为个人知识库、Obsidian 笔记库、本地文章归档等提供兼具“字面精准匹配”与“深层语义理解”的检索能力。

---

## 🌟 特性亮点

- **双引擎混合检索 (Hybrid Search)**：
  - **关键词检索**：基于 SQLite FTS5，采用中文二字、三字滑动窗口片段及英文字词构建倒排索引，精准匹配专有名词、人名、日期与术语。
  - **语义检索**：基于 BGE 中文向量模型（如 `BAAI/bge-small-zh-v1.5`），在 CPU/GPU 上生成高维稠密向量，支持模糊问答与语义联想。
  - **RRF 融合重排**：结合倒数排名融合算法（Reciprocal Rank Fusion）平衡双路打分，检索结果排序更稳定。
- **纯本地离线运行**：所有文档分块、文本解析、向量计算与数据库查询全部在本地执行，无需网络请求，严密保护隐私数据。
- **专为 Markdown / Obsidian 优化**：
  - 自动剥离内嵌图片 Base64、HTML 标记与干扰字符，仅提取纯净正文与标题层级。
  - 智能滑动窗口分块，优先在自然标点句尾截断，保留上下文完整性。
  - 搜索结果直接提供 `obsidian://open?path=...` 协议链接，支持一键在 Obsidian 中定位并打开笔记。
- **极简通用与易集成**：
  - 单文件即为核心引擎（`hybrid_search.py`），无需复杂的外部向量数据库或重型检索服务。
  - 支持 CLI 查询与 `--json` 格式输出，方便被本地 AI Agent、自动化脚本或 Web 前端无缝集成。

---

## 🚀 快速开始

### 1. 安装环境

推荐使用 Python 3.9 及以上版本：

```bash
git clone https://github.com/yiquyibu/shujige.git
cd shujige

pip install -r requirements.txt
```

### 2. 准备文档

将你自己的 Markdown 笔记或文章放入任意目录（如仓库自带的 `./docs`，或你本机的 Obsidian 库目录）。

### 3. 构建索引

指定你的文档目录，执行建库：

```bash
# 对 ./docs 目录下的所有 markdown 文件构建索引（默认保存在 ./index）
python hybrid_search.py build --corpus ./docs

# 也可以直接指向外部的 Obsidian 笔记库或文章目录：
python hybrid_search.py build --corpus "D:/MyVault/Notes" --index ./my_index
```

可选参数：
- `--corpus` / `-c`：文档目录路径（会自动递归扫描所有 `.md` 文件）。
- `--index` / `-i`：索引文件保存目录（默认 `./index`）。
- `--model` / `-m`：指定 HuggingFace 模型名称或本地模型路径（默认 `BAAI/bge-small-zh-v1.5`）。
- `--chunk-size`：切块字符大小（默认 420）。
- `--overlap`：切块重叠字符数（默认 70）。
- `--if-needed`：仅在文档发生新增/修改时增量或按需构建。

### 4. 执行检索

```bash
# 默认混合检索模式 (Hybrid)
python hybrid_search.py search "什么是倒排索引与向量检索？"

# 仅使用关键词检索 (BM25)
python hybrid_search.py search "李世民" --mode keyword

# 仅使用语义向量检索 (Semantic)
python hybrid_search.py search "如何提升睡眠质量？" --mode semantic

# 指定返回前 5 条结果，以 JSON 格式输出供程序消费
python hybrid_search.py search "核心概念" --limit 5 --json
```

---

## 💻 命令行参数详解

### `build` 命令
```text
usage: hybrid_search.py build [-h] [--corpus CORPUS] [--index INDEX] [--model MODEL]
                              [--device DEVICE] [--chunk-size CHUNK_SIZE]
                              [--overlap OVERLAP] [--if-needed]

选项:
  --corpus, -c        文档目录路径（包含 .md 文件），默认自动检测 ./docs
  --index, -i         索引存储目录路径（默认 ./index）
  --model, -m         向量模型路径或 HuggingFace ID（默认 BAAI/bge-small-zh-v1.5）
  --device, -d        运行设备 (cpu / cuda)
  --chunk-size        分块大小（默认 420 字符）
  --overlap           分块重叠大小（默认 70 字符）
  --if-needed         若索引已存在且未变更则跳过构建
```

### `search` 命令
```text
usage: hybrid_search.py search [-h] [--mode {hybrid,keyword,semantic}]
                               [--limit LIMIT] [--index INDEX]
                               [--corpus CORPUS] [--model MODEL] [--json]
                               query

选项:
  query               检索查询语句
  --mode              检索模式：hybrid（默认混合）、keyword（纯关键词）、semantic（纯语义）
  --limit, -n         返回结果数量上限（默认 10）
  --index, -i         索引存储目录路径
  --corpus, -c        原始文档目录路径（用于定位文件并生成 obsidian 链接）
  --model, -m         向量模型路径或 HuggingFace ID
  --json              以 JSON 格式输出结果
```

---

## 🛠️ 项目结构

```text
shujige/
├── docs/                 # 示例与自定义文档存放目录
│   └── example.md        # 示例文档
├── hybrid_search.py      # 核心检索引擎（文档解析、切块、向量化、FTS5检索、混合重排）
├── requirements.txt      # Python 依赖清单
├── .gitignore            # 忽略本地私人文章、缓存与生成的二进制索引
└── README.md             # 项目使用说明
```

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。欢迎 Star、Fork 与贡献改进！