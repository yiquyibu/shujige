"""
shujige - 通用本地混合检索系统 (Local Hybrid Document Search Engine)

结合 SQLite FTS5 (BM25 词法检索，支持中文分词) 与 Dense Embedding (向量语义检索)，
实现高精度、纯离线、零外部依赖的个人知识库 / Markdown 文档混合检索。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from bs4 import BeautifulSoup, Comment, NavigableString
from markdown_it import MarkdownIt

# 默认分块参数
DEFAULT_CHUNK_SIZE = 420
DEFAULT_OVERLAP = 70
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
VERSION = 1

# 正则模式
IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(data:image/[^)]*\)", re.IGNORECASE)
MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LEXICAL_RUN = re.compile(r"[\u3400-\u9fff]+|[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
FACT_QUERY = re.compile(r"真假|假的|虚构|杜撰|史实|考证|属实|可信|编造|捏造|是不是真的")
FACT_VERDICT = re.compile(r"半真半假|假的|虚构|杜撰|不可考|纯属小说")
FACT_SUBJECT_END = re.compile(r"(?:故事|传说|情节|事件|说法|记载|内容|部分|的|是|中|里|有)+$")
BREAK_TAGS = {"article", "section", "div", "p", "li", "ul", "ol", "blockquote", "pre", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
SKIP_TAGS = {"script", "style", "svg", "noscript"}


def resolve_corpus_path(custom: str | Path | None = None) -> Path:
    """自动解析文档源目录"""
    if custom:
        p = Path(custom).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"指定的文档目录不存在：{custom}")
        return p
    # 查找默认候选路径
    here = Path(__file__).resolve().parent
    candidates = [
        here / "docs",
        here / "修复版网页",
        here.parent / "修复版网页",
        Path.cwd() / "docs",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.md")):
            return c.resolve()
    # 默认返回 docs 目录
    default_dir = here / "docs"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def resolve_index_path(custom: str | Path | None = None) -> Path:
    """自动解析索引目录"""
    if custom:
        return Path(custom).resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "检索系统" / "index",
        here / "index",
        Path.cwd() / "index",
    ]
    for c in candidates:
        if c.is_dir() and (c / "chunks.sqlite3").is_file():
            return c.resolve()
    return (here / "index").resolve()


def resolve_model_target(custom: str | None = None) -> str:
    """自动检测向量模型（优先本地路径，其次环境变量，最后 HuggingFace Hub）"""
    if custom:
        return custom
    if os.environ.get("EMBEDDING_MODEL"):
        return os.environ["EMBEDDING_MODEL"]
    local_default = Path(r"D:\AI_project\LM\Tips_For_MiniLM\bge-small-zh-v1.5")
    if local_default.is_dir():
        return str(local_default)
    return "BAAI/bge-small-zh-v1.5"


def find_documents(corpus_dir: Path) -> list[Path]:
    """获取所有待检索的 Markdown 文档"""
    files = sorted(corpus_dir.glob("**/*.md"))
    if not files:
        files = sorted(corpus_dir.glob("**/*.txt"))
    if not files:
        raise RuntimeError(f"未在目录中找到可检索的 Markdown (.md) 文件：{corpus_dir}")
    return files


def lexical_terms(text: str) -> list[str]:
    """生成中文 2-gram、3-gram 及英文 token 列表用于 FTS5 倒排索引"""
    terms: list[str] = []
    for match in LEXICAL_RUN.finditer(text.lower()):
        token = match.group()
        if '\u3400' <= token[0] <= '\u9fff':
            if len(token) == 1:
                terms.append(token)
            else:
                terms.extend(token[i:i + 2] for i in range(len(token) - 1))
                terms.extend(token[i:i + 3] for i in range(len(token) - 2))
        else:
            terms.append(token)
    return terms


def fact_check_subject(query: str) -> str:
    """提取真伪考证类查询的主语"""
    if not FACT_QUERY.search(query):
        return ""
    subject = re.split(r"哪些|哪几|哪个|什么|是否|真假|假的|真的|虚构|杜撰|史实|考证|属实|可信|编造|捏造", query, maxsplit=1)[0]
    subject = FACT_SUBJECT_END.sub("", subject.strip(" ？?，,：:"))
    subject = re.sub(r"^(?:请问|关于|有关|我想知道)", "", subject)
    return subject if len(subject) >= 2 else ""


def _flatten(node, parts: list[str]) -> None:
    if isinstance(node, Comment):
        return
    if isinstance(node, NavigableString):
        parts.append(str(node))
        return
    name = getattr(node, "name", None)
    if name in SKIP_TAGS or name == "img":
        return
    if name == "br":
        parts.append("\n")
        return
    if name in BREAK_TAGS:
        parts.append("\n")
    for child in getattr(node, "children", []):
        _flatten(child, parts)
    if name in BREAK_TAGS:
        parts.append("\n")


def extract_text(path: Path, renderer: MarkdownIt) -> tuple[str, list[tuple[int, str]]]:
    """从 Markdown 提取纯文本与章节标题层级，剥离图片与无用标记"""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # 剥离内嵌图片 Base64 与标准 Markdown/HTML 图片
    raw = IMAGE_LINK.sub(lambda m: m.group(1).strip() if len(m.group(1).strip()) > 2 else "", raw)
    raw = MD_IMAGE.sub(r"\1", raw)
    raw = HTML_IMAGE.sub("", raw)
    raw = HTML_COMMENT.sub("", raw)

    soup = BeautifulSoup(renderer.render(raw), "html.parser")
    parts: list[str] = []
    _flatten(soup, parts)
    lines = [" ".join(line.split()) for line in "".join(parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    headings: list[tuple[int, str]] = []
    cursor = 0
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        label = " ".join(heading.get_text(" ", strip=True).split())
        if label:
            pos = text.find(label, cursor)
            if pos >= 0:
                headings.append((pos, label))
                cursor = pos + len(label)
    return text, headings


def split_chunks(text: str, headings: list[tuple[int, str]], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """滑动窗口与自然句尾断句切块"""
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            lower = start + int(chunk_size * 0.65)
            boundary = max((i for i in range(lower, end) if text[i] in "\n。！？；;"), default=-1)
            if boundary >= 0:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            heading = next((label for pos, label in reversed(headings) if pos <= start), "")
            yield ordinal, start, end, heading, chunk
            ordinal += 1
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1


def load_model(model_target: str, device: str | None = None):
    """加载向量模型"""
    import torch
    from sentence_transformers import SentenceTransformer

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    if Path(model_target).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = SentenceTransformer(model_target, device=device)
    model.max_seq_length = 512
    return model


def build_index(
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    model_name: str | None = None,
    device: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    if_needed: bool = False,
) -> None:
    """构建或更新索引"""
    corpus_dir = resolve_corpus_path(corpus_dir)
    index_dir = resolve_index_path(index_dir)
    model_target = resolve_model_target(model_name)
    files = find_documents(corpus_dir)

    manifest_path = index_dir / "manifest.json"
    if if_needed and all((index_dir / name).is_file() for name in ("manifest.json", "chunks.sqlite3", "vectors.npy")):
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (old.get("version") == VERSION and old.get("chunk_size") == chunk_size
                    and old.get("overlap") == overlap and old.get("documents") == len(files)):
                print(f"索引已存在且未检测到更新：{index_dir}")
                return
        except Exception:
            pass

    renderer = MarkdownIt("commonmark", {"html": True}).enable("table")
    scratch = index_dir / ".scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="build_", dir=scratch))
    db_path = stage / "chunks.sqlite3"
    con = sqlite3.connect(db_path)

    try:
        con.executescript("""
            CREATE TABLE documents (id INTEGER PRIMARY KEY, name TEXT UNIQUE, title TEXT, relpath TEXT, text TEXT);
            CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER, ordinal INTEGER,
                char_start INTEGER, char_end INTEGER, heading TEXT, text TEXT);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(terms, content='');
            CREATE INDEX chunks_doc ON chunks(document_id, ordinal);
        """)
        embed_inputs: list[str] = []
        total_chars = 0
        for number, path in enumerate(files, 1):
            text, headings = extract_text(path, renderer)
            if not text.strip():
                continue
            title = path.stem
            try:
                relpath = str(path.relative_to(corpus_dir))
            except ValueError:
                relpath = path.name
            cur = con.execute("INSERT INTO documents(name, title, relpath, text) VALUES (?, ?, ?, ?)",
                              (path.name, title, relpath, text))
            doc_id = cur.lastrowid
            total_chars += len(text)
            for ordinal, start, end, heading, chunk in split_chunks(text, headings, chunk_size, overlap):
                cur = con.execute("INSERT INTO chunks(document_id, ordinal, char_start, char_end, heading, text) VALUES (?, ?, ?, ?, ?, ?)",
                                  (doc_id, ordinal, start, end, heading, chunk))
                con.execute("INSERT INTO chunks_fts(rowid, terms) VALUES (?, ?)",
                            (cur.lastrowid, " ".join(lexical_terms(title + " " + heading + " " + chunk))))
                embed_inputs.append((title[:60] + "\n" + heading[:30] + "\n" + chunk)[:480])

            if number % 50 == 0 or number == len(files):
                print(f"抽取文本进度：{number}/{len(files)} 篇文档...", flush=True)

        con.commit()
        con.close()

        if not embed_inputs:
            raise RuntimeError(f"在 {corpus_dir} 中未能抽取到有效文本片段")

        print(f"正在生成向量：共 {len(embed_inputs)} 个片段，使用模型 {model_target}...", flush=True)
        model = load_model(model_target, device=device)
        batches = []
        batch_size = 32 if device == "cuda" else 16
        for offset in range(0, len(embed_inputs), 128):
            batch = model.encode(embed_inputs[offset:offset + 128], batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype("float32")
            batches.append(batch)
            if (offset + len(batch)) % 512 < 128 or offset + len(batch) == len(embed_inputs):
                print(f"向量进度：{offset + len(batch)}/{len(embed_inputs)} 个片段", flush=True)
        vectors = np.vstack(batches)
        np.save(stage / "vectors.npy", vectors)

        manifest = {
            "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": str(corpus_dir),
            "model": model_target,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "documents": len(files),
            "text_chars": total_chars,
            "chunks": len(embed_inputs),
            "vector_dimensions": int(vectors.shape[1]),
            "dependencies": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version}
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        index_dir.mkdir(parents=True, exist_ok=True)
        for name in ("chunks.sqlite3", "vectors.npy", "manifest.json"):
            target_file = index_dir / name
            if target_file.exists():
                target_file.unlink()
            os.replace(stage / name, target_file)

        print("\n=== 索引构建成功 ===")
        print(f"文档总数: {manifest['documents']}")
        print(f"字符总数: {manifest['text_chars']}")
        print(f"片段总数: {manifest['chunks']}")
        print(f"向量维度: {manifest['vector_dimensions']}")
        print(f"保存位置: {index_dir}\n")
    finally:
        try:
            con.close()
        except Exception:
            pass
        if stage.exists():
            import shutil
            shutil.rmtree(stage, ignore_errors=True)


class SearchEngine:
    """本地混合检索引擎"""
    def __init__(self, index_dir: Path | None = None, corpus_dir: Path | None = None, model_name: str | None = None):
        self.index_dir = resolve_index_path(index_dir)
        self.corpus_dir = resolve_corpus_path(corpus_dir)
        db_file = self.index_dir / "chunks.sqlite3"
        vec_file = self.index_dir / "vectors.npy"

        if not db_file.is_file() or not vec_file.is_file():
            raise RuntimeError(f"索引文件缺失（{self.index_dir}）。请先运行: python hybrid_search.py build")

        self.con = sqlite3.connect(db_file, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.vectors = np.load(vec_file, mmap_mode="r")

        # 检查 manifest 读取 corpus 与 model
        manifest_file = self.index_dir / "manifest.json"
        if manifest_file.is_file():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                saved_corpus = manifest_data.get("corpus")
                if saved_corpus and not corpus_dir and Path(saved_corpus).is_dir():
                    self.corpus_dir = Path(saved_corpus)
                saved_model = manifest_data.get("model")
                if saved_model and not model_name:
                    model_name = saved_model
            except Exception:
                pass

        self.model_target = resolve_model_target(model_name)
        self.model = load_model(self.model_target)

        # 标记结尾区域（如精选留言等干扰内容）
        self.article_ends = {}
        for row in self.con.execute("SELECT id, text FROM documents"):
            marker = row["text"].find("\n精选留言\n")
            self.article_ends[row["id"]] = marker if marker >= 0 else len(row["text"])

    def search(self, query: str, mode: str = "hybrid", limit: int = 10) -> list[dict]:
        query = " ".join(query.strip().split())
        if not query:
            return []
        if mode not in {"hybrid", "keyword", "semantic"}:
            raise ValueError("mode 必须是 hybrid、keyword 或 semantic")
        limit = max(1, int(limit))
        count = max(400, limit * 4)
        ranks: dict[int, dict[str, int]] = {}

        # 1. BM25 关键词检索 (SQLite FTS5)
        if mode in {"hybrid", "keyword"}:
            terms = list(dict.fromkeys(lexical_terms(query)))[:64]
            if terms:
                expression = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
                try:
                    rows = self.con.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                        (expression, count)
                    ).fetchall()
                    for rank, row in enumerate(rows, 1):
                        ranks.setdefault(row[0], {})["keyword"] = rank
                except sqlite3.OperationalError:
                    pass

        # 2. 语义向量检索 (BGE Embedding)
        if mode in {"hybrid", "semantic"}:
            vector = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0]
            similarities = np.asarray(self.vectors) @ vector
            top = np.argsort(similarities)[-count:][::-1]
            for rank, index in enumerate(top, 1):
                ranks.setdefault(int(index) + 1, {})["semantic"] = rank

        if not ranks:
            return []

        ids = list(ranks)
        placeholders = ",".join("?" for _ in ids)
        rows = self.con.execute(
            f"SELECT c.id, c.document_id, c.ordinal, c.char_start, c.heading, c.text, d.name, d.title "
            f"FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id IN ({placeholders})", ids
        )

        candidates = []
        normalized_query = query.casefold()
        subject = fact_check_subject(query)
        document_focus: dict[int, float] = {}
        if subject:
            for doc in self.con.execute("SELECT id, title, text FROM documents"):
                body = doc["text"][:self.article_ends.get(doc["id"], len(doc["text"]))]
                if subject not in doc["title"] and body[:2400].count(subject) < 2:
                    continue
                verdicts = len(FACT_VERDICT.findall(body))
                if verdicts >= 2:
                    document_focus[doc["id"]] = 0.006 * min(verdicts, 5)

        for row in rows:
            body_end = self.article_ends.get(row["document_id"], len(row["text"]))
            if row["char_start"] >= body_end:
                continue
            visible_text = row["text"][:body_end - row["char_start"]].rstrip()
            if len(visible_text) < 20:
                continue
            found = ranks[row["id"]]
            # RRF (Reciprocal Rank Fusion)
            score = sum((1.1 if name == "keyword" else 1.0) / (60 + rank) for name, rank in found.items())
            if normalized_query in row["title"].casefold():
                score += 0.006
            elif normalized_query in visible_text.casefold():
                score += 0.003
            focus = document_focus.get(row["document_id"], 0.0)
            if focus:
                score += focus + 0.004 * min(len(FACT_VERDICT.findall(visible_text)), 3)
            candidates.append((score, row, found, visible_text))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        per_doc: dict[int, int] = {}
        for score, row, found, visible_text in candidates:
            doc_id = row["document_id"]
            if per_doc.get(doc_id, 0) >= 2:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            source_file = self.corpus_dir / row["name"]
            results.append({
                "chunk_id": row["id"],
                "document_id": doc_id,
                "title": row["title"],
                "source": row["name"],
                "heading": row["heading"],
                "position": row["ordinal"] + 1,
                "char_start": row["char_start"],
                "text": visible_text,
                "file_path": str(source_file),
                "obsidian_url": "obsidian://open?path=" + quote(str(source_file), safe=""),
                "score": round(score, 5),
                "matched_by": list(found)
            })
            if len(results) >= limit:
                break
        return results

    def document(self, document_id: int) -> dict | None:
        row = self.con.execute("SELECT id, name, title, text FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description="通用本地混合检索系统（Hybrid Search）- BM25 关键词 + 向量语义检索")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build = sub.add_parser("build", help="建立或更新关键词与向量索引")
    build.add_argument("--corpus", "-c", type=str, default=None, help="文档目录路径（包含 .md 文件），默认自动检测 ./docs 或现有知识库")
    build.add_argument("--index", "-i", type=str, default=None, help="索引存储目录路径（默认 ./index）")
    build.add_argument("--model", "-m", type=str, default=None, help="向量模型路径或 HuggingFace ID（默认 BAAI/bge-small-zh-v1.5）")
    build.add_argument("--device", "-d", type=str, default=None, help="运行设备 (cpu / cuda)")
    build.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"分块大小（默认 {DEFAULT_CHUNK_SIZE} 字符）")
    build.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help=f"分块重叠大小（默认 {DEFAULT_OVERLAP} 字符）")
    build.add_argument("--if-needed", action="store_true", help="若索引未变更则跳过构建")

    # search
    search = sub.add_parser("search", help="执行文档检索")
    search.add_argument("query", help="检索查询语句")
    search.add_argument("--mode", choices=("hybrid", "keyword", "semantic"), default="hybrid", help="检索模式（默认 hybrid 混合检索）")
    search.add_argument("--limit", "-n", type=int, default=10, help="返回结果数量上限（默认 10）")
    search.add_argument("--index", "-i", type=str, default=None, help="索引存储目录路径")
    search.add_argument("--corpus", "-c", type=str, default=None, help="原始文档目录路径（用于定位文件）")
    search.add_argument("--model", "-m", type=str, default=None, help="向量模型路径或 HuggingFace ID")
    search.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")

    args = parser.parse_args()

    if args.command == "build":
        build_index(
            corpus_dir=args.corpus,
            index_dir=args.index,
            model_name=args.model,
            device=args.device,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            if_needed=args.if_needed
        )
    elif args.command == "search":
        engine = SearchEngine(index_dir=args.index, corpus_dir=args.corpus, model_name=args.model)
        results = engine.search(args.query, mode=args.mode, limit=args.limit)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            if not results:
                print(f"未找到与 '{args.query}' 相关的匹配结果。")
                return
            print(f"\n找到 {len(results)} 条相关结果：\n" + "-" * 60)
            for i, res in enumerate(results, 1):
                matched = ",".join(res["matched_by"])
                heading_info = f" | {res['heading']}" if res.get("heading") else ""
                print(f"\n{i}. [{res['score']}] {res['title']}{heading_info} (片段 #{res['position']}，匹配源: {matched})")
                print(f"   摘要: {res['text'][:200].replace(chr(10), ' ')}...")
                print(f"   路径: {res['file_path']}")
                print(f"   链接: {res['obsidian_url']}")
            print("-" * 60)


if __name__ == "__main__":
    main()
