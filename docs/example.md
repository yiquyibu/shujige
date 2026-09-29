# 示例知识库文档：本地混合检索系统使用指南

这是一个示例 Markdown 文档，用于验证本地混合检索系统的分块、倒排索引与向量计算功能。

## 一、系统简介

本地混合检索系统（Hybrid Search）专为个人笔记、Obsidian 知识库和本地文档设计。它结合了：
1. **SQLite FTS5 倒排索引**：基于 BM25 算法的关键词精确匹配，结合中文二字/三字片段切分，提供毫秒级全文检索。
2. **Dense Vector 语义向量**：基于 BGE 等开源嵌入模型，捕捉自然语言语义与上下文关联，即使关键词未完全匹配也能找到语义相关的段落。
3. **RRF 融合重排**：通过倒数排名融合（Reciprocal Rank Fusion）平衡两路检索结果，大幅提升 Top-K 检索的准确度。

## 二、快速上手

如果你刚克隆此仓库，可以按照以下步骤快速体验：

1. 安装依赖环境：
   ```bash
   pip install -r requirements.txt
   ```
2. 针对指定文档目录构建索引：
   ```bash
   python hybrid_search.py build --corpus ./docs
   ```
3. 执行检索：
   ```bash
   python hybrid_search.py search "什么是倒排索引与向量检索？"
   ```

## 三、适用场景

- 个人 Obsidian 仓库与 Markdown 笔记检索
- 微信公众号/长篇博客文章本地归档查询
- AI 智能体（Agent）或 RAG 系统本地知识检索模块
- 完全离线隐私保护环境下的本地搜索
