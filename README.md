# 🚀 Hierarchical Reasoning RAG (Vectorless)

> 一个基于文档层级结构（Hierarchical Tree）和 LLM 推理的检索增强生成系统。告别传统 Embedding 的“语义模糊”问题，实现极其精准的工业级文档检索。

---

## 💡 核心设计哲学 (The Core Philosophy)

本项目摒弃了传统的“切片 -> 向量化 -> 相似度匹配”流程，采用 **Tree-based Reasoning** 路径：

* **Structure-Aware Indexing**: 自动解析 Markdown 的 H1-H6 层级，保留文档的血统和逻辑关联。
* **Bottom-Up Synthesis**: 节点摘要自底向上汇聚。子节点的关键词支撑父节点，父节点的摘要浓缩子节点，形成“全方位、多维度”的导航树。
* **Global-to-Local Routing**: 
    1. 通过 `global_index.json` 确定文档范围。
    2. 通过精简的 `doc_nav_tree` 导航到具体的 `node_id`。
    3. 从 `node_content_store` 提取原子级正文。

---

## 🛠️ 技术特性 (Key Features)

- [x] **Markdown 结构化解析**: 自动构建 Tree 结构，支持代码块过滤，防止内容干扰。
- [x] **双层索引机制**:
    - **Global Index**: 跨文档导航，快速定位相关文件。
    - **Local Nav Tree**: 文档内导航，LLM 像读目录一样精准定位章节。
- [x] **原子级存储 (Content Store)**: 导航树与正文内容解耦，索引极其轻量（Token 消耗降低 80%）。
- [x] **Pydantic 强制 Schema**: 所有 LLM 输出均经过格式验证，确保 Summary 与 Keywords 的稳定性。
- [x] **高性能并发处理**: 采用异步 IO (`asyncio.gather`) 实现多文档、多节点的并行索引构建。

---

## 📂 项目结构 (Project Structure)
```text
├── data/
│   ├── input/              # 原始 Markdown 文档
│   ├── tree_results/       # 文档导航树索引 (doc_id.json)
│   ├── fs_store/ 
│   │   ├── docs # 存储文档树 (doc_id -> tree)
│   │   └── nodes # 原子正文存储 (node_id -> text)
│   ├── storage.py          # 基于 Key-Value 的存储实现
│   └── output/       # 文档导航树索引 (doc_id.json)
├── core/
│   ├── md2tree.py          # 核心: Markdown 解析与树构建逻辑
│   ├── workflow/            # LangGraph 节点处理逻辑
│   └── reasoning_retriever.py # 检索器
├── global_index.json       # 全局顶级索引 (doc_id, summary, keywords)
└── README.md
```
## 🚀 快速开始 (Getting Started)
### 1. 建立文档索引
将要构建索引的md文档放入 /input 文件夹，运行以下命令，系统将自动扫描文档，生成全局 ID，并构建多层级摘要：
``` python
python md2tree.py
```
### 2. 使用检索器retriever
```python
# 输入：str = query
# 输出：list = [content_1, content_2, ...]
```
### 3. 检索器逻辑
``` python
# 1. 加载全局索引
# 2. LLM 决策目标文档 (Doc Routing)
# 3. 加载目标文档的轻量级 Tree
# 4. LLM 决策目标节点 (Node Routing)
# 5. LLM 评价节点内容 (Node Grading)
# 6. 返回相关节点的内容
``` 
## 📊 数据 Schema 展示
### 节点索引 (Node Metadata)
每个节点在构建时都会参考子节点信息：
``` json
{
"node_id": "0006",
"path": "backends > Backends > Built-in backends > StoreBackend (LangGraph Store)",
"title": "StoreBackend (LangGraph Store)",
"keywords": ["StoreBackend", "LangGraph Store", "InMemoryStore", "BaseStore", "deep agents", "cross-thread storage"],
"summary": "Describes the configuration and usage of StoreBackend with LangGraph Store for durable cross-thread storage in deep agents.",
"nodes": []
}
```
### 文档树（Doc Tree）
每个文档有自己的doc_id，它的树是一个导航目录供LLM阅读
```json
{
  "doc_id": "doc_0001",
  "doc_name": "backends",
  "summary": "Configure and route filesystem backends for deep agents with policy enforcement, including built-in and custom options.",
  "keywords": [
    "filesystem backends",
    "deep agent",
    "CompositeBackend",
    "BackendProtocol",
    "policy hooks",
    "virtual filesystem"
  ],
  "structure": [{"node_1": "node"}, {"node_2": "node"}]
}
```
### 顶级目录
每个文件的metadata（doc_id, summary, keywords）列表
```json
[
  {
    "doc_id": "doc_0001",
    "doc_name": "backends",
    "keywords": [
      "filesystem backends",
      "deep agent",
      "CompositeBackend",
      "BackendProtocol",
      "policy hooks",
      "virtual filesystem"
    ],
    "summary": "Configure and route filesystem backends for deep agents with policy enforcement, including built-in and custom options."
  },
  {
    "doc_id": "doc_0002",
    "doc_name": "cli",
    "keywords": [
      "Deep Agents CLI",
      "persistent memory",
      "file operations",
      "shell commands",
      "web search",
      "HTTP requests",
      "task planning",
      "memory storage",
      "human approval"
    ],
    "summary": "Deep Agents CLI is an interactive terminal for building agents with persistent memory, supporting various operations like file handling, shell commands, web search, and more."
  }
]
```