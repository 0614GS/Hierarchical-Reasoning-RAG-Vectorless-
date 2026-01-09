import glob
import json
import os
import re
import time
from typing import List, Dict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from data.storage import doc_tree_store
from data.storage import node_content_store

# 加载环境变量
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENROUTER_API_KEY")
# os.environ['OPENAI_BASE_URL'] = os.getenv("OPENROUTER_API_KEY")

# 关键词和摘要的提取模型
extract_model = ChatOpenAI(model="Qwen/Qwen2.5-32B-Instruct", temperature=0)


# extract_model = ChatOpenAI(model="Qwen/Qwen3-Omni-30B-A3B-Instruct", temperature=0)
# extract_model = ChatOpenAI(model="gpt-oss-120b", temperature=0)


class GlobalCounter:
    """全局计数器，用于生成跨文件的唯一 ID"""

    def __init__(self, prefix=""):
        self.count = 0
        self.prefix = prefix

    def get_next(self):
        self.count += 1
        return f"{self.prefix}{str(self.count).zfill(4)}"


# 初始化全局计数器
node_counter = GlobalCounter(prefix="")  # 生成 0001, 0002...
doc_counter = GlobalCounter(prefix="doc_")  # 生成 doc_0001...


# 获取扁平化的node列表
def extract_nodes_from_markdown(markdown_content: str) -> List[Dict]:
    """
    解析 Markdown 内容，将每个标题及其下方的文本提取为一个节点列表。

    Args:
        markdown_content: Markdown 文件内容字符串

    Returns:
        包含 'level', 'title', 'text' 的扁平节点列表
    """
    # 匹配 Markdown 标题 (例如: ## Title)
    header_pattern = r'^(#{1,6})\s+(.+)$'
    # 匹配代码块标记，用于避免在代码块内部匹配标题
    code_block_pattern = r'^```'

    lines = markdown_content.split('\n')
    node_list = []
    current_node = None
    in_code_block = False

    # 虚拟根节点，用于捕获文件开头没有标题的内容（如果有）
    # 但通常 markdown 都是从 # Title 开始

    for line in lines:
        stripped_line = line.strip()

        # 1. 状态检查：是否在代码块中
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block

        # 2. 检查是否是标题行 (且不在代码块中)
        header_match = re.match(header_pattern, stripped_line)
        if header_match and not in_code_block:
            # 如果之前有正在处理的节点，先保存其文本内容
            if current_node:
                current_node['text'] = '\n'.join(current_node['text_lines']).strip()
                del current_node['text_lines']  # 移除临时列表
                node_list.append(current_node)
                # print(current_node)

            # 创建新节点
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_node = {
                'level': level,
                'title': title,
                'text_lines': []
            }
        else:
            # 3. 如果是普通行，归属于当前节点
            if current_node:
                current_node['text_lines'].append(line)

    # 处理最后一个节点
    if current_node:
        current_node['text'] = '\n'.join(current_node['text_lines']).strip()
        del current_node['text_lines']
        node_list.append(current_node)

    return node_list


# 从扁平化列表中构建带有text的树
def build_tree_from_flat_nodes(node_list: List[Dict]) -> List[Dict]:
    """
    使用栈算法将扁平的节点列表转换为嵌套的树结构。
    逻辑源自 page_index_md.py 的 build_tree_from_nodes。
    """
    if not node_list:
        return []

    stack = []  # 用于追踪父节点路径 [(node, level), ...]
    root_nodes = []  # 最终的树根列表

    for node in node_list:
        current_level = node['level']

        # 基础树节点结构
        tree_node = {
            'title': node['title'],
            'text': node['text'],  # 暂时保留 text 用于给 LLM 分析
            'nodes': []  # 子节点列表
        }

        # 栈逻辑：如果栈顶节点的层级 >= 当前层级，说明栈顶节点不是当前节点的父级
        # 需要弹出，直到找到一个层级比当前小的节点（即父节点）
        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            # 如果栈空了，说明当前节点是顶层节点（Root）
            root_nodes.append(tree_node)
        else:
            # 栈顶元素即为父节点，将当前节点加入其 nodes 列表
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)

        # 将当前节点压入栈，作为潜在的下一级父节点
        stack.append((tree_node, current_level))

    return root_nodes


# 使用大模型提取关键词与摘要
def generate_metadata_with_llm(title: str, path: str, content: str, children_summary: str = "") -> Dict:
    """
    调用大模型生成 Summary 和 Keywords。
    为了稳定性，使用了简单的重试机制。
    """

    class outputSchema(BaseModel):
        keywords: List[str] = Field(description="一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等）")
        summary: str = Field(description="50字以内的中文内容极简摘要，需涵盖子章节的核心主题")

    system_prompt = """
    你是一个专业的技术文档分析助手。请根据提供的文档节点信息提取元数据。
    请返回指定格式，包含以下字段：
    1. "summary": 50字以内的中文内容极简摘要。如果是父节点，需涵盖子节点的核心主题，in English。
    2. "keywords": 一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等），in English。
    """

    user_prompt = f"""
    文档路径: {path}
    章节标题: {title}
    本章节内容:
    {content if content else "（无直接正文）"}
    
    子章节摘要内容：
    {children_summary if children_summary else "（无子章节）"}
    """

    print("正在提取title:", title, "的关键词与总结")
    # print("子章节内容：", children_summary, "\n")
    print("*" * 100)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = extract_model.with_structured_output(schema=outputSchema).invoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            return response.model_dump()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {path}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(5)  # 等待后重试


# 树的递归处理与 ID 生成
def process_tree_recursive(nodes: List[Dict], parent_path: str) -> List[Dict]:
    """
    递归遍历树，为每个节点生成 ID、Path、Summary 和 Keywords。
    """
    processed_nodes = []

    for node in nodes:
        # 构建当前节点的完整路径 (Path)
        current_title = node['title']
        current_path = f"{parent_path} > {current_title}" if parent_path else current_title

        # 获取 node_id
        node_id = node_counter.get_next()

        # 立即存入本地内容存储 (node_id -> text)
        content = node.get('text', '')
        node_content_store.mset([
            (node_id, content)
        ])

        print(f"  [Storage] Saved content for node {node_id}")

        children = []
        children_info_for_parent = ""
        if node.get('nodes'):
            # 等待子节点处理完成
            children = process_tree_recursive(node['nodes'], current_path)

            # 3. 【汇总子信息】将所有子节点的 title 和 summary 拼接，给父节点参考
            summary_list = [
                (f"- {child['title']}: {child['summary']} "
                 f"(Keywords: {', '.join(child['keywords'])})") for child in children
            ]
            children_info_for_parent = "\n".join(summary_list)

        # 生成当前节点任务
        metadata = generate_metadata_with_llm(
            title=current_title,
            path=current_path,
            content=content,
            children_summary=children_info_for_parent  # 传入参考信息
        )

        final_node = {
            "node_id": node_id,
            "path": current_path,  # 修正变量名为 current_path
            "title": current_title,
            "keywords": metadata.get("keywords", []),
            "summary": metadata.get("summary", ""),
            "nodes": children
        }
        processed_nodes.append(final_node)

    return processed_nodes


# 主流程
def analyze_markdown_file(file_path: str):
    """
    主函数：读取文件 -> 解析 -> 处理 -> 保存
    """
    print(f"正在处理文件: {file_path}")

    # 读取 Markdown 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取扁平节点
    flat_nodes = extract_nodes_from_markdown(content)
    print("扁平节点", flat_nodes)

    # 构建树状结构
    tree_structure = build_tree_from_flat_nodes(flat_nodes)
    print("树形结构", tree_structure)

    # 递归增强节点信息 (ID, Path, Summary, Keywords)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    processed_tree = process_tree_recursive(
        nodes=tree_structure,
        parent_path=file_name  # 将文件名作为 Path 的第一级
    )

    # 获取文件的metadata
    doc_overview = generate_doc_global_summary(file_name, processed_tree)

    # 为整个文档生成 doc_id 并存入 doc_tree_store
    doc_id = doc_counter.get_next()
    doc_data = {
        "doc_id": doc_id,
        "doc_name": file_name,
        "summary": doc_overview["summary"],
        "keywords": doc_overview["keywords"],
        "structure": processed_tree
    }

    doc_tree_store.mset([
        (doc_id, doc_data)
    ])

    print(f"文档 {file_name} 处理完成。DocID: {doc_id}, 导航树已存入 doc_tree_store")

    return doc_data


def generate_doc_global_summary(doc_name: str, level1_nodes: List[Dict]) -> Dict:
    """基于所有一级标题的信息，生成整篇文档的总览摘要和关键词"""

    class outputSchema(BaseModel):
        keywords: List[str] = Field(description="一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等）")
        summary: str = Field(description="50字以内的中文内容极简摘要。如果是父节点，需涵盖子节点的核心主题")

    # 汇总一级标题的信息作为上下文
    context_list = [f"标题: {n['title']}\n摘要: {n['summary']}\n关键词: {', '.join(n['keywords'])}" for n in
                    level1_nodes]
    context_text = "\n\n".join(context_list)

    system_prompt = "你是一个文档索引专家。请根据文档各章节的摘要，为整篇文档生成一份总览元数据。"
    user_prompt = f"""
    文档名称: {doc_name}
    各章节核心内容汇总:
    {context_text}

    你是一个专业的技术文档分析助手。请根据提供的文档节点信息提取元数据。
    请返回指定格式，包含以下字段：
    1. "summary": 50字以内的中文内容极简摘要。如果是父节点，需涵盖子节点的核心主题，in English。
    2. "keywords": 一个字符串列表，包含5-10个关键技术名词（API名称、特定概念等），in English。
    """

    print("各章节汇总信息\n", context_text, "\n\n")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = extract_model.with_structured_output(schema=outputSchema).invoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            return response.model_dump()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error generating metadata for {doc_name}: {e}")
                return {"summary": "生成失败", "keywords": []}
            time.sleep(1)  # 等待后重试


# 批量处理主入口
def batch_process_markdowns(input_dir: str, output_dir: str):
    """批量处理入口"""
    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    if not md_files:
        print("未找到任何 .md 文件")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 用于存储所有文档的元数据，最后生成全局索引
    global_index_list = []

    for md_file in md_files:
        doc_result = analyze_markdown_file(md_file)

        with open(os.path.join(output_dir, doc_result["doc_id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(doc_result, f, indent=2, ensure_ascii=False)

        # 收集元数据用于全局索引
        global_index_list.append({
            "doc_id": doc_result["doc_id"],
            "doc_name": doc_result["doc_name"],
            "keywords": doc_result["keywords"],
            "summary": doc_result["summary"]
        })

        print(f"收集文档: {doc_result['doc_name']} (ID: {doc_result['doc_id']})")

    # 生成顶层目录索引 json
    global_index_path = os.path.join(output_dir, "global_index.json")
    with open(global_index_path, "w", encoding="utf-8") as f:
        json.dump(global_index_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    INPUT_DIR = "../data/input/deepagents"
    OUTPUT_DIR = "../data/output"

    batch_process_markdowns(INPUT_DIR, OUTPUT_DIR)
