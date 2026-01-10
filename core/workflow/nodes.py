import json
import os
from typing import List

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.types import Send, Command
from pydantic import BaseModel, Field

from data.storage import doc_tree_store, node_content_store
from prompts import global_index
from states import State, GradeState

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0,
)


def select_docs(state: State):
    class output(BaseModel):
        doc_ids: List[str] = Field(description="相关文档的doc_id")

    system_prompt = f"""你是一个 LangChain 生态系统的语义路由专家。
    你的任务是分析用户的提问，仔细阅读目录中对每个文件的summary，从提供的目录列表中挑选出所有可能回答用户问题的doc_id。

    【操作指南】：
    1. 优先选择直接相关的技术模块。
    2. 仅输出目录中的doc_id组成的列表。如果没有任何标签相关，请返回空列表 []。
    
    langchain生态文档目录如下：
    {global_index}"""

    query = state["query"]
    response = model.with_structured_output(schema=output).invoke([
        SystemMessage(
            content=system_prompt),
        HumanMessage(content=f"这是提问'{query}'")
    ])
    # print(response)
    return {"doc_ids": response.doc_ids}


def fetch_catalog_and_send(state: State):
    doc_ids = state["doc_ids"]
    trees = doc_tree_store.mget(doc_ids)

    # Send 必须把 query 也传过去，因为 parallel 运行的节点是在独立作用域，看不到父节点的 state['query']
    return [
        Send("select_nodes", {"query": state["query"], "catalog": json.dumps(tree)})
        for tree in trees
    ]


def select_nodes(state: GradeState):
    class output(BaseModel):
        node_ids: List[str] = Field(description="相关节点的node_id列表")

    query = state["query"]
    # 假设你在 state 中已经通过 doc_id 拿到了对应的 tree_structure
    tree_structure = state["catalog"]
    system_prompt = f"""
    你是一个文档导航助手。你的任务是从给定的【文档层级树】中，识别出与用户问题最相关的节点 ID（node_id）。

    检索规则：
    1. 层级理解：文档采用树状结构。如果父节点的主题相关，请深入查看其子节点（nodes）。
    2. 多点检索：如果你不能确定某一个node一定包含相关主题，返回多个相关的 node_id。
    3. 排除无关：如果某些节点显然不相关，请忽略它们。
    4. 如果你确定没有相关主题，返回空列表

    ### 当前文档层级树：
    {tree_structure}

    ### 注意事项：
    - 请仅从上方提供的树结构中选择存在的 `node_id`。
    - 如果树中没有相关内容，请返回空列表。
    """

    # 调用模型
    response = model.with_structured_output(schema=output).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(f"用户的问题是：'{query}'。请给出最相关的 node_id 列表。")
    ])
    # print(response)
    # 此时你可以根据返回的 node_ids 去之前的 LocalFileStore (doc_tree_store)
    # 获取真正的 text 内容，或者直接在这里记录
    return {"node_ids": response.node_ids}


def nodes_check(state: State):
    current_loop = state.get("loop_count", 0)

    # 检查 web_pages 是否为空
    if not state.get("node_ids") or len(state["node_ids"]) == 0:
        # 如果重试次数超过 3 次，强制结束，防止死循环
        if current_loop >= 3:
            return Command(
                update={"answer": "抱歉，经过多次重试仍未找到相关文档。"},
                goto=END
            )

        return Command(
            update={
                "web_pages": [],
                "topics": [],  # 建议同时也清空 topics
                "loop_count": current_loop + 1
            },
            goto="rewrite_query"
        )

    # 如果有文档，跳转到生成回答
    # 注意：作为节点返回 Command 时，可以直接指定 goto 下一个节点
    return Command(goto="generate_answer")


def rewrite_query(state: State):
    class newQuery(BaseModel):
        new_query: str

    SYS_PROMPT = """你是一个搜索优化专家。用户的原始问题可能比较模糊或口语化，请将其重写为一个更适合在技术文档中检索的“独立搜索指令”。

    【要求】：
    1. 补全信息（例如Send指令扩充为：langgraph中的send指令）
    2. 提取核心技术名词。
    3. 保持简洁，不要输出解释，仅返回重写后的问题文本。
    4. 严禁反问用户。"""

    query = state["query"]
    response = model.with_structured_output(schema=newQuery).invoke([
        SystemMessage(SYS_PROMPT),
        HumanMessage(f"重写问题：{query}")
    ])
    return {"query": response.new_query}


def generate_answer(state: State):
    query = state["query"]
    node_ids = state["node_ids"]

    content = ""
    for content_block in node_content_store.mget(node_ids):
        content += content_block + "\n\n"

    print(content)

    system_prompt = f"""你是一个资深的 LangChain 技术支持工程师。
    请仔细阅读并总结下方提供的参考文档。

    【回复规范】：
    1. 忠于文档：仅基于提供的参考资料回答。如果资料中没有提到相关信息，请诚实告知用户你不知道，严禁编造 API 或参数。
    2. 结构清晰：使用 Markdown 格式，适当使用代码块展示示例。
    3. 术语准确：保持 LangChain 生态系统中的专业术语一致性。
    4. 简洁明了：直接回答问题，避免啰嗦。

    【参考资料】：{content}**"""
    response = model.invoke([
        SystemMessage(system_prompt),
        HumanMessage(query)
    ])
    return {"answer": response.content, "loop_count": 0}


if __name__ == "__main__":
    tree = doc_tree_store.mget(["0083"])[0]
    print(tree)
    res = select_nodes(GradeState(query="", catalog=json.dumps(tree)))
    print(node_content_store.mget(res["node_ids"]))

