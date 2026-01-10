import json
import os
from typing import List, Literal

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.types import Send, Command
from pydantic import BaseModel, Field

from data.storage import doc_tree_store, node_content_store
from core.workflow.prompts import global_index
from core.workflow.states import State, ReadTreeState, ReadNodeState

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

search_model = ChatOpenAI(
    model="MiniMaxAI/MiniMax-M2",
    temperature=0,
)

grade_model = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    temperature=0
)


def select_docs(state: State):
    class output(BaseModel):
        doc_ids: List[str] = Field(description="相关文档的doc_id")

    system_prompt = f"""你是一个 LangChain 生态系统的语义路由专家。
    你的任务是分析用户的提问，仔细阅读目录中对每个文件的summary和keywords，从提供的目录列表中挑选出所有可能回答用户问题的doc_id。

    【操作指南】：
    1. 优先选择直接相关的技术模块。
    2. 仅输出目录中的doc_id组成的列表。如果没有任何标签相关，请返回空列表 []。
    
    langchain生态文档目录如下：
    {global_index}"""

    query = state["query"]
    response = search_model.with_structured_output(schema=output).invoke([
        SystemMessage(
            content=system_prompt),
        HumanMessage(content=f"这是提问'{query}'")
    ])
    return {"doc_ids": response.doc_ids}


def fetch_tree_and_send(state: State):
    doc_ids = state["doc_ids"]
    if len(doc_ids) == 0:
        return Command(goto=END)
    trees = doc_tree_store.mget(doc_ids)
    # Send 必须把 query 也传过去，因为 parallel 运行的节点是在独立作用域，看不到父节点的 state['query']
    return [
        Send("select_nodes", {"query": state["query"], "catalog": json.dumps(tree)})
        for tree in trees
    ]


# 一个LLM处理一个文档，map-reduce 范式
def select_nodes(state: ReadTreeState):
    class output(BaseModel):
        node_ids: List[str] = Field(description="相关节点的node_id列表")

    query = state["query"]
    # 假设你在 state 中已经通过 doc_id 拿到了对应的 tree_structure
    tree_structure = state["catalog"]
    # print(tree_structure)
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
    response = search_model.with_structured_output(schema=output).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(f"用户的问题是：'{query}'。请给出最相关的 node_id列表。")
    ])
    # print(response)
    return {"node_ids": response.node_ids}


def fetch_node_and_send(state: State):
    node_ids = state["node_ids"]
    if len(node_ids) == 0:
        return Command(goto=END)

    # 如果有文档，跳转到生成回答
    # 注意：作为节点返回 Command 时，可以直接指定 goto 下一个节点
    return [
        Send("grade_node_content", {"query": state["query"], "node_id": node_id}) for node_id in node_ids
    ]


# LLM并行评价文档content，map-reduce范式
def grade_node_content(state: ReadNodeState):
    class output(BaseModel):
        ans: str = Field(description="文档是否与问题相关，只能“yes” or “no”")

    query = state["query"]
    node_id = state["node_id"]
    node_content = node_content_store.mget([node_id])[0]
    system_prompt = f"""
    你是一个文档阅读专家，你需要判断用户提供的内容是否与以下问题相关，只回答“yes” 或 “no”，若内容为空回复“no”：
    问题： {query}
    """
    response = grade_model.with_structured_output(schema=output).invoke([
        SystemMessage(system_prompt),
        HumanMessage(f"内容如下：{node_content}")
    ])
    print(response.ans)
    if response.ans == "yes":
        return {"final_node_ids": [node_id], "content": [node_content]}
    else:
        return None
