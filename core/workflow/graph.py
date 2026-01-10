from langgraph.constants import START, END
from langgraph.graph import StateGraph

from core.workflow.nodes import *
from core.workflow.states import State

search_workflow = (
    StateGraph(State)
    # 选择文件
    .add_node(select_docs)
    # 选择某一个文件内节点
    .add_node(select_nodes)
    # 评分节点
    .add_node(grade_node_content)

    .add_edge(START, "select_docs")
    # map，send到挑选节点 select_nodes
    .add_conditional_edges("select_docs", fetch_tree_and_send)
    # reduce 汇聚边 并 send到评价节点grade_node_content
    .add_conditional_edges("select_nodes", fetch_node_and_send)
    # 直接汇聚到END
    .add_edge("grade_node_content", END)
    .compile()
)

if __name__ == "__main__":
    # print(node_content_store.mget(["0001"]))
    # for chunk in search_workflow.stream(
    #         {
    #             "query": input("提问：\n")
    #         },
    #         stream_mode="updates",
    # ):
    #     print(chunk)

    res = search_workflow.invoke({"query": "中间件应该怎么使用？"})
    print(res)
