from langgraph.constants import START, END
from langgraph.graph import StateGraph

from nodes import *
from states import State

search_workflow = (
    StateGraph(State)
    .add_node(select_docs)
    .add_node(select_nodes)
    .add_node(nodes_check)
    .add_node(rewrite_query)
    .add_node(generate_answer)

    .add_edge(START, "select_docs")
    .add_conditional_edges("select_docs", fetch_catalog_and_send, ["select_nodes"])
    # 汇聚节点
    .add_edge("select_nodes", "nodes_check")
    # docs_check 内部通过 Command 决定去 rewrite 还是 generate
    .add_edge("rewrite_query", "select_docs")
    .add_edge("generate_answer", END)
    .compile()
)
