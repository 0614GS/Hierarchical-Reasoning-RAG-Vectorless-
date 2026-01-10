import operator
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document


# 专门给 Map 任务用的 State Schema
class ReadTreeState(TypedDict):
    query: str
    catalog: str


class ReadNodeState(TypedDict):
    query: str
    node_id: str


class State(TypedDict):
    query: str
    doc_ids: List[str]
    node_ids: Annotated[List[str], operator.add]
    final_node_ids: Annotated[List[str], operator.add]
    content: Annotated[List[str], operator.add]
