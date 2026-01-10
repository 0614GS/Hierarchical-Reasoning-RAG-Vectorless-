import operator
from typing import TypedDict, List, Annotated


# 专门给 Map 任务用的 State Schema
class GradeState(TypedDict):
    query: str
    catalog: str


class State(TypedDict):
    query: str
    doc_ids: List[str]
    node_ids: Annotated[list[str], operator.add]
    answer: str
    loop_count: int
