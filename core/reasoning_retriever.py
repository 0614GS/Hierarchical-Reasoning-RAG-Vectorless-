from workflow.graph import search_workflow


def reasoning_retriever(query: str) -> list[str]:
    """
    :param query: 问题
    :return: 相关的文档块
    """
    res = search_workflow.invoke({"query": query})
    return res["content"]


if __name__ == "__main__":
    query = input("问题")
    content = reasoning_retriever(query)

    for content_block in content:
        print(content_block)
