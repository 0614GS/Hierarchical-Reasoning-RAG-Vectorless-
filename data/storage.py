import json
import os

from langchain_classic.storage import LocalFileStore, EncoderBackedStore

# 获取当前文件 (storage.py) 的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取 storage.py 所在的目录 (core/)
current_dir = os.path.dirname(current_file_path)

# 获取项目根目录 (vectorless-RAG/)
project_root = os.path.dirname(current_dir)

# 拼接出准确的 data 路径
nodes_path = os.path.join(project_root, "data", "fs_store", "nodes")
docs_path = os.path.join(project_root, "data", "fs_store", "docs")

# 初始化存储器
node_fs = LocalFileStore(nodes_path)
doc_tree_fs = LocalFileStore(docs_path)

# 封装成支持字符串存取的 Store，用于存放(node_id, content)
node_content_store = EncoderBackedStore(
    store=node_fs,
    key_encoder=lambda k: k,
    value_serializer=lambda v: v.encode('utf-8'),
    value_deserializer=lambda v: v.decode('utf-8')
)

# 用于存放(doc_id, tree_structure)
doc_tree_store = EncoderBackedStore(
    store=doc_tree_fs,
    key_encoder=lambda k: k,
    # 存入：将 Python 对象转为 JSON 字符串，再转为字节
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
    # 取出：将字节转回字符串，再解析为 Python 对象（Dict/List）
    value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
)