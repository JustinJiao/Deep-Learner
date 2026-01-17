from pyspark import SparkContext
from .config import ResourceFactory
from .parsers import UniversalParser
from .chunkers import SemanticChunker
from .dual_writer import DualWriter
from .config import AppConfig
import os
# 获取当前脚本所在目录的父目录（即项目根目录）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 拼接成绝对路径
absolute_docs_path= os.path.join(base_dir, AppConfig.DATA_PATH)

def process_file(path_binary):
    # 保持纯粹的业务逻辑：解析 -> 分块
    path, _ = path_binary
    local_path = path.replace("file:", "")
    
    parser = UniversalParser()
    chunker = SemanticChunker()
    
    md_text = parser.to_markdown(local_path)
    return chunker.split_with_overlap(md_text, local_path)

if __name__ == "__main__":
    sc = SparkContext(appName="DeepLearner-Ingestion")
    raw_rdd = sc.binaryFiles(absolute_docs_path)
    
    raw_rdd = raw_rdd.filter(lambda x: 
        not os.path.basename(x[0]).startswith("~$") and 
        not os.path.basename(x[0]).startswith(".")
    )
    # 1. 纯逻辑处理
    all_chunks = raw_rdd.flatMap(process_file).collect()
    
    # 2. 通过工厂动态获取连接资源，注入到写入器
    es_client = ResourceFactory.get_es_client()
    milvus_col = ResourceFactory.get_milvus_collection()
    
    writer = DualWriter(milvus_col, es_client)
    writer.write_all(all_chunks)
    print(f"总共处理并写入 {len(all_chunks)} 个文本块。")