from pyspark import SparkContext
from config.settings import ResourceFactory, AppConfig
from ingestion.parsers import UniversalParser
from ingestion.chunkers import SemanticChunker
from ingestion.dual_writer import DualWriter
import os

# 获取项目根目录的绝对路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 🌟 修复报错点：通过 AppConfig 获取配置的相对路径并拼装成绝对路径
absolute_docs_path = os.path.join(base_dir, AppConfig.DATA_PATH)

def process_file(path_binary):
    """
    Spark RDD 处理函数：解析 -> 切分
    """
    path, _ = path_binary
    # 适配不同操作系统的文件路径格式
    local_path = path.replace("file:", "")
    
    parser = UniversalParser()
    chunker = SemanticChunker()
    
    # 解析为 Markdown
    md_text = parser.to_markdown(local_path)
    # 进行语义切分，并返回带 metadata 的字典列表
    return chunker.split_with_overlap(md_text, local_path)

if __name__ == "__main__":
    print(f"📂 正在扫描数据路径: {absolute_docs_path}")
    
    sc = SparkContext(appName="DeepLearner-Ingestion")
    raw_rdd = sc.binaryFiles(absolute_docs_path)
    
    # 过滤影子文件和隐藏文件（这是你之前的面试亮点逻辑）
    raw_rdd = raw_rdd.filter(lambda x: 
        not os.path.basename(x[0]).startswith("~$") and 
        not os.path.basename(x[0]).startswith(".")
    )
    
    # 1. 并行处理文件
    all_chunks = raw_rdd.flatMap(process_file).collect()
    
    if not all_chunks:
        print("⚠️ 未发现可处理的文件，请检查 data/docs/ 路径。")
    else:
        # 2. 写入数据库
        # 通过 Factory 动态获取已经配置好 Schema 的连接资源
        es_client = ResourceFactory.get_es_client()
        milvus_col = ResourceFactory.get_milvus_collection()
        
        writer = DualWriter(milvus_col, es_client)
        writer.write_all(all_chunks)
        
        print(f"✅ 成功处理并写入 {len(all_chunks)} 个文本块到双路索引。")