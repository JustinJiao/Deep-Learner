from pyspark import SparkContext
from config.settings import ResourceFactory, AppConfig
from ingestion.parsers import UniversalParser
from ingestion.chunkers import SemanticChunker
from ingestion.dual_writer import DualWriter
import os

# 🌟 自动定位项目根目录，确保 data 路径在任何地方执行都有效
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
absolute_docs_path = os.path.join(base_dir, AppConfig.DATA_PATH)

def process_file(path_binary):
    """
    Spark RDD 处理函数：解析 -> 切分
    """
    path, _ = path_binary
    local_path = path.replace("file:", "")
    
    # 实例化解析组件
    parser = UniversalParser()
    chunker = SemanticChunker()
    
    # 解析并切分
    try:
        md_text = parser.to_markdown(local_path)
        return chunker.split_with_overlap(md_text, local_path)
    except Exception as e:
        print(f"⚠️ 解析文件 {local_path} 失败: {e}")
        return []

if __name__ == "__main__":
    print(f"🚀 Deep-Learner Ingestion 启动")
    print(f"📂 扫描路径: {absolute_docs_path}")
    
    sc = SparkContext(appName="DeepLearner-Ingestion")
    raw_rdd = sc.binaryFiles(absolute_docs_path)
    
    # 过滤影子文件和系统隐藏文件
    raw_rdd = raw_rdd.filter(lambda x: 
        not os.path.basename(x[0]).startswith("~$") and 
        not os.path.basename(x[0]).startswith(".")
    )
    
    # 1. 分布式并行处理
    all_chunks = raw_rdd.flatMap(process_file).collect()
    
    if not all_chunks:
        print("⚠️ 未发现有效数据，请检查 data/docs/ 目录。")
    else:
        # 2. 批量写入 (利用工厂单例)
        writer = DualWriter()
        writer.write_all(all_chunks)
        
        print(f"🎉 流程结束：共入库 {len(all_chunks)} 个知识块。")