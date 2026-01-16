from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType
# from .parse import parse_markdown_structure
# from .chunk import split_into_semantic_chunks


class DeepIngestor:
    def __init__(self, app_name="DeepLearner-V3"):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()

    def run_pipeline(self, input_path):
        """
        核心流水线：控制数据的读取、转换和展示
        """
        # 读取
        raw_rdd = self.spark.sparkContext.wholeTextFiles(input_path)
        
        # 分布式转换
        chunks_rdd = raw_rdd.flatMap(lambda x : DeepIngestor.process_file_content(x))
        
        # 转为结构化 DataFrame
        df = chunks_rdd.toDF() # 自动推导 schema
        
        # 展开嵌套的 metadata 列，方便观察
        return df.select(
            col("id").alias("doc_id"),
            col("text").alias("content"),
            col("metadata.h1").alias("h1"),
            col("metadata.h2").alias("h2"),
            col("metadata.source").alias("source")
        )

    @staticmethod
    def process_file_content(path_content):
        """
        组合模块 B 和 C：这个函数将被分发到各个计算节点执行
        """
        path, content = path_content
        from .parse import parse_markdown_structure
        from .chunk import split_into_semantic_chunks
        # 1. 先解析结构
        structured_data = parse_markdown_structure(content)
        # 2. 再执行分块
        chunks = split_into_semantic_chunks(structured_data)
        
        # 3. 注入文件名信息
        for c in chunks:
            c['metadata']['source'] = path
        return chunks
# --- 执行入口 ---
if __name__ == "__main__":
    pipeline = DeepIngestor()
    # 1. 运行清洗流水线
    final_df = pipeline.run_pipeline("data/knowledge.txt")
    
    if final_df:
        # 2. 将 Spark DataFrame 转为 Python 列表 (中小规模数据直接 collect)
        # 如果数据量极大，建议在 run_pipeline 中使用 foreachPartition 分片写入
        data_to_write = [row.asDict() for row in final_df.collect()]
        
        # 3. 调用双路写入服务
        from .dual_writer import DualWriter
        writer = DualWriter()
        writer.write_all(data_to_write)
        
        print("\n🚀 [Success] 全体 Ingestion 任务完成！")