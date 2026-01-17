import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from elasticsearch import Elasticsearch

load_dotenv()

class AppConfig:
    """全局静态配置，统一管理所有参数"""
    # 架构
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = os.getenv("ES_PORT", "9200")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    
    # 名称
    ES_INDEX = os.getenv("ES_INDEX_NAME", "deep_learner_knowledge")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME", "deep_learner_vectors")
    
    # 算法参数 (确保转换为 int 类型)
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_LINES", 3))
    
    # 路径
    DATA_PATH = os.getenv("DATA_SOURCE_PATH", "data/docs/*")

class ResourceFactory:
    """资源工厂，引用 AppConfig 中的参数"""
    _es_client = None
    _milvus_col = None

    @classmethod
    def get_es_client(cls):
        if cls._es_client is None:
            cls._es_client = Elasticsearch([f"http://{AppConfig.ES_HOST}:{AppConfig.ES_PORT}"])
        return cls._es_client

    @classmethod
    def get_milvus_collection(cls):
        if cls._milvus_col is None:
            connections.connect("default", host=AppConfig.MILVUS_HOST, port=AppConfig.MILVUS_PORT)
            
            if not utility.has_collection(AppConfig.MILVUS_COLLECTION):
                fields = [
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    # 使用动态维度
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=AppConfig.EMBEDDING_DIM),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="h1", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="h2", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500)
                ]
                schema = CollectionSchema(fields, description="Vector Store")
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION, schema)
                
                # 索引参数同样可以提取，这里保持 HNSW 结构
                index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
                cls._milvus_col.create_index(field_name="vector", index_params=index_params)
            else:
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION)
            
            cls._milvus_col.load()
        return cls._milvus_col