# config/settings.py
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from elasticsearch import Elasticsearch

load_dotenv()

class AppConfig:
    """全局静态配置，所有模块共享"""
    # 基础设施
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = os.getenv("ES_PORT", "9200")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    
    # 路径与名称
    DATA_PATH = os.getenv("DATA_SOURCE_PATH", "data/docs/*")
    ES_INDEX = os.getenv("ES_INDEX_NAME", "deep_learner_knowledge")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME", "deep_learner_vectors")
    
    # 核心算法参数
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_LINES", 3))

class ResourceFactory:
    """单例资源工厂：确保整个项目（入库+检索）共用连接池"""
    _es_client = None
    _milvus_col = None

    @classmethod
    def get_es_client(cls):
        if cls._es_client is None:
            cls._es_client = Elasticsearch([f"http://{AppConfig.ES_HOST}:{AppConfig.ES_PORT}"])
        return cls._es_client

    @classmethod
    def get_milvus_collection(cls):
        """
        检索模块也会调用此方法：
        如果集合已存在，直接加载并返回；如果不存在（仅在入库阶段），则初始化 Schema。
        """
        if cls._milvus_col is None:
            connections.connect("default", host=AppConfig.MILVUS_HOST, port=AppConfig.MILVUS_PORT)
            
            if not utility.has_collection(AppConfig.MILVUS_COLLECTION):
                # 此部分逻辑主要服务于 Ingestion 阶段的初始化
                fields = [
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=AppConfig.EMBEDDING_DIM),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON) 
                ]
                schema = CollectionSchema(fields, description="Deep-Learner Vector Store")
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION, schema)
                
                # 创建索引
                index_params = {
                    "metric_type": "L2", 
                    "index_type": "HNSW", 
                    "params": {"M": 8, "efConstruction": 64}
                }
                cls._milvus_col.create_index(field_name="vector", index_params=index_params)
            else:
                # 检索模块运行到这里时，直接获取现有集合
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION)
            
            # 💡 无论入库还是检索，都必须 Load 进内存才能操作
            cls._milvus_col.load()
        return cls._milvus_col