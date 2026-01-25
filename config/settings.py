import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from elasticsearch import Elasticsearch

load_dotenv()

class AppConfig:
    """全局静态配置，包含模型与基础设施参数"""
    # 基础设施地址
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = os.getenv("ES_PORT", "9200")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    
    # 模型 Provider 选择: 'openai' 或 'ollama'
    LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai")

    # OpenAI 参数
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Ollama 参数 (针对你的 Windows 模型端配置)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    
    # 存储与索引名称
    ES_INDEX = os.getenv("ES_INDEX_NAME", "deep_learner_knowledge")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME", "deep_learner_vectors")
    
    # 算法规格
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))

    # 检索流水线参数
    RRF_K = int(os.getenv("RRF_K", 60))
    RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", 50))
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 5))
    
    # Reranker 配置
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH")
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
    
class ResourceFactory:
    """单例资源工厂：确保全项目共用连接池"""
    _es_client = None
    _milvus_col = None
    _llm_service = None
    _embed_service = None

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
                # 如果不存在则初始化 (Ingestion 逻辑)
                fields = [
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=AppConfig.EMBEDDING_DIM),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON) 
                ]
                schema = CollectionSchema(fields, description="Deep-Learner Vector Store")
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION, schema)
                index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
                cls._milvus_col.create_index(field_name="vector", index_params=index_params)
            else:
                cls._milvus_col = Collection(AppConfig.MILVUS_COLLECTION)
            cls._milvus_col.load()
        return cls._milvus_col

    @classmethod
    def get_llm_service(cls):
        from llm.llm_service import LLMService
        if cls._llm_service is None:
            cls._llm_service = LLMService()
        return cls._llm_service

    @classmethod
    def get_embedding_service(cls):
        from llm.embedding_service import EmbeddingService
        if cls._embed_service is None:
            cls._embed_service = EmbeddingService()
        return cls._embed_service