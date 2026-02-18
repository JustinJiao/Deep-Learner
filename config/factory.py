# config/factory.py
from typing import Optional

try:
    from elasticsearch import Elasticsearch  # type: ignore
except Exception:  # pragma: no cover
    Elasticsearch = None  # type: ignore

try:
    from pymilvus import (  # type: ignore
        connections,
        Collection,
        FieldSchema,
        CollectionSchema,
        DataType,
        utility,
    )
except Exception:  # pragma: no cover
    connections = Collection = FieldSchema = CollectionSchema = DataType = utility = None  # type: ignore

from config.settings import AppConfig


class ResourceFactory:
    """
    单例资源工厂：确保全项目共用连接池
    （第一阶段迁移版：逻辑保持不变）
    """

    _es_client: Optional[Elasticsearch] = None
    _milvus_col: Optional[Collection] = None
    _milvus_ltm_col: Optional[Collection] = None
    _llm_service = None
    _embed_service = None

    # ---------- Elasticsearch ----------

    @classmethod
    def get_es_client(cls) -> Elasticsearch:
        if Elasticsearch is None:
            raise ImportError('elasticsearch package is not installed; ES retrieval is unavailable')
        if cls._es_client is None:
            cls._es_client = Elasticsearch(
                [f"http://{AppConfig.ES_HOST}:{AppConfig.ES_PORT}"]
            )
        return cls._es_client

    # ---------- Milvus: 主知识库 ----------

    @classmethod
    def get_milvus_collection(cls) -> Collection:
        if connections is None:
            raise ImportError('pymilvus is not installed; Milvus is unavailable')
        if cls._milvus_col is None:
            connections.connect(
                "default",
                host=AppConfig.MILVUS_HOST,
                port=AppConfig.MILVUS_PORT,
            )

            if not utility.has_collection(AppConfig.MILVUS_COLLECTION):
                fields = [
                    FieldSchema(
                        name="doc_id",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        max_length=100,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=AppConfig.EMBEDDING_DIM,
                    ),
                    FieldSchema(
                        name="content",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                    ),
                    FieldSchema(
                        name="metadata",
                        dtype=DataType.JSON,
                    ),
                ]

                schema = CollectionSchema(
                    fields,
                    description="Deep-Learner Vector Store",
                )

                cls._milvus_col = Collection(
                    AppConfig.MILVUS_COLLECTION, schema
                )

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                }
                cls._milvus_col.create_index(
                    field_name="vector",
                    index_params=index_params,
                )
            else:
                cls._milvus_col = Collection(
                    AppConfig.MILVUS_COLLECTION
                )

            cls._milvus_col.load()

        return cls._milvus_col

    # ---------- Milvus: 长期记忆 ----------

    @classmethod
    def get_milvus_ltm_collection(cls) -> Collection:
        if connections is None:
            raise ImportError('pymilvus is not installed; Milvus LTM is unavailable')

        if cls._milvus_ltm_col is None:
            connections.connect(
                "default",
                host=AppConfig.MILVUS_HOST,
                port=AppConfig.MILVUS_PORT,
            )

            collection_name = AppConfig.MILVUS_LTM_COLLECTION

            if not utility.has_collection(collection_name):
                fields = [
                    FieldSchema(
                        name="key",
                        dtype=DataType.VARCHAR,
                        is_primary=True,
                        auto_id=False,
                        max_length=255,
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=AppConfig.EMBEDDING_DIM,
                    ),
                    FieldSchema(
                        name="content",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                    ),
                    FieldSchema(
                        name="type",
                        dtype=DataType.VARCHAR,
                        max_length=50,
                    ),
                    FieldSchema(
                        name="score",
                        dtype=DataType.FLOAT,
                    ),
                    FieldSchema(
                        name="timestamp",
                        dtype=DataType.INT64,
                    ),
                ]


                schema = CollectionSchema(
                    fields,
                    description="Deep-Learner Long-Term Memory (Structured)",
                )

                cls._milvus_ltm_col = Collection(
                    collection_name,
                    schema,
                )

                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64},
                }

                cls._milvus_ltm_col.create_index(
                    field_name="vector",
                    index_params=index_params,
                )
            else:
                cls._milvus_ltm_col = Collection(collection_name)

            cls._milvus_ltm_col.load()

        return cls._milvus_ltm_col


    # ---------- LLM / Embedding ----------

    @classmethod
    def get_llm_service(cls):
        # 延迟 import，避免循环依赖
        from llm.client import LLMService

        if cls._llm_service is None:
            cls._llm_service = LLMService()
        return cls._llm_service

    @classmethod
    def get_embedding_service(cls):
        from llm.embeddings import EmbeddingService

        if cls._embed_service is None:
            cls._embed_service = EmbeddingService()
        return cls._embed_service