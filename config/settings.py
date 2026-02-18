# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    """全局静态配置：Deep-Learner 核心参数枢纽"""

    # === 记忆与修复控制 ===
    MAX_REPAIR_LOOPS = int(os.getenv("MAX_REPAIR_LOOPS", 3))

        # Steps log 截断，避免长会话膨胀
    MAX_STEPS_LOG = int(os.getenv("MAX_STEPS_LOG", 200))

# === 1. 基础设施地址 ===
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = os.getenv("ES_PORT", "9200")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    # === 2. 模型 Provider 配置 ===
    LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
    OPENAI_EMBED_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

    # === 3. RAG / 检索参数 ===
    ES_INDEX = os.getenv("ES_INDEX_NAME", "deep_learner_knowledge")
    MILVUS_COLLECTION = os.getenv(
        "MILVUS_COLLECTION_NAME", "deep_learner_vectors"
    )

    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))

    # 检索流水线
    RRF_K = int(os.getenv("RRF_K", 60))
    RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", 50))
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 5))

    # Reranker
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH")
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")

    # === 4. 长期记忆 (LTM) ===
    MILVUS_LTM_COLLECTION = os.getenv(
        "MILVUS_LTM_COLLECTION", "user_long_term_memory"
    )

    LTM_SEARCH_PREFIX = os.getenv("LTM_SEARCH_PREFIX", "search_query: ")
    LTM_RECALL_THRESHOLD = float(os.getenv("LTM_RECALL_THRESHOLD", 0.4))
    LTM_RECALL_TOP_K = int(os.getenv("LTM_RECALL_TOP_K", 5))
    LTM_IMPORTANCE_THRESHOLD = int(
        os.getenv("LTM_IMPORTANCE_THRESHOLD", 7)
    )

    # === 5. 短期记忆 (STM) ===
    MEMORY_N_CHUNK = int(os.getenv("MEMORY_N_CHUNK", 5))
    MEMORY_K_BUFFER = int(os.getenv("MEMORY_K_BUFFER", 3))

    # config/settings.py
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    STM_COMPRESS_INTERVAL = int(os.getenv("STM_COMPRESS_INTERVAL", 5))
    STM_RECENT_WINDOW = int(os.getenv("STM_RECENT_WINDOW", 3))
    STM_COMPRESS_BATCH = int(os.getenv("STM_COMPRESS_BATCH", 5))
