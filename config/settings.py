# config/settings.py
import os
from dotenv import load_dotenv
from utils.config_utils import env_bool

load_dotenv()


class AppConfig:
    """全局静态配置：Deep-Learner 核心参数枢纽"""

    # === 记忆与修复控制 ===
    MAX_REPAIR_LOOPS = int(os.getenv("MAX_REPAIR_LOOPS", 3))
    # Runtime V2 作为默认执行路径；如需紧急回退可显式设置为 false
    RUNTIME_V2_ENABLED = env_bool("RUNTIME_V2_ENABLED", True)
    RUNTIME_MAX_TRANSITIONS = int(os.getenv("RUNTIME_MAX_TRANSITIONS", 12))
    RUNTIME_ENFORCE_CONTRACT = env_bool("RUNTIME_ENFORCE_CONTRACT", True)
    # Compose/Verify 的上下文裁剪，避免大上下文导致延迟放大
    RUNTIME_COMPOSE_CONTEXT_TOP_K = int(os.getenv("RUNTIME_COMPOSE_CONTEXT_TOP_K", 8))
    RUNTIME_VERIFY_CONTEXT_TOP_K = int(os.getenv("RUNTIME_VERIFY_CONTEXT_TOP_K", 8))
    # Verify 统一评分阈值（评分主导，verdict 由分数映射）
    VERIFY_PASS_SCORE_THRESHOLD = float(os.getenv("VERIFY_PASS_SCORE_THRESHOLD", "0.68"))
    STRICT_VERIFY_PASS_SCORE_THRESHOLD = float(os.getenv("STRICT_VERIFY_PASS_SCORE_THRESHOLD", "0.75"))
    MEMORY_SUFFICIENT_SCORE_THRESHOLD = float(os.getenv("MEMORY_SUFFICIENT_SCORE_THRESHOLD", "0.70"))
    # 当已有检索证据时，禁止 compose 直接输出“不确定”
    RUNTIME_FORCE_ANSWER_ON_EVIDENCE = env_bool("RUNTIME_FORCE_ANSWER_ON_EVIDENCE", True)
    # repair 阶段失败后，优先转为抽取式回答以降低逻辑扩写错误
    RUNTIME_REPAIR_EXTRACTIVE_FALLBACK = env_bool("RUNTIME_REPAIR_EXTRACTIVE_FALLBACK", True)

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
    # Runtime V2: 两阶段检索参数
    PHASE1_VECTOR_TOP_K = int(os.getenv("PHASE1_VECTOR_TOP_K", 30))
    PHASE1_KEYWORD_TOP_K = int(os.getenv("PHASE1_KEYWORD_TOP_K", 30))
    PHASE1_RERANK_TOP_N = int(os.getenv("PHASE1_RERANK_TOP_N", 20))
    PHASE2_VECTOR_TOP_K = int(os.getenv("PHASE2_VECTOR_TOP_K", 80))
    PHASE2_KEYWORD_TOP_K = int(os.getenv("PHASE2_KEYWORD_TOP_K", 80))
    PHASE2_RERANK_TOP_N = int(os.getenv("PHASE2_RERANK_TOP_N", 40))
    # 质量门控（Rerank 后生效）
    RETRIEVAL_SCORE_GATE_ENABLED = env_bool(
        "RETRIEVAL_SCORE_GATE_ENABLED", True
    )
    RETRIEVAL_MIN_RERANK_SCORE = float(
        os.getenv("RETRIEVAL_MIN_RERANK_SCORE", "0.05")
    )

    # Reranker
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH")
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")

    # === 4. 长期记忆 (LTM) ===
    MILVUS_LTM_COLLECTION = os.getenv(
        "MILVUS_LTM_COLLECTION", "user_long_term_memory"
    )

    LTM_SEARCH_PREFIX = os.getenv("LTM_SEARCH_PREFIX", "search_query:")
    LTM_RECALL_THRESHOLD = float(os.getenv("LTM_RECALL_THRESHOLD", "0.4"))
    LTM_RECALL_TOP_K = int(os.getenv("LTM_RECALL_TOP_K", 5))
    LTM_IMPORTANCE_THRESHOLD = float(
        os.getenv("LTM_IMPORTANCE_THRESHOLD", "0.7")
    )

    # === 5. 短期记忆 (STM) ===
    MEMORY_N_CHUNK = int(os.getenv("MEMORY_N_CHUNK", 5))
    MEMORY_K_BUFFER = int(os.getenv("MEMORY_K_BUFFER", 3))

    # chunk overlap 以“行”为单位；兼容旧变量 CHUNK_OVERLAP_LINES
    CHUNK_OVERLAP = int(
        os.getenv("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP_LINES", 50))
    )
    
    STM_COMPRESS_INTERVAL = int(os.getenv("STM_COMPRESS_INTERVAL", 5))
    STM_RECENT_WINDOW = int(os.getenv("STM_RECENT_WINDOW", 3))
    STM_COMPRESS_BATCH = int(os.getenv("STM_COMPRESS_BATCH", 5))
