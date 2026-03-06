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
    # 当仅凭记忆判为 SUFFICIENT 时，是否仍强制进入检索链路（10-K 证据问答建议开启）
    RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT = env_bool(
        "RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT",
        False,
    )
    # Verify 统一评分阈值（评分主导，verdict 由分数映射）
    VERIFY_PASS_SCORE_THRESHOLD = float(os.getenv("VERIFY_PASS_SCORE_THRESHOLD", "0.68"))
    STRICT_VERIFY_PASS_SCORE_THRESHOLD = float(os.getenv("STRICT_VERIFY_PASS_SCORE_THRESHOLD", "0.75"))
    MEMORY_SUFFICIENT_SCORE_THRESHOLD = float(os.getenv("MEMORY_SUFFICIENT_SCORE_THRESHOLD", "0.70"))
    # Strict Verify（三层控制）
    SV_WEIGHT_CITATION = float(os.getenv("SV_WEIGHT_CITATION", "0.35"))
    SV_WEIGHT_HALLUCINATION = float(os.getenv("SV_WEIGHT_HALLUCINATION", "0.25"))
    SV_WEIGHT_LOGIC = float(os.getenv("SV_WEIGHT_LOGIC", "0.20"))
    SV_WEIGHT_COMPLETENESS = float(os.getenv("SV_WEIGHT_COMPLETENESS", "0.15"))
    SV_WEIGHT_FORMAT = float(os.getenv("SV_WEIGHT_FORMAT", "0.05"))
    SV_TOTAL_THRESHOLD = float(os.getenv("SV_TOTAL_THRESHOLD", "3.5"))
    SV_CRITICAL_SCORE_FLOOR = float(os.getenv("SV_CRITICAL_SCORE_FLOOR", "1.0"))
    SV_BLOCK_UNSUPPORTED_CLAIM = env_bool("SV_BLOCK_UNSUPPORTED_CLAIM", True)
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
    LLM_ROUTING_ENABLED = env_bool("LLM_ROUTING_ENABLED", True)

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
    OPENAI_COMPOSE_MODEL = os.getenv("OPENAI_COMPOSE_MODEL", OPENAI_CHAT_MODEL)
    OPENAI_VERIFY_MODEL = os.getenv("OPENAI_VERIFY_MODEL", "gpt-4o-mini")
    OPENAI_REWRITE_MODEL = os.getenv("OPENAI_REWRITE_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", 60))
    OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", 2))

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_COMPOSE_MODEL = os.getenv(
        "ANTHROPIC_COMPOSE_MODEL",
        "claude-sonnet-4-20250514",
    )
    ANTHROPIC_MEMORY_MODEL = os.getenv(
        "ANTHROPIC_MEMORY_MODEL",
        "claude-3-haiku-20240307",
    )
    ANTHROPIC_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0"))
    ANTHROPIC_TIMEOUT_SECONDS = int(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", 60))
    ANTHROPIC_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", 2))

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", 60))

    # Unstructured parser
    UNSTRUCTURED_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "fast")
    UNSTRUCTURED_LANGUAGES = [
        lang.strip()
        for lang in os.getenv("UNSTRUCTURED_LANGUAGES", "chi_sim,eng").split(",")
        if lang.strip()
    ]
    UNSTRUCTURED_INFER_TABLE_STRUCTURE = env_bool(
        "UNSTRUCTURED_INFER_TABLE_STRUCTURE", False
    )
    PDF_EXTRACT_TABLES_WITH_PDFPLUMBER = env_bool(
        "PDF_EXTRACT_TABLES_WITH_PDFPLUMBER", True
    )
    # 0 表示不限制页数；用于调试/压测时可缩小抽表范围
    PDF_PDFPLUMBER_MAX_PAGES = int(os.getenv("PDF_PDFPLUMBER_MAX_PAGES", 0))
    PDFPLUMBER_VERTICAL_STRATEGY = os.getenv(
        "PDFPLUMBER_VERTICAL_STRATEGY", "text"
    )
    PDFPLUMBER_HORIZONTAL_STRATEGY = os.getenv(
        "PDFPLUMBER_HORIZONTAL_STRATEGY", "text"
    )

    # === 3. RAG / 检索参数 ===
    ES_INDEX = os.getenv("ES_INDEX_NAME", "deep_learner_knowledge")
    MILVUS_COLLECTION = os.getenv(
        "MILVUS_COLLECTION_NAME", "deep_learner_vectors"
    )

    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2200))
    # Milvus index/search knobs
    MILVUS_INDEX_METRIC_TYPE = os.getenv("MILVUS_INDEX_METRIC_TYPE", "COSINE")
    MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
    MILVUS_INDEX_M = int(os.getenv("MILVUS_INDEX_M", 8))
    MILVUS_INDEX_EF_CONSTRUCTION = int(
        os.getenv("MILVUS_INDEX_EF_CONSTRUCTION", 64)
    )
    MILVUS_SEARCH_METRIC_TYPE = os.getenv("MILVUS_SEARCH_METRIC_TYPE", "COSINE")
    MILVUS_SEARCH_NPROBE = int(os.getenv("MILVUS_SEARCH_NPROBE", 10))
    MILVUS_LTM_SEARCH_METRIC_TYPE = os.getenv(
        "MILVUS_LTM_SEARCH_METRIC_TYPE", "COSINE"
    )
    MILVUS_LTM_SEARCH_EF = int(os.getenv("MILVUS_LTM_SEARCH_EF", 64))

    # 检索流水线
    RRF_K = int(os.getenv("RRF_K", 30))
    RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", 50))
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 8))
    # 仅允许在指定文档集合中检索（10-K 场景建议开启）
    RETRIEVAL_RESTRICT_TO_ALLOWED_SOURCES = env_bool(
        "RETRIEVAL_RESTRICT_TO_ALLOWED_SOURCES",
        True,
    )
    RETRIEVAL_ALLOWED_SOURCES = tuple(
        s.strip().lower()
        for s in os.getenv(
            "RETRIEVAL_ALLOWED_SOURCES",
            "Amazon 10K 2024.pdf,Alphabet 10K 2024.pdf,MSFT 10-K.pdf",
        ).split(",")
        if s.strip()
    )
    # Runtime V2: 两阶段检索参数
    PHASE1_VECTOR_TOP_K = int(os.getenv("PHASE1_VECTOR_TOP_K", 50))
    PHASE1_KEYWORD_TOP_K = int(os.getenv("PHASE1_KEYWORD_TOP_K", 70))
    PHASE1_RERANK_TOP_N = int(os.getenv("PHASE1_RERANK_TOP_N", 28))
    PHASE2_VECTOR_TOP_K = int(os.getenv("PHASE2_VECTOR_TOP_K", 120))
    PHASE2_KEYWORD_TOP_K = int(os.getenv("PHASE2_KEYWORD_TOP_K", 140))
    PHASE2_RERANK_TOP_N = int(os.getenv("PHASE2_RERANK_TOP_N", 50))
    RETRIEVAL_EXPAND_GENERIC_MULTI_COMPANY = env_bool(
        "RETRIEVAL_EXPAND_GENERIC_MULTI_COMPANY",
        False,
    )
    # 高召回模式：在 rerank 后补齐来源覆盖，降低“单文档挤占”导致的漏召回
    RETRIEVAL_ENFORCE_SOURCE_COVERAGE = env_bool(
        "RETRIEVAL_ENFORCE_SOURCE_COVERAGE",
        False,
    )
    RETRIEVAL_MIN_SOURCE_COVERAGE = int(
        os.getenv("RETRIEVAL_MIN_SOURCE_COVERAGE", 0)
    )
    # source 覆盖约束生效窗口；0 表示跟随 compose top-k
    RETRIEVAL_SOURCE_COVERAGE_WINDOW_TOP_K = int(
        os.getenv("RETRIEVAL_SOURCE_COVERAGE_WINDOW_TOP_K", 0)
    )
    # 质量门控（Rerank 后生效）
    RETRIEVAL_SCORE_GATE_ENABLED = env_bool(
        "RETRIEVAL_SCORE_GATE_ENABLED", True
    )
    RETRIEVAL_MIN_RERANK_SCORE = float(
        os.getenv("RETRIEVAL_MIN_RERANK_SCORE", "0.10")
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
        os.getenv("CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP_LINES", 6))
    )
    
    STM_COMPRESS_INTERVAL = int(os.getenv("STM_COMPRESS_INTERVAL", 5))
    STM_RECENT_WINDOW = int(os.getenv("STM_RECENT_WINDOW", 3))
    STM_COMPRESS_BATCH = int(os.getenv("STM_COMPRESS_BATCH", 5))
