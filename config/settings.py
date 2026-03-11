# config/settings.py
import os 
from dotenv import load_dotenv 
from utils .config_utils import env_bool 

load_dotenv ()


class AppConfig :
    """Global static configuration: Deep-Learner core parameter hub"""

    # === Memory and Repair Control ===
    MAX_REPAIR_LOOPS =int (os .getenv ("MAX_REPAIR_LOOPS",3 ))
    # Runtime V2 is used as the default execution path; if emergency fallback is required, it can be explicitly set to false
    RUNTIME_V2_ENABLED =env_bool ("RUNTIME_V2_ENABLED",True )
    RUNTIME_MAX_TRANSITIONS =int (os .getenv ("RUNTIME_MAX_TRANSITIONS",12 ))
    RUNTIME_ENFORCE_CONTRACT =env_bool ("RUNTIME_ENFORCE_CONTRACT",True )
    # Context clipping of Compose/Verify to avoid delay amplification caused by large contexts
    RUNTIME_COMPOSE_CONTEXT_TOP_K =int (os .getenv ("RUNTIME_COMPOSE_CONTEXT_TOP_K",8 ))
    RUNTIME_VERIFY_CONTEXT_TOP_K =int (os .getenv ("RUNTIME_VERIFY_CONTEXT_TOP_K",8 ))
    # When the judgment is SUFFICIENT based on memory alone, is it still forced to enter the retrieval link (10-K evidence Q&A is recommended to be turned on)
    RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT =env_bool (
    "RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT",
    False ,
    )
    # Verify unified scoring threshold (score-led, verify is mapped by scores)
    VERIFY_PASS_SCORE_THRESHOLD =float (os .getenv ("VERIFY_PASS_SCORE_THRESHOLD","0.68"))
    STRICT_VERIFY_PASS_SCORE_THRESHOLD =float (os .getenv ("STRICT_VERIFY_PASS_SCORE_THRESHOLD","0.75"))
    MEMORY_SUFFICIENT_SCORE_THRESHOLD =float (os .getenv ("MEMORY_SUFFICIENT_SCORE_THRESHOLD","0.70"))
    # Strict Verify (three-layer control)
    SV_WEIGHT_CITATION =float (os .getenv ("SV_WEIGHT_CITATION","0.35"))
    SV_WEIGHT_HALLUCINATION =float (os .getenv ("SV_WEIGHT_HALLUCINATION","0.25"))
    SV_WEIGHT_LOGIC =float (os .getenv ("SV_WEIGHT_LOGIC","0.20"))
    SV_WEIGHT_COMPLETENESS =float (os .getenv ("SV_WEIGHT_COMPLETENESS","0.15"))
    SV_WEIGHT_FORMAT =float (os .getenv ("SV_WEIGHT_FORMAT","0.05"))
    SV_TOTAL_THRESHOLD =float (os .getenv ("SV_TOTAL_THRESHOLD","3.5"))
    SV_CRITICAL_SCORE_FLOOR =float (os .getenv ("SV_CRITICAL_SCORE_FLOOR","1.0"))
    SV_BLOCK_UNSUPPORTED_CLAIM =env_bool ("SV_BLOCK_UNSUPPORTED_CLAIM",True )
    SV_BLOCK_CITATION_MISSING =env_bool ("SV_BLOCK_CITATION_MISSING",True )
    SV_BLOCK_CITATION_NOT_IN_CONTEXT =env_bool ("SV_BLOCK_CITATION_NOT_IN_CONTEXT",True )
    SV_ALLOW_NONEMPTY_CITATIONS_OVERRIDE_MISSING =env_bool (
    "SV_ALLOW_NONEMPTY_CITATIONS_OVERRIDE_MISSING",
    False ,
    )
    SV_NUMERIC_CITATION_MISMATCH_MIN_COUNT =int (
    os .getenv ("SV_NUMERIC_CITATION_MISMATCH_MIN_COUNT",1 )
    )
    # When there is retrieval evidence, compose is prohibited from directly outputting "uncertain"
    RUNTIME_FORCE_ANSWER_ON_EVIDENCE =env_bool ("RUNTIME_FORCE_ANSWER_ON_EVIDENCE",True )
    # After the repair stage fails, priority will be given to extractive answers to reduce logical expansion errors.
    RUNTIME_REPAIR_EXTRACTIVE_FALLBACK =env_bool ("RUNTIME_REPAIR_EXTRACTIVE_FALLBACK",True )
    # Additional retries on memory draft node JSON/format failure (total attempts = 1 + this value)
    RUNTIME_MEMORY_DRAFT_MAX_RETRIES =int (os .getenv ("RUNTIME_MEMORY_DRAFT_MAX_RETRIES",1 ))
    # Before the memory is determined to be SUFFICIENT, whether it must undergo strict verification (if it fails, retrieval will be forced)
    RUNTIME_MEMORY_STRICT_GATE_ENABLED =env_bool (
    "RUNTIME_MEMORY_STRICT_GATE_ENABLED",
    True ,
    )
    # Evidence table (evidence_table) extraction switch: turned off by default to avoid structured extraction noise affecting generation and verification
    RUNTIME_ENABLE_EVIDENCE_TABLE =env_bool ("RUNTIME_ENABLE_EVIDENCE_TABLE",False )
    # Compose whether to use all search results as context (not crop according to top-k)
    RUNTIME_COMPOSE_INCLUDE_ALL_RETRIEVED_CONTEXT =env_bool (
    "RUNTIME_COMPOSE_INCLUDE_ALL_RETRIEVED_CONTEXT",
    False ,
    )

    # Steps log truncation to avoid long session expansion
    MAX_STEPS_LOG =int (os .getenv ("MAX_STEPS_LOG",200 ))

    # === 1. Infrastructure address ===
    ES_HOST =os .getenv ("ES_HOST","localhost")
    ES_PORT =os .getenv ("ES_PORT","9200")

    MILVUS_HOST =os .getenv ("MILVUS_HOST","localhost")
    MILVUS_PORT =os .getenv ("MILVUS_PORT","19530")

    # === 2. Model Provider configuration ===
    LLM_PROVIDER =os .getenv ("DEFAULT_LLM_PROVIDER","openai")
    EMBEDDING_PROVIDER =os .getenv ("DEFAULT_EMBEDDING_PROVIDER","openai")
    LLM_ROUTING_ENABLED =env_bool ("LLM_ROUTING_ENABLED",True )
    # Not mandatory when empty; after setting, all tasks will be routed to the specified provider (openai/anthropic/gemini/ollama)
    LLM_FORCE_PROVIDER =os .getenv ("LLM_FORCE_PROVIDER","").strip ().lower ()

    # OpenAI
    OPENAI_API_KEY =os .getenv ("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL =os .getenv ("OPENAI_CHAT_MODEL","gpt-4-turbo")
    OPENAI_COMPOSE_MODEL =os .getenv ("OPENAI_COMPOSE_MODEL",OPENAI_CHAT_MODEL )
    OPENAI_VERIFY_MODEL =os .getenv ("OPENAI_VERIFY_MODEL","gpt-4o-mini")
    OPENAI_REWRITE_MODEL =os .getenv ("OPENAI_REWRITE_MODEL","gpt-4o-mini")
    OPENAI_MEMORY_MODEL =os .getenv ("OPENAI_MEMORY_MODEL",OPENAI_VERIFY_MODEL )
    OPENAI_EMBED_MODEL =os .getenv (
    "OPENAI_EMBEDDING_MODEL","text-embedding-3-small"
    )
    OPENAI_TEMPERATURE =float (os .getenv ("OPENAI_TEMPERATURE","0"))
    OPENAI_TIMEOUT_SECONDS =int (os .getenv ("OPENAI_TIMEOUT_SECONDS",60 ))
    OPENAI_MAX_RETRIES =int (os .getenv ("OPENAI_MAX_RETRIES",2 ))

    # Anthropic
    ANTHROPIC_API_KEY =os .getenv ("ANTHROPIC_API_KEY")
    ANTHROPIC_COMPOSE_MODEL =os .getenv (
    "ANTHROPIC_COMPOSE_MODEL",
    "claude-sonnet-4-20250514",
    )
    ANTHROPIC_CHAT_MODEL =os .getenv ("ANTHROPIC_CHAT_MODEL",ANTHROPIC_COMPOSE_MODEL )
    ANTHROPIC_VERIFY_MODEL =os .getenv ("ANTHROPIC_VERIFY_MODEL",ANTHROPIC_COMPOSE_MODEL )
    ANTHROPIC_REWRITE_MODEL =os .getenv ("ANTHROPIC_REWRITE_MODEL",ANTHROPIC_COMPOSE_MODEL )
    ANTHROPIC_MEMORY_MODEL =os .getenv (
    "ANTHROPIC_MEMORY_MODEL",
    "claude-3-haiku-20240307",
    )
    ANTHROPIC_TEMPERATURE =float (os .getenv ("ANTHROPIC_TEMPERATURE","0"))
    ANTHROPIC_TIMEOUT_SECONDS =int (os .getenv ("ANTHROPIC_TIMEOUT_SECONDS",60 ))
    ANTHROPIC_MAX_RETRIES =int (os .getenv ("ANTHROPIC_MAX_RETRIES",2 ))

    # Gemini
    GEMINI_API_KEY =os .getenv ("GEMINI_API_KEY")or os .getenv ("GOOGLE_API_KEY")
    GEMINI_CHAT_MODEL =os .getenv ("GEMINI_CHAT_MODEL","gemini-2.5-flash")
    GEMINI_COMPOSE_MODEL =os .getenv ("GEMINI_COMPOSE_MODEL",GEMINI_CHAT_MODEL )
    GEMINI_VERIFY_MODEL =os .getenv ("GEMINI_VERIFY_MODEL",GEMINI_CHAT_MODEL )
    GEMINI_REWRITE_MODEL =os .getenv ("GEMINI_REWRITE_MODEL",GEMINI_CHAT_MODEL )
    GEMINI_MEMORY_MODEL =os .getenv ("GEMINI_MEMORY_MODEL",GEMINI_CHAT_MODEL )
    GEMINI_TEMPERATURE =float (os .getenv ("GEMINI_TEMPERATURE","0"))
    GEMINI_TIMEOUT_SECONDS =int (os .getenv ("GEMINI_TIMEOUT_SECONDS",60 ))
    GEMINI_MAX_RETRIES =int (os .getenv ("GEMINI_MAX_RETRIES",0 ))
    # The free tier has a common 5 RPM limit; the default is 12.5s throttling to avoid 429 triggers.
    GEMINI_MIN_CALL_INTERVAL_SECONDS =float (
    os .getenv ("GEMINI_MIN_CALL_INTERVAL_SECONDS","12.5")
    )

    # Ollama
    OLLAMA_BASE_URL =os .getenv ("OLLAMA_BASE_URL","http://localhost:11434")
    OLLAMA_CHAT_MODEL =os .getenv ("OLLAMA_CHAT_MODEL","llama3")
    OLLAMA_EMBED_MODEL =os .getenv ("OLLAMA_EMBEDDING_MODEL","mxbai-embed-large")
    OLLAMA_TEMPERATURE =float (os .getenv ("OLLAMA_TEMPERATURE","0"))
    OLLAMA_TIMEOUT_SECONDS =int (os .getenv ("OLLAMA_TIMEOUT_SECONDS",60 ))

    # Unstructured parser
    UNSTRUCTURED_STRATEGY =os .getenv ("UNSTRUCTURED_STRATEGY","fast")
    UNSTRUCTURED_LANGUAGES =[
    lang .strip ()
    for lang in os .getenv ("UNSTRUCTURED_LANGUAGES","chi_sim,eng").split (",")
    if lang .strip ()
    ]
    UNSTRUCTURED_INFER_TABLE_STRUCTURE =env_bool (
    "UNSTRUCTURED_INFER_TABLE_STRUCTURE",False 
    )
    PDF_EXTRACT_TABLES_WITH_PDFPLUMBER =env_bool (
    "PDF_EXTRACT_TABLES_WITH_PDFPLUMBER",True 
    )
    # 0 means no limit on the number of pages; the table range can be narrowed when used for debugging/stress testing
    PDF_PDFPLUMBER_MAX_PAGES =int (os .getenv ("PDF_PDFPLUMBER_MAX_PAGES",0 ))
    PDFPLUMBER_VERTICAL_STRATEGY =os .getenv (
    "PDFPLUMBER_VERTICAL_STRATEGY","text"
    )
    PDFPLUMBER_HORIZONTAL_STRATEGY =os .getenv (
    "PDFPLUMBER_HORIZONTAL_STRATEGY","text"
    )

    # === 3. RAG / Retrieval parameters ===
    ES_INDEX =os .getenv ("ES_INDEX_NAME","deep_learner_knowledge")
    MILVUS_COLLECTION =os .getenv (
    "MILVUS_COLLECTION_NAME","deep_learner_vectors"
    )

    EMBEDDING_DIM =int (os .getenv ("EMBEDDING_DIM",1536 ))
    CHUNK_SIZE =int (os .getenv ("CHUNK_SIZE",2200 ))
    TABLE_CHUNK_WINDOW_ROWS =int (os .getenv ("TABLE_CHUNK_WINDOW_ROWS",10 ))
    TABLE_CHUNK_EMIT_WINDOW =env_bool ("TABLE_CHUNK_EMIT_WINDOW",True )
    TABLE_CHUNK_EMIT_ROW_FACTS =env_bool ("TABLE_CHUNK_EMIT_ROW_FACTS",True )
    TABLE_CHUNK_EMIT_RAW =env_bool ("TABLE_CHUNK_EMIT_RAW",True )
    # When true, only the row-fact text chunk will be retained first, and the original/window table structure will not be retained.
    TABLE_CHUNK_ROW_FACT_ONLY =env_bool ("TABLE_CHUNK_ROW_FACT_ONLY",False )
    TABLE_CHUNK_MAX_ROW_FACTS =int (os .getenv ("TABLE_CHUNK_MAX_ROW_FACTS",160 ))
    # Milvus index/search knobs
    MILVUS_INDEX_METRIC_TYPE =os .getenv ("MILVUS_INDEX_METRIC_TYPE","COSINE")
    MILVUS_INDEX_TYPE =os .getenv ("MILVUS_INDEX_TYPE","HNSW")
    MILVUS_INDEX_M =int (os .getenv ("MILVUS_INDEX_M",8 ))
    MILVUS_INDEX_EF_CONSTRUCTION =int (
    os .getenv ("MILVUS_INDEX_EF_CONSTRUCTION",64 )
    )
    MILVUS_SEARCH_METRIC_TYPE =os .getenv ("MILVUS_SEARCH_METRIC_TYPE","COSINE")
    MILVUS_SEARCH_NPROBE =int (os .getenv ("MILVUS_SEARCH_NPROBE",10 ))
    MILVUS_LTM_SEARCH_METRIC_TYPE =os .getenv (
    "MILVUS_LTM_SEARCH_METRIC_TYPE","COSINE"
    )
    MILVUS_LTM_SEARCH_EF =int (os .getenv ("MILVUS_LTM_SEARCH_EF",64 ))

    # Retrieval pipeline
    RRF_K =int (os .getenv ("RRF_K",30 ))
    RECALL_TOP_K =int (os .getenv ("RECALL_TOP_K",50 ))
    FINAL_TOP_K =int (os .getenv ("FINAL_TOP_K",8 ))
    # Only allowed to search in the specified document collection (recommended to be turned on in the 10-K scenario)
    RETRIEVAL_RESTRICT_TO_ALLOWED_SOURCES =env_bool (
    "RETRIEVAL_RESTRICT_TO_ALLOWED_SOURCES",
    True ,
    )
    RETRIEVAL_ALLOWED_SOURCES =tuple (
    s .strip ().lower ()
    for s in os .getenv (
    "RETRIEVAL_ALLOWED_SOURCES",
    "Amazon 10K 2024.pdf,Alphabet 10K 2024.pdf,MSFT 10-K.pdf",
    ).split (",")
    if s .strip ()
    )
    # Runtime V2: Two-stage retrieval parameters
    PHASE1_VECTOR_TOP_K =int (os .getenv ("PHASE1_VECTOR_TOP_K",50 ))
    PHASE1_KEYWORD_TOP_K =int (os .getenv ("PHASE1_KEYWORD_TOP_K",70 ))
    PHASE1_RERANK_TOP_N =int (os .getenv ("PHASE1_RERANK_TOP_N",28 ))
    # multi-query retrieval (MVP): decompose comparative question into several focused queries
    RETRIEVAL_MULTI_QUERY_ENABLED =env_bool ("RETRIEVAL_MULTI_QUERY_ENABLED",False )
    RETRIEVAL_MULTI_QUERY_MAX_QUERIES =int (os .getenv ("RETRIEVAL_MULTI_QUERY_MAX_QUERIES",4 ))
    RETRIEVAL_MULTI_QUERY_PARALLEL_WORKERS =int (
    os .getenv ("RETRIEVAL_MULTI_QUERY_PARALLEL_WORKERS",4 )
    )
    PHASE1_PER_QUERY_VECTOR_TOP_K =int (os .getenv ("PHASE1_PER_QUERY_VECTOR_TOP_K",80 ))
    PHASE1_PER_QUERY_KEYWORD_TOP_K =int (os .getenv ("PHASE1_PER_QUERY_KEYWORD_TOP_K",100 ))
    PHASE1_PER_QUERY_KEEP_TOP_M =int (os .getenv ("PHASE1_PER_QUERY_KEEP_TOP_M",90 ))
    RETRIEVAL_MULTI_QUERY_COVERAGE_WEIGHT =float (
    os .getenv ("RETRIEVAL_MULTI_QUERY_COVERAGE_WEIGHT","0.18")
    )
    RETRIEVAL_MULTI_QUERY_RERANK_COVERAGE_WEIGHT =float (
    os .getenv ("RETRIEVAL_MULTI_QUERY_RERANK_COVERAGE_WEIGHT","0.16")
    )
    RETRIEVAL_MULTI_QUERY_RERANK_PRIOR_WEIGHT =float (
    os .getenv ("RETRIEVAL_MULTI_QUERY_RERANK_PRIOR_WEIGHT","0.80")
    )
    RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WEIGHT =float (
    os .getenv ("RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WEIGHT","0.38")
    )
    RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WINDOW =int (
    os .getenv ("RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WINDOW",24 )
    )
    RETRIEVAL_MULTI_QUERY_ROUTE_TOP_K =int (
    os .getenv ("RETRIEVAL_MULTI_QUERY_ROUTE_TOP_K",24 )
    )
    RETRIEVAL_ROUTE_ENTITY_MATCH_BONUS =float (
    os .getenv ("RETRIEVAL_ROUTE_ENTITY_MATCH_BONUS","0.10")
    )
    RETRIEVAL_ROUTE_ENTITY_MISMATCH_PENALTY =float (
    os .getenv ("RETRIEVAL_ROUTE_ENTITY_MISMATCH_PENALTY","0.42")
    )
    RETRIEVAL_ROUTE_METRIC_SCOPE_PENALTY =float (
    os .getenv ("RETRIEVAL_ROUTE_METRIC_SCOPE_PENALTY","0.16")
    )
    # The number of candidates reserved for each subquery in compose context (recommended 2~3, default 3)
    RETRIEVAL_MULTI_QUERY_CONTEXT_TOP_K =int (
    os .getenv ("RETRIEVAL_MULTI_QUERY_CONTEXT_TOP_K",8 )
    )
    PHASE2_VECTOR_TOP_K =int (os .getenv ("PHASE2_VECTOR_TOP_K",120 ))
    PHASE2_KEYWORD_TOP_K =int (os .getenv ("PHASE2_KEYWORD_TOP_K",140 ))
    PHASE2_RERANK_TOP_N =int (os .getenv ("PHASE2_RERANK_TOP_N",50 ))
    PHASE2_PER_QUERY_VECTOR_TOP_K =int (
    os .getenv (
    "PHASE2_PER_QUERY_VECTOR_TOP_K",
    str (PHASE1_PER_QUERY_VECTOR_TOP_K ),
    )
    )
    PHASE2_PER_QUERY_KEYWORD_TOP_K =int (
    os .getenv (
    "PHASE2_PER_QUERY_KEYWORD_TOP_K",
    str (PHASE1_PER_QUERY_KEYWORD_TOP_K ),
    )
    )
    PHASE2_PER_QUERY_KEEP_TOP_M =int (
    os .getenv (
    "PHASE2_PER_QUERY_KEEP_TOP_M",
    str (PHASE1_PER_QUERY_KEEP_TOP_M ),
    )
    )
    RETRIEVAL_EXPAND_GENERIC_MULTI_COMPANY =env_bool (
    "RETRIEVAL_EXPAND_GENERIC_MULTI_COMPANY",
    False ,
    )
    # High recall mode: Complete source coverage after rerank to reduce missed recalls caused by "single document crowding"
    RETRIEVAL_ENFORCE_SOURCE_COVERAGE =env_bool (
    "RETRIEVAL_ENFORCE_SOURCE_COVERAGE",
    False ,
    )
    RETRIEVAL_MIN_SOURCE_COVERAGE =int (
    os .getenv ("RETRIEVAL_MIN_SOURCE_COVERAGE",0 )
    )
    # source overrides the constraint effective window; 0 means following compose top-k
    RETRIEVAL_SOURCE_COVERAGE_WINDOW_TOP_K =int (
    os .getenv ("RETRIEVAL_SOURCE_COVERAGE_WINDOW_TOP_K",0 )
    )
    # Quality gating (effective after Rerank)
    RETRIEVAL_SCORE_GATE_ENABLED =env_bool (
    "RETRIEVAL_SCORE_GATE_ENABLED",True 
    )
    RETRIEVAL_MIN_RERANK_SCORE =float (
    os .getenv ("RETRIEVAL_MIN_RERANK_SCORE","0.10")
    )

    # Reranker
    RERANK_MODEL_PATH =os .getenv ("RERANK_MODEL_PATH")
    RERANK_DEVICE =os .getenv ("RERANK_DEVICE","cpu")

    # === 4. Long Term Memory (LTM) ===
    MILVUS_LTM_COLLECTION =os .getenv (
    "MILVUS_LTM_COLLECTION","user_long_term_memory"
    )

    LTM_SEARCH_PREFIX =os .getenv ("LTM_SEARCH_PREFIX","search_query:")
    LTM_RECALL_THRESHOLD =float (os .getenv ("LTM_RECALL_THRESHOLD","0.4"))
    LTM_RECALL_TOP_K =int (os .getenv ("LTM_RECALL_TOP_K",5 ))
    LTM_IMPORTANCE_THRESHOLD =float (
    os .getenv ("LTM_IMPORTANCE_THRESHOLD","0.7")
    )
    # Pause LTM writes (recall only, no new memory written)
    LTM_WRITE_ENABLED =env_bool ("LTM_WRITE_ENABLED",False )

    # === 5. Short-term memory (STM) ===
    MEMORY_N_CHUNK =int (os .getenv ("MEMORY_N_CHUNK",5 ))
    MEMORY_K_BUFFER =int (os .getenv ("MEMORY_K_BUFFER",3 ))

    # chunk overlap is in "line" units; compatible with the old variable CHUNK_OVERLAP_LINES
    CHUNK_OVERLAP =int (
    os .getenv ("CHUNK_OVERLAP",os .getenv ("CHUNK_OVERLAP_LINES",6 ))
    )

    STM_COMPRESS_INTERVAL =int (os .getenv ("STM_COMPRESS_INTERVAL",5 ))
    STM_RECENT_WINDOW =int (os .getenv ("STM_RECENT_WINDOW",3 ))
    STM_COMPRESS_BATCH =int (os .getenv ("STM_COMPRESS_BATCH",5 ))
