# llm/client.py
import time 
from typing import Any ,Optional 

from langchain_openai import ChatOpenAI 
from langchain_ollama import ChatOllama 

try :
    from langchain_anthropic import ChatAnthropic 
except Exception :# pragma: no cover
    ChatAnthropic =None 

try :
    from langchain_google_genai import ChatGoogleGenerativeAI 
except Exception :# pragma: no cover
    ChatGoogleGenerativeAI =None 

from langchain_core .messages import HumanMessage ,SystemMessage 

from config .settings import AppConfig 


class LLMService :
    """Unified encapsulation of chat models (deterministic output)"""

    def __init__ (self ,provider :Optional [str ]=None ):
        self .provider =self ._normalize_provider (provider or AppConfig .LLM_PROVIDER )
        self ._client_cache :dict [tuple [str ,str ],object ]={}
        self ._provider_last_call_at :dict [str ,float ]={}

    @staticmethod 
    def _normalize_provider (provider :str )->str :
        text =str (provider or "").strip ().lower ()
        if text =="claude":
            return "anthropic"
        return text 

    def _default_model_for_provider (self ,provider :str )->str :
        if provider =="openai":
            return AppConfig .OPENAI_CHAT_MODEL 
        if provider =="ollama":
            return AppConfig .OLLAMA_CHAT_MODEL 
        if provider =="anthropic":
            return AppConfig .ANTHROPIC_CHAT_MODEL 
        if provider =="gemini":
            return AppConfig .GEMINI_CHAT_MODEL 
        raise ValueError (f"Unsupported LLM provider: {provider}")

    def _resolve_forced_provider (self ,provider :str ,task_norm :str )->tuple [str ,str ]:
        if provider =="openai":
            model_map ={
            "compose":AppConfig .OPENAI_COMPOSE_MODEL ,
            "verify":AppConfig .OPENAI_VERIFY_MODEL ,
            "rewrite":AppConfig .OPENAI_REWRITE_MODEL ,
            "memory":AppConfig .OPENAI_MEMORY_MODEL ,
            "default":AppConfig .OPENAI_CHAT_MODEL ,
            }
            return ("openai",model_map .get (task_norm ,AppConfig .OPENAI_CHAT_MODEL ))

        if provider =="anthropic":
            model_map ={
            "compose":AppConfig .ANTHROPIC_COMPOSE_MODEL ,
            "verify":AppConfig .ANTHROPIC_VERIFY_MODEL ,
            "rewrite":AppConfig .ANTHROPIC_REWRITE_MODEL ,
            "memory":AppConfig .ANTHROPIC_MEMORY_MODEL ,
            "default":AppConfig .ANTHROPIC_CHAT_MODEL ,
            }
            return ("anthropic",model_map .get (task_norm ,AppConfig .ANTHROPIC_CHAT_MODEL ))

        if provider =="gemini":
            model_map ={
            "compose":AppConfig .GEMINI_COMPOSE_MODEL ,
            "verify":AppConfig .GEMINI_VERIFY_MODEL ,
            "rewrite":AppConfig .GEMINI_REWRITE_MODEL ,
            "memory":AppConfig .GEMINI_MEMORY_MODEL ,
            "default":AppConfig .GEMINI_CHAT_MODEL ,
            }
            return ("gemini",model_map .get (task_norm ,AppConfig .GEMINI_CHAT_MODEL ))

        if provider =="ollama":
            return ("ollama",AppConfig .OLLAMA_CHAT_MODEL )

        raise ValueError (f"Unsupported LLM_FORCE_PROVIDER: {provider}")

    def _resolve_route (self ,task :str )->tuple [str ,str ]:
        task_norm =str (task or "default").strip ().lower ()
        forced_provider =self ._normalize_provider (AppConfig .LLM_FORCE_PROVIDER )
        if forced_provider :
            return self ._resolve_forced_provider (forced_provider ,task_norm )

        if AppConfig .LLM_ROUTING_ENABLED :
            if task_norm =="compose":
                return ("openai",AppConfig .OPENAI_COMPOSE_MODEL )
            if task_norm in {"verify","rewrite"}:
                model =(
                AppConfig .OPENAI_VERIFY_MODEL 
                if task_norm =="verify"
                else AppConfig .OPENAI_REWRITE_MODEL 
                )
                return ("openai",model )
            if task_norm =="memory":
                return ("openai",AppConfig .OPENAI_MEMORY_MODEL )

        provider =self ._normalize_provider (self .provider )
        return (provider ,self ._default_model_for_provider (provider ))

    def _build_client (self ,provider :str ,model :str ):
        if provider =="openai":
            return ChatOpenAI (
            model =model ,
            api_key =AppConfig .OPENAI_API_KEY ,
            temperature =AppConfig .OPENAI_TEMPERATURE ,
            timeout =AppConfig .OPENAI_TIMEOUT_SECONDS ,
            max_retries =AppConfig .OPENAI_MAX_RETRIES ,
            )
        if provider =="ollama":
            return ChatOllama (
            model =model ,
            base_url =AppConfig .OLLAMA_BASE_URL ,
            temperature =AppConfig .OLLAMA_TEMPERATURE ,
            timeout =AppConfig .OLLAMA_TIMEOUT_SECONDS ,
            )
        if provider =="anthropic":
            if ChatAnthropic is None :
                raise ImportError (
                "langchain-anthropic is not installed; Anthropic routing is unavailable"
                )
            return ChatAnthropic (
            model =model ,
            api_key =AppConfig .ANTHROPIC_API_KEY ,
            temperature =AppConfig .ANTHROPIC_TEMPERATURE ,
            timeout =AppConfig .ANTHROPIC_TIMEOUT_SECONDS ,
            max_retries =AppConfig .ANTHROPIC_MAX_RETRIES ,
            )
        if provider =="gemini":
            if ChatGoogleGenerativeAI is None :
                raise ImportError (
                "langchain-google-genai is not installed; Gemini routing is unavailable"
                )
            if not AppConfig .GEMINI_API_KEY :
                raise ValueError ("GEMINI_API_KEY/GOOGLE_API_KEY is required for Gemini provider")
            return ChatGoogleGenerativeAI (
            model =model ,
            google_api_key =AppConfig .GEMINI_API_KEY ,
            temperature =AppConfig .GEMINI_TEMPERATURE ,
            timeout =AppConfig .GEMINI_TIMEOUT_SECONDS ,
            max_retries =AppConfig .GEMINI_MAX_RETRIES ,
            )
        raise ValueError (f"Unsupported LLM provider: {provider}")

    def _get_client (self ,provider :str ,model :str ):
        key =(provider ,model )
        client =self ._client_cache .get (key )
        if client is None :
            client =self ._build_client (provider ,model )
            self ._client_cache [key ]=client 
        return client 

    def _maybe_rate_limit_provider (self ,provider :str )->None :
        if provider !="gemini":
            return 

        min_interval =float (AppConfig .GEMINI_MIN_CALL_INTERVAL_SECONDS )
        if min_interval <=0 :
            return 

        now =time .monotonic ()
        last =self ._provider_last_call_at .get (provider ,0.0 )
        wait_s =min_interval -(now -last )
        if wait_s >0 :
            time .sleep (wait_s )

    def chat_completion_with_meta (
    self ,
    prompt :str ,
    system_prompt :str ,
    task :str ="default",
    )->dict [str ,Any ]:
        provider ,model =self ._resolve_route (task )
        llm =self ._get_client (provider ,model )
        self ._maybe_rate_limit_provider (provider )
        self ._provider_last_call_at [provider ]=time .monotonic ()
        messages =[
        SystemMessage (content =system_prompt ),
        HumanMessage (content =prompt ),
        ]
        start =time .perf_counter ()
        response =llm .invoke (messages )
        latency_ms =(time .perf_counter ()-start )*1000.0 
        content =response .content if hasattr (response ,"content")else str (response )
        return {
        "content":str (content ),
        "provider":provider ,
        "model":model ,
        "task":str (task or "default"),
        "latency_ms":float (latency_ms ),
        }

    def chat_completion (
    self ,
    prompt :str ,
    system_prompt :str ,
    task :str ="default",
    )->str :
        payload =self .chat_completion_with_meta (
        prompt =prompt ,
        system_prompt =system_prompt ,
        task =task ,
        )
        return str (payload ["content"])
