from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import AppConfig

class LLMService:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or AppConfig.LLM_PROVIDER
        
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=AppConfig.OPENAI_CHAT_MODEL,
                api_key=AppConfig.OPENAI_API_KEY,
                temperature=0 
            )
        elif self.provider == "ollama":
            self.llm = ChatOllama(
                model=AppConfig.OLLAMA_CHAT_MODEL,
                base_url=AppConfig.OLLAMA_BASE_URL,
                temperature=0,
                timeout=60 # 给远程 Windows 调用留足时间
            )

    def chat_completion(self, prompt: str, system_prompt: str = "你是一个专业的 Deep-Learner 导师") -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"❌ 调用失败: {str(e)}"