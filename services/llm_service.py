import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class LLMService:
    def __init__(self, provider: Optional[str] = None):
        """
        初始化 LLM 服务。
        provider: 'openai' 或 'ollama'。如果不传，默认读取 .env。
        """
        self.provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_CHAT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0  # RAG 场景建议设置为 0，保证输出稳定性
            )
        elif self.provider == "ollama":
            self.llm = ChatOllama(
                model=os.getenv("OLLAMA_CHAT_MODEL"),
                base_url=os.getenv("OLLAMA_BASE_URL"),
                temperature=0,
                timeout=30 # 给远程调用留出足够的响应时间
            )
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

    def chat_completion(self, prompt: str, system_prompt: str = "你是一个专业的人工智能助手") -> str:
        """
        通用的对话补全函数。
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            error_msg = f"❌ [{self.provider}] LLM 调用失败: {e}"
            print(error_msg)
            return error_msg

    def stream_completion(self, prompt: str, system_prompt: str = "你是一个专业的人工智能助手"):
        """
        流式输出（适用于需要前端打字机效果的场景）。
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        return self.llm.stream(messages)

# --- 简单测试脚本 ---
if __name__ == "__main__":
    # 默认加载测试
    service = LLMService()
    print(f"当前运行模式: {service.provider}")
    
    test_prompt = "简单介绍一下什么是 RAG 架构？"
    result = service.chat_completion(test_prompt)
    print(f"回答内容:\n{result}")