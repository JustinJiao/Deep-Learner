from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
import os
# 1. 填入你的 Windows IP
WINDOWS_IP = "192.168.1.157"  # 请替换为你的 Windows 机器的实际 IP 地址
llm = ChatOllama(
    model="qwen:7b", # 确保 Windows 上有这个模型
    base_url=f"http://{WINDOWS_IP}:11434",
    temperature=0,
    timeout=10 # 10秒不回就报错
)
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest", # 确保 Windows 上有这个模型
    base_url=f"http://{WINDOWS_IP}:11434",
)
llm_openai = ChatOpenAI(
    model="gpt-4o-mini", # 或者 "gpt-4o-mini"
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def chat_completion(prompt: str, system_prompt: str = "你是一个专业的人工智能助手"):
    """
    适配 LangChain Ollama 的通用调用函数
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    
    # 使用 LangChain 的 invoke 方法
    response = llm_openai.invoke(messages)
    
    # 返回字符串内容
    return response.content

def get_actual_dim():
    # 动态测试一次，确保拿到真实维度
    test_vec = embedding_model.embed_query("test")
    return len(test_vec)