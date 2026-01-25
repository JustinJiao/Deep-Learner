import sys
import os

# 确保项目根目录在系统路径中，防止 Import Error
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.agent.graph import create_deep_learner_graph
from schemas.agent_state import AgentState
from config.settings import AppConfig
from colorama import Fore, Style, init

# 初始化终端颜色显示
init(autoreset=True)

def debug_infrastructure():
    from pymilvus import connections
    from elasticsearch import Elasticsearch
    from config.settings import AppConfig
    
    print("\n🔍 [基础设施健康检查]...")
    
    # 1. 测试 Milvus
    try:
        connections.connect("default", host=AppConfig.MILVUS_HOST, port=AppConfig.MILVUS_PORT, timeout=5)
        print(f"✅ Milvus ({AppConfig.MILVUS_HOST}): 连接成功")
        connections.disconnect("default")
    except Exception as e:
        print(f"❌ Milvus 连接失败: 请检查 Docker 端口映射或 VPN 状态")
        return False

    # 2. 测试 ES
    try:
        es = Elasticsearch([f"http://{AppConfig.ES_HOST}:{AppConfig.ES_PORT}"], request_timeout=5)
        if es.ping():
            print(f"✅ Elasticsearch: 连接成功")
        else:
            raise Exception("Ping 失败")
    except Exception as e:
        print(f"❌ Elasticsearch 连接失败: 请检查 Docker 状态")
        return False
    
    return True
def run_deep_learner():
    print(Fore.CYAN + "="*50)
    print(Fore.CYAN + "🚀 Deep-Learner Agentic RAG 系统启动")
    print(Fore.CYAN + f"📍 模型端: {AppConfig.OLLAMA_BASE_URL} ({AppConfig.LLM_PROVIDER})")
    print(Fore.CYAN + "="*50)

    # 1. 初始化大脑 (编译 LangGraph)
    app = create_deep_learner_graph()

    while True:
        user_input = input(Fore.YELLOW + "\n👨‍💻 请输入你的学习问题 (输入 'exit' 退出): ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break

        # 2. 初始化工作记忆状态
        # 严格遵循我们在 schemas/agent_state.py 中定义的协议
        initial_state = {
            "query": user_input,
            "plan": [],
            "retrieved_contents": [],
            "response": "",
            "is_hallucination": False,
            "steps_log": [],
            "messages": [],
            "metadata": {"start_time": os.times()[4]}
        }

        # 3. 运行 Agent 流水线
        print(Fore.GREEN + "\n🧠 Deep-Learner 正在思考中...")
        
        # 工业级配置：这里可以传入 thread_id 供后期接入持久化记忆 (Memory)
        config = {"configurable": {"thread_id": "test_session_001"}}
        
        try:
            # 执行图并获取最终状态
            final_state = app.invoke(initial_state, config=config)

            # 4. 展示决策链路 (思考链可视化)
            print(Fore.MAGENTA + "\n--- 🛰️ Agent 决策链路 (Thought Chain) ---")
            for step in final_state.get("steps_log", []):
                print(Fore.WHITE + f"[{step.node.upper()}] {step.thought}")

            # 5. 输出最终教学内容
            print(Fore.CYAN + "\n--- 👨‍🏫 导师回答 ---")
            print(Fore.WHITE + final_state.get("response", "未能生成回答"))

            # 6. 输出审计意见 (如果审计未通过，用户能看到反思结果)
            if final_state.get("is_hallucination"):
                print(Fore.RED + "\n⚠️ 警告：该内容可能包含幻觉，审计员建议重新校验。")
            else:
                print(Fore.GREEN + "\n✅ 逻辑审计通过，内容准确性达标。")

        except Exception as e:
            print(Fore.RED + f"❌ 运行异常: {str(e)}")

if __name__ == "__main__":
    if debug_infrastructure():
        run_deep_learner()
    else:
        print(Fore.RED + "❌ 基础设施检查未通过，无法启动 Deep-Learner。请修复上述问题后重试。")