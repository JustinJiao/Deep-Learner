import sys
import os
import uuid
# 确保项目根目录在系统路径中，防止 Import Error
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.agent.graph import create_deep_learner_graph
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

    # 🌟 为本次 CLI 会话生成一个统一的 Thread ID，确保多轮对话记忆生效
    session_thread_id = f"cli_session_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": session_thread_id}, "recursion_limit": 50}

    while True:
        user_input = input(Fore.YELLOW + "\n👨‍💻 请输入你的学习问题 (输入 'exit' 退出): ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break

        # 2. 🌟 修正后的初始化状态
        # 只需要传入该轮次必要的触发参数，其他状态由 Graph 内部维护或 Finalizer 重置
        initial_input = {
            "query": user_input,
            "loop_count": 0,           # 确保从 0 开始计数
            "current_step_idx": 0,     # 确保从第一步开始规划
            "context_pool": [],        # 清空本轮临时知识池
            "source_mapping": {},      # 清空本轮映射
            "steps_log": []            # 清空本轮思考链
        }

        print(Fore.GREEN + "\n🧠 Deep-Learner 正在思考中...")
        
        try:
            # 3. 运行 Agent 流水线
            # 这里不需要传入 messages，MemorySaver 会根据 thread_id 自动加载历史
            final_state = app.invoke(initial_input, config=config)

            # 4. 展示决策链路
            print(Fore.MAGENTA + "\n--- 🛰️ Agent 决策链路 (Thought Chain) ---")
            for step in final_state.get("steps_log", []):
                # 兼容处理：支持对象或字典格式
                node_name = step.node if hasattr(step, 'node') else step.get('node', 'UNKNOWN')
                thought = step.thought if hasattr(step, 'thought') else step.get('thought', '')
                print(Fore.WHITE + f"[{node_name.upper()}] {thought}")

            # 5. 输出最终教学内容
            print(Fore.CYAN + "\n--- 👨‍🏫 导师回答 ---")
            print(Fore.WHITE + final_state.get("response", "未能生成回答"))

            # 6. 审计提醒
            if final_state.get("is_hallucination"):
                print(Fore.RED + "\n⚠️ 警告：该内容未通过逻辑审计，可能包含幻觉。")
            else:
                print(Fore.GREEN + "\n✅ 逻辑审计通过。")

        except Exception as e:
            print(Fore.RED + f"❌ 运行异常: {str(e)}")
            import traceback
            traceback.print_exc() # 打印堆栈方便排查为何节点报错

if __name__ == "__main__":
    if debug_infrastructure():
        run_deep_learner()
    else:
        print(Fore.RED + "❌ 基础设施检查未通过，无法启动 Deep-Learner。请修复上述问题后重试。")