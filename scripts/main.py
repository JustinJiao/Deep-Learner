# scripts/main.py

"""
Deep-Learner 主程序入口（生产模式）

运行：
    python -m scripts.main
"""

from core.executor import AgentExecutor


def main():
    executor = AgentExecutor()
    session_id = "main-session"

    print("\n==============================")
    print("Deep-Learner Production Mode")
    print("==============================")
    print("Commands: /exit, /new")
    print()

    while True:
        try:
            user_input = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Bye.")
            break

        if user_input == "/new":
            session_id = "main-session-new"
            print("🔄 New session started.")
            continue

        state = executor.run(
            session_id=session_id,
            query=user_input
        )

        response = state.get("response", "")
        status = state.get("run_status")

        print("\nAssistant> ", response)

        if status != "ok":
            print(f"\n[System status: {status}]")

        print()


if __name__ == "__main__":
    main()
