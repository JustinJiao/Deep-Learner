import streamlit as st
import uuid

from core.executor import AgentExecutor

st.set_page_config(page_title="Deep-Learner", layout="wide")
st.title("🧠 Deep-Learner 2.1")

# Session 初始化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "executor" not in st.session_state:
    st.session_state.executor = AgentExecutor()

# Sidebar
with st.sidebar:
    st.subheader("Session Control")
    st.code(st.session_state.session_id)

    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    debug_mode = st.checkbox("Show Debug Info", value=False)

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 输入框
if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = st.session_state.executor.run(
                query=prompt,
                session_id=st.session_state.session_id
            )

            # 兼容旧结构
            if isinstance(result, str):
                response = result
                citations = []
                run_status = "ok"
                steps_log = None
            else:
                response = result.get("response", "")
                citations = result.get("citations", [])
                run_status = result.get("run_status", "")
                steps_log = result.get("steps_log", [])

            st.write(response)

            # ===== 引用显示 =====
            if citations:
                st.markdown("### 📚 References")
                for c in citations:
                    with st.expander(
                        f"{c.get('title', 'Unknown')} "
                        f"(score={c.get('score', 0):.2f})"
                    ):
                        st.write(f"Document ID: {c.get('id')}")
                        quote = str(c.get("quote", "") or "").strip()
                        if quote:
                            st.caption(quote)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Debug
    if debug_mode:
        st.divider()
        st.subheader("🔍 Debug Info")
        st.write("Run Status:", run_status)
        if steps_log:
            st.json(steps_log)
