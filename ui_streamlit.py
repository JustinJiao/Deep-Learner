import streamlit as st 
import uuid 

from core .executor import AgentExecutor 

st .set_page_config (page_title ="Deep-Learner",layout ="wide")
st .title ("🧠 Deep-Learner 3.0")


def _escape_latex_dollar (text :str )->str :
# Streamlit markdown treats `$...$` as math; escape to keep plain-text currency display.
    return str (text or "").replace ("$",r"\$")

    # Session initialization
if "session_id"not in st .session_state :
    st .session_state .session_id =str (uuid .uuid4 ())

if "messages"not in st .session_state :
    st .session_state .messages =[]

if "executor"not in st .session_state :
    st .session_state .executor =AgentExecutor ()

    # Sidebar
with st .sidebar :
    st .subheader ("Session Control")
    st .code (st .session_state .session_id )

    if st .button ("🔄 New Session"):
        st .session_state .session_id =str (uuid .uuid4 ())
        st .session_state .messages =[]
        st .rerun ()

    debug_mode =st .checkbox ("Show Debug Info",value =False )

    # Show historical messages
for msg in st .session_state .messages :
    with st .chat_message (msg ["role"]):
        st .write (_escape_latex_dollar (msg ["content"]))

        # Input box
if prompt :=st .chat_input ("Ask something..."):

    st .session_state .messages .append ({
    "role":"user",
    "content":prompt 
    })

    with st .chat_message ("user"):
        st .write (_escape_latex_dollar (prompt ))

    with st .chat_message ("assistant"):
        with st .spinner ("Thinking..."):

            result =st .session_state .executor .run (
            query =prompt ,
            session_id =st .session_state .session_id 
            )

            # Compatible with older structures
            if isinstance (result ,str ):
                response =result 
                citations =[]
                run_status ="ok"
                steps_log =None 
            else :
                response =result .get ("response","")
                citations =result .get ("citations",[])
                run_status =result .get ("run_status","")
                steps_log =result .get ("steps_log",[])

            st .write (_escape_latex_dollar (response ))

            # ===== Reference display =====
            if citations :
                st .markdown ("### 📚 References")
                for c in citations :
                    with st .expander (
                    f"{c.get('title', 'Unknown')} "
                    f"(score={c.get('score', 0):.2f})"
                    ):
                        st .write (f"Document ID: {c.get('id')}")
                        quote =str (c .get ("quote","")or "").strip ()
                        if quote :
                            st .caption (_escape_latex_dollar (quote ))

    st .session_state .messages .append ({
    "role":"assistant",
    "content":_escape_latex_dollar (response )
    })

    # Debug
    if debug_mode :
        st .divider ()
        st .subheader ("🔍 Debug Info")
        st .write ("Run Status:",run_status )
        if steps_log :
            st .json (steps_log )
