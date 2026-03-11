# scripts/agent_memory_stress_test.py

"""Deep-Learner 10 rounds of Memory stress test (enhanced version)

Function:
- turn cumulative verification
- recent window verification
- summary generates verification
- compressed_until check
- memory recall verification
- Automatically dump full STM in case of failure"""

import sys 
from core .executor import AgentExecutor 
from session .store import get_session 


def fail_dump (session_id ,state ,message ):
    print ("\n❌ FAIL:",message )

    ctx =get_session (session_id )
    stm =ctx .stm 

    print ("\n===== DEBUG DUMP =====")
    print ("Response:",state .get ("response"))
    print ("\nRecent Messages:",state .get ("recent_messages"))
    print ("\nShort Term Memory:",state .get ("short_term_memory"))
    print ("\nSummary Blocks:",stm .get ("summary"))
    print ("\nCompressed Until:",stm .get ("compressed_until"))
    print ("\nAll Messages:",stm .get ("messages"))
    print ("======================\n")

    sys .exit (1 )


def assert_true (cond ,msg ,session_id =None ,state =None ):
    if not cond :
        if session_id and state :
            fail_dump (session_id ,state ,msg )
        else :
            print (f"❌ FAIL: {msg}")
            sys .exit (1 )
    else :
        print (f"✅ PASS: {msg}")


def run_test ():

    print ("\n==============================")
    print ("10-ROUND MEMORY STRESS TEST")
    print ("==============================\n")

    executor =AgentExecutor ()
    session_id ="memory-stress-session"

    queries =[
    "i like blue",
    "I live in Beijing",
    "I am 25 years old",
    "i like cats",
    "I like programming",
    "Where did I just say I live?",
    "What color did I just say I like?",
    "What animal did I just say I like?",
    "Did I just say how old I am?",
    "To summarize the personal information we talked about"
    ]

    expected_memory_checks ={
    5 :"Beijing",
    6 :"blue",
    7 :"cat",
    8 :"25",
    }

    for i ,q in enumerate (queries ):
        print (f"\n---- Round {i+1} ----")
        state =executor .run (session_id =session_id ,query =q )

        assert_true (
        state .get ("run_status")=="ok",
        "run_status ok",
        session_id ,
        state 
        )

        ctx =get_session (session_id )
        stm =ctx .stm 

        messages =stm .get ("messages",[])
        recent =stm .get ("recent_messages",[])
        summary =stm .get ("summary",[])
        compressed_until =stm .get ("compressed_until")

        assert_true (
        len (messages )==i +1 ,
        f"messages count correct at round {i+1}",
        session_id ,
        state 
        )

        assert_true (
        len (recent )<=3 ,
        f"recent_messages window correct at round {i+1}",
        session_id ,
        state 
        )

        assert_true (
        compressed_until <=len (messages ),
        f"compressed_until valid at round {i+1}",
        session_id ,
        state 
        )

        # Must be compressed after 5 rounds
        if i +1 >=5 :
            assert_true (
            len (summary )>=1 ,
            "summary generated after threshold",
            session_id ,
            state 
            )

            # memory recall check
        if i in expected_memory_checks :
            expected_word =expected_memory_checks [i ]
            response =state .get ("response","")

            assert_true (
            expected_word in response ,
            f"memory recall correct for {expected_word}",
            session_id ,
            state 
            )

    print ("\n==============================")
    print ("🎉 MEMORY STRESS TEST PASSED")
    print ("==============================\n")


if __name__ =="__main__":
    run_test ()
