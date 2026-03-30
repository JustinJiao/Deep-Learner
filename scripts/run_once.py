# scripts/run_once.py

from core .debug_executor import DebugAgentExecutor ,pretty 
from session .store import get_session 
from memory .ltm import LTM 

    # -------------------------
    # STM printing
    # -------------------------

def print_stm (session_id :str ):
    ctx =get_session (session_id )
    stm =ctx .stm 

    print ("\n"+"="*80 )
    print ("SESSION STM STATE")
    print ("="*80 )

    pretty (stm )


    # -------------------------
    # LTM printing
    # -------------------------

def print_ltm ():
    print ("\n"+"="*80 )
    print ("LTM SNAPSHOT (Top 10)")
    print ("="*80 )

    try :
        ltm =LTM ()
        col =ltm .collection 

        results =col .query (
        expr ="key != ''",
        output_fields =["key","content","type","score","timestamp"],
        limit =10 ,
        )

        pretty (results )

    except Exception as e :
        print (f"LTM read error: {e}")


        # -------------------------
        # main function
        # -------------------------

def main ():
    executor =DebugAgentExecutor ()
    session_id ="trace-session"

    print ("\nDeep-Learner TRACE MODE")
    print ("Commands: /exit, /new")

    while True :
        print ("\n"+"#"*100 )
        print (f"Current session: {session_id}")
        print ("#"*100 )

        query =input ("\nUser> ").strip ()

        if not query :
            continue 
        if query =="/exit":
            break 
        if query =="/new":
            session_id =f"trace-session-new"
            print ("New session created.")
            continue 

            # -------------------------
            # BEFORE RUN
            # -------------------------

        print ("\n===== SESSION BEFORE RUN =====")
        print_stm (session_id )

        print ("\n===== EXECUTION TRACE START =====")

        state =executor .run (
        session_id =session_id ,
        query =query 
        )

        print ("\n===== EXECUTION TRACE END =====")

        # -------------------------
        # FINAL RESPONSE
        # -------------------------

        print ("\n"+"="*80 )
        print (f"FINAL RESPONSE (status={state.get('run_status')})")
        print ("="*80 )
        print (state .get ("response"))

        # -------------------------
        # FINAL AGENT STATE
        # -------------------------

        print ("\n===== FINAL AGENT STATE =====")
        pretty (state )

        # -------------------------
        # SESSION AFTER
        # -------------------------

        print ("\n===== SESSION AFTER RUN =====")
        print_stm (session_id )

        # -------------------------
        # LTM
        # -------------------------

        print_ltm ()


if __name__ =="__main__":
    main ()
