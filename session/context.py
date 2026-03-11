# session/context.py

from core .state import STMState 


class SessionContext :

    def __init__ (self ,session_id :str ):
        self .session_id =session_id 

        # Initialize STMState (match new version structure)
        self .stm :STMState ={
        "summary":[],
        "messages":[],
        "recent_messages":[],
        "compressed_until":0 ,
        }
