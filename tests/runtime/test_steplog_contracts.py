from tools .retrieve_tool .base import SearchResult 
from nodes .recall_ltm import recall_ltm_node 
from nodes .persist_ltm import persist_ltm_node 
from nodes .retrieve import retrieve_node 
from config .settings import AppConfig 


def test_recall_ltm_steplog_contains_threshold_and_topk (monkeypatch ):
    class FakeLTM :
        def recall (self ,query ):
            assert query =="what is transformer"
            return ["Memory A","Memory B"]

    monkeypatch .setattr ("nodes.recall_ltm.LTM",FakeLTM )

    state ={"query":"what is transformer","steps_log":[]}
    out =recall_ltm_node (state )

    assert out ["long_term_memory"]=="Memory A | Memory B"
    log =out ["steps_log"][-1 ]
    assert log .node =="recall_ltm"
    assert log .info ["memory"]["memory_count"]==2 
    assert log .info ["memory"]["recall_top_k"]==AppConfig .LTM_RECALL_TOP_K 
    assert log .info ["memory"]["recall_threshold"]==AppConfig .LTM_RECALL_THRESHOLD 


def test_persist_ltm_steplog_tracks_filter_counts (monkeypatch ):
    saved ={}

    class FakeLTM :
        def upsert (self ,entries ):
            saved ["entries"]=entries 
            return len (entries )

    def fake_run_prompt (prompt_cls ,state ):
        return {
        "fact_candidates":[
        {"key":"k1","type":"fact","content":"c1","score":0.9 },
        {"key":"k2","type":"fact","content":"c2","score":0.1 },
        {"type":"fact","content":"c3","score":0.95 },
        ]
        }

    monkeypatch .setattr ("nodes.persist_ltm.LTM",FakeLTM )
    monkeypatch .setattr ("nodes.persist_ltm.run_prompt",fake_run_prompt )
    monkeypatch .setattr (AppConfig ,"LTM_WRITE_ENABLED",True )

    state ={
    "query":"q1",
    "response":"r1",
    "steps_log":[],
    }
    out =persist_ltm_node (state )

    assert len (saved ["entries"])==1 
    assert saved ["entries"][0 ]["key"]=="k1"
    assert saved ["entries"][0 ]["score"]==0.9 

    log =out ["steps_log"][-1 ]
    assert log .node =="persist_ltm"
    assert log .info ["llm_output"]["fact_candidates_raw_count"]==3 
    assert log .info ["llm_output"]["fact_candidates_kept_count"]==1 
    assert log .info ["memory"]["stored_count"]==1 


def test_retrieve_steplog_reflects_rewritten_query_and_count (monkeypatch ):
    class FakePipeline :
        def run (self ,query ):
            assert query =="rewritten q"
            return [
            SearchResult (
            id ="doc-1",
            content ="doc-content",
            score =0.88 ,
            metadata ={"title":"t1"},
            source_type ="vector",
            )
            ]

    monkeypatch .setattr ("nodes.retrieve.RetrievalPipeline",FakePipeline )

    state ={
    "query":"raw q",
    "rewritten_query":"rewritten q",
    "steps_log":[],
    }
    out =retrieve_node (state )

    assert len (out ["context_pool"])==1 
    assert out ["context_pool"][0 ]["id"]=="doc-1"

    log =out ["steps_log"][-1 ]
    assert log .node =="retrieve"
    assert log .info ["state"]["used_rewritten_query"]is True 
    assert log .info ["memory"]["context_pool_count"]==1 
    assert log .info ["memory"]["context_pool_preview"][0 ]["id"]=="doc-1"
