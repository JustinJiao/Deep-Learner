from typing import List ,Dict 
from tools .retrieve_tool .base import SearchResult 
from tools .retrieve_tool .vector import VectorRetriever 
from tools .retrieve_tool .keyword import KeywordRetriever 
from tools .retrieve_tool .rerank import Reranker 
from config .settings import AppConfig 

class RetrievalPipeline :
    def __init__ (self ,vector_r =None ,keyword_r =None ,reranker =None ):
        """Fully automated initialization, supports manual injection for testing"""
        self .vector_r =vector_r or VectorRetriever ()
        self .keyword_r =keyword_r or KeywordRetriever ()
        self .reranker =reranker or Reranker ()
        self .rrf_k =AppConfig .RRF_K 
        self .score_gate_enabled =AppConfig .RETRIEVAL_SCORE_GATE_ENABLED 
        self .min_rerank_score =AppConfig .RETRIEVAL_MIN_RERANK_SCORE 

    def _apply_rerank_score_gate (
    self ,results :List [SearchResult ]
    )->List [SearchResult ]:
        if not self .score_gate_enabled :
            return results 
        return [doc for doc in results if doc .score >=self .min_rerank_score ]

    def _rrf_fusion (self ,v_results :List [SearchResult ],k_results :List [SearchResult ])->List [SearchResult ]:
        r"""
        Implements RRF fusion algorithm:
        $$RRFscore(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)}$$
        """
        rrf_scores :Dict [str ,float ]={}
        doc_map :Dict [str ,SearchResult ]={}

        for rank ,res in enumerate (v_results ,start =1 ):
            rrf_scores [res .id ]=rrf_scores .get (res .id ,0 )+1.0 /(self .rrf_k +rank )
            doc_map [res .id ]=res 

        for rank ,res in enumerate (k_results ,start =1 ):
            rrf_scores [res .id ]=rrf_scores .get (res .id ,0 )+1.0 /(self .rrf_k +rank )
            if res .id not in doc_map :
                doc_map [res .id ]=res 

        sorted_items =sorted (rrf_scores .items (),key =lambda x :x [1 ],reverse =True )

        final_candidates =[]
        for doc_id ,combined_score in sorted_items :
            doc =doc_map [doc_id ]
            doc .score =combined_score 
            final_candidates .append (doc )

        return final_candidates 

    def run (self ,query :str )->List [SearchResult ]:
    # Use global configuration uniformly
        recall_top_k =AppConfig .RECALL_TOP_K 
        final_top_k =AppConfig .FINAL_TOP_K 

        # 1. Dual-way recall
        v_res =self .vector_r .search (query ,top_k =recall_top_k )
        k_res =self .keyword_r .search (query ,top_k =recall_top_k )

        # 2. RRF fusion
        candidates =self ._rrf_fusion (v_res ,k_res )

        # 3. Fine arrangement
        if self .reranker and candidates :
            reranked =self .reranker .rerank (
            query ,candidates ,top_n =len (candidates )
            )
            filtered =self ._apply_rerank_score_gate (reranked )
            return filtered [:final_top_k ]

        return candidates [:final_top_k ]
