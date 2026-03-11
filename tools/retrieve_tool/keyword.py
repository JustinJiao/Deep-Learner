from typing import List 
from tools .retrieve_tool .base import BaseRetriever ,SearchResult 
from config .settings import AppConfig 
from config .factory import ResourceFactory 

class KeywordRetriever (BaseRetriever ):
    def __init__ (self ,es_client =None ):
        super ().__init__ (name ="ESKeywordRetriever")
        # 🌟 Unified resource acquisition
        self .es =es_client or ResourceFactory .get_es_client ()
        self .index_name =AppConfig .ES_INDEX 

    def search (self ,query :str ,top_k :int )->List [SearchResult ]:
        query_body ={
        "query":{"match":{"content":query }},
        "size":top_k 
        }
        response =self .es .search (index =self .index_name ,body =query_body )

        standard_results =[]
        for hit in response ['hits']['hits']:
            source =hit ['_source']
            standard_results .append (SearchResult (
            id =hit ['_id'],
            content =source .get ("content"),
            score =hit ['_score'],
            metadata =source .get ("metadata",{}),
            source_type ="keyword"
            ))
        return standard_results 