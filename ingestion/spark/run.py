import os 

from pyspark import SparkContext 
from ingestion .parsers .universal import UniversalParser 
from ingestion .chunkers .semantic import SemanticChunker 
from ingestion .writers .dual_writer import DualWriter 

# Automatically locate the project root directory to ensure that the data path is valid wherever it is executed
project_root =os .path .dirname (
os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
)
relative_docs_path =os .getenv ("DATA_PATH","data/docs")
if os .path .isabs (relative_docs_path ):
    absolute_docs_path =relative_docs_path 
else :
    absolute_docs_path =os .path .join (project_root ,relative_docs_path )

def process_file (path_binary ):
    """Spark RDD processing function: Parse -> Split"""
    path ,_ =path_binary 
    local_path =path .replace ("file:","")

    # Instantiate the parsing component
    parser =UniversalParser ()
    chunker =SemanticChunker ()

    # Parse and slice
    try :
        md_text =parser .to_markdown (local_path )
        return chunker .split_with_overlap (md_text ,local_path )
    except Exception as e :
        print (f"⚠️ Failed to parse file {local_path} failed: {e}")
        return []

if __name__ =="__main__":
    print (f"🚀 Deep-Learner ingestion started")
    print (f"📂 Scan path: {absolute_docs_path}")

    sc =SparkContext (appName ="DeepLearner-Ingestion")
    raw_rdd =sc .binaryFiles (absolute_docs_path )

    # Filter shadow files and system hidden files
    raw_rdd =raw_rdd .filter (lambda x :
    not os .path .basename (x [0 ]).startswith ("~$")and 
    not os .path .basename (x [0 ]).startswith (".")
    )

    # 1. Distributed parallel processing
    all_chunks =raw_rdd .flatMap (process_file ).collect ()

    if not all_chunks :
        print ("⚠️ No valid data found, please check the data/docs/ directory.")
    else :
    # 2. Batch writing (using factory singleton)
        writer =DualWriter ()
        writer .write_all (all_chunks )

        print (f"🎉 Pipeline completed: total indexed chunks = {len(all_chunks)}")
