import os
import sys
from dotenv import load_dotenv
from pymilvus import connections, utility
from elasticsearch import Elasticsearch

# --- 确保项目根目录在 sys.path 中，支持绝对导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.settings import AppConfig

def cleanup_milvus():
    """清理 Milvus 集合"""
    print(f"\n--- 正在清理 Milvus ---")
    try:
        connections.connect(
            alias="default", 
            host=AppConfig.MILVUS_HOST, 
            port=AppConfig.MILVUS_PORT
        )
        coll_name = AppConfig.MILVUS_COLLECTION
        if utility.has_collection(coll_name):
            utility.drop_collection(coll_name)
            print(f"✅ 已删除 Milvus 集合: {coll_name}")
        else:
            print(f"ℹ️ Milvus 集合 '{coll_name}' 不存在，无需操作。")
    except Exception as e:
        print(f"❌ Milvus 清理失败: {e}")
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

def cleanup_elasticsearch():
    """清理 Elasticsearch 索引"""
    print(f"\n--- 正在清理 Elasticsearch ---")
    try:
        es_url = f"http://{AppConfig.ES_HOST}:{AppConfig.ES_PORT}"
        es = Elasticsearch([es_url])
        index_name = AppConfig.ES_INDEX
        
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"✅ 已删除 ES 索引: {index_name}")
        else:
            print(f"ℹ️ ES 索引 '{index_name}' 不存在，无需操作。")
    except Exception as e:
        print(f"❌ Elasticsearch 清理失败: {e}")

def run_total_cleanup():
    load_dotenv()
    print("🛠️ 开始 Deep-Learner 数据库全量清理任务...")
    
    cleanup_milvus()
    cleanup_elasticsearch()
    
    print("\n" + "="*50)
    print("✨ 所有旧数据已清理。你可以开始运行新的 Ingestion 脚本了。")
    print("="*50)

if __name__ == "__main__":
    run_total_cleanup()