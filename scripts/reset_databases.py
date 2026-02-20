from pymilvus import connections, utility
from config.factory import ResourceFactory
from config.settings import AppConfig


def reset_milvus_collections():

    connections.connect(
        "default",
        host=AppConfig.MILVUS_HOST,
        port=AppConfig.MILVUS_PORT,
    )

    # 删除主知识库
    if utility.has_collection(AppConfig.MILVUS_COLLECTION):
        utility.drop_collection(AppConfig.MILVUS_COLLECTION)
        print(f"Dropped collection: {AppConfig.MILVUS_COLLECTION}")

    # 删除 LTM
    if utility.has_collection(AppConfig.MILVUS_LTM_COLLECTION):
        utility.drop_collection(AppConfig.MILVUS_LTM_COLLECTION)
        print(f"Dropped collection: {AppConfig.MILVUS_LTM_COLLECTION}")

    print("Milvus reset complete.")


def reset_es_index():
    es = ResourceFactory.get_es_client()
    index_name = AppConfig.ES_INDEX

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Dropped ES index: {index_name}")
    else:
        print(f"ES index not found, skip: {index_name}")

    print("Elasticsearch reset complete.")


def reset_databases():
    reset_milvus_collections()
    reset_es_index()


if __name__ == "__main__":
    reset_databases()
