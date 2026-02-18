from pymilvus import connections, utility
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


if __name__ == "__main__":
    reset_milvus_collections()
