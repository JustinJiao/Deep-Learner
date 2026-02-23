# scripts/rebuild_databases.py

from config.factory import ResourceFactory
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.writers.dual_writer import DualWriter
from ingestion.parsers.universal import UniversalParser
from utils.crypto_utils import generate_content_id
import os


def rebuild():

    print("Creating collections...")

    ResourceFactory.get_milvus_collection()
    ResourceFactory.get_milvus_ltm_collection()

    print("Collections ready.")

    docs_path = os.getenv("DOCS_PATH", "data/docs")
    print(f"Using docs path: {docs_path}")

    chunker = SemanticChunker()
    writer = DualWriter()
    parser = UniversalParser()

    for root, _, files in os.walk(docs_path):
        for file in files:
            path = os.path.join(root, file)

            print(f"Ingesting: {path}")

            try:
                if file.endswith((".txt", ".md", ".py")):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    # 使用 unstructured 解析 docx/pdf 等
                    text = parser.to_markdown(path)

            except Exception as e:
                print(f"Skip {path}: {e}")
                continue

            chunks = chunker.split_with_overlap(text, path)

            data_list = []
            for idx, chunk in enumerate(chunks):
                content = chunk["content"] if isinstance(chunk, dict) else chunk
                doc_id = generate_content_id(content)
                chunk_metadata = (
                    dict(chunk.get("metadata", {}))
                    if isinstance(chunk, dict)
                    else {}
                )
                chunk_metadata.update({
                    "source": path,
                    "chunk_index": idx,
                })

                data_list.append({
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": chunk_metadata,
                })

            writer.write_all(data_list)

    print("Rebuild complete.")


if __name__ == "__main__":
    rebuild()
