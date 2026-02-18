# scripts/rebuild_databases.py

from config.factory import ResourceFactory
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.writers.dual_writer import DualWriter
from ingestion.parsers.universal import UniversalParser
import os
import uuid


def rebuild():

    print("Creating collections...")

    ResourceFactory.get_milvus_collection()
    ResourceFactory.get_milvus_ltm_collection()

    print("Collections ready.")

    docs_path = "data/docs"

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
                doc_id = f"{file}_{idx}_{uuid.uuid4().hex[:8]}"

                data_list.append({
                    "doc_id": doc_id,
                    "content": chunk["content"] if isinstance(chunk, dict) else chunk,
                    "metadata": {
                        "source": path,
                        "chunk_index": idx,
                    }
                })

            writer.write_all(data_list)

    print("Rebuild complete.")


if __name__ == "__main__":
    rebuild()
