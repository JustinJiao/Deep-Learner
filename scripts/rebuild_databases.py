import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from config.factory import ResourceFactory
from config.settings import AppConfig
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.parsers.universal import UniversalParser
from ingestion.writers.dual_writer import DualWriter
from utils.crypto_utils import generate_content_id
from pymilvus import connections, utility


SUPPORTED_EXTS = {".txt", ".md", ".py", ".pdf", ".docx", ".doc", ".rtf"}
TEXT_EXTS = {".txt", ".md", ".py"}


@dataclass
class ParsedFile:
    path: str
    chunks: list[dict]
    elapsed_sec: float
    table_chunks: int
    numeric_chunks: int


def _discover_files(docs_path: str) -> list[str]:
    root = Path(docs_path)
    if not root.exists():
        return []
    files: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(str(p))
    files.sort()
    return files


def _parse_and_chunk_file(path: str) -> ParsedFile:
    t0 = time.time()
    parser = UniversalParser()
    chunker = SemanticChunker()
    file_suffix = Path(path).suffix.lower()

    if file_suffix in TEXT_EXTS:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = parser.to_markdown(path)

    chunks = chunker.split_with_overlap(text, path)
    table_chunks = 0
    numeric_chunks = 0
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c, dict) else {}
        if bool(meta.get("is_table")):
            table_chunks += 1
        content = c.get("content", "") if isinstance(c, dict) else str(c)
        if any(ch.isdigit() for ch in content):
            numeric_chunks += 1

    return ParsedFile(
        path=path,
        chunks=chunks,
        elapsed_sec=round(time.time() - t0, 2),
        table_chunks=table_chunks,
        numeric_chunks=numeric_chunks,
    )


def _iter_data_rows(parsed: ParsedFile):
    for idx, chunk in enumerate(parsed.chunks):
        content = chunk["content"] if isinstance(chunk, dict) else str(chunk)
        # 避免跨文件或重复段落内容导致 doc_id 冲突覆盖
        doc_key = f"{parsed.path}::chunk-{idx}::{content}"
        doc_id = generate_content_id(doc_key)
        chunk_metadata = dict(chunk.get("metadata", {})) if isinstance(chunk, dict) else {}
        chunk_metadata.update(
            {
                "source": parsed.path,
                "chunk_index": idx,
            }
        )
        yield {
            "doc_id": doc_id,
            "content": content,
            "metadata": chunk_metadata,
        }


def rebuild():
    docs_path = os.getenv("DOCS_PATH", "data/docs")
    parse_workers = int(
        os.getenv("INGEST_PARSE_WORKERS", str(max(1, min((os.cpu_count() or 1), 4))))
    )
    write_batch_size = int(os.getenv("INGEST_WRITE_BATCH_CHUNKS", "96"))
    reset_target = str(os.getenv("INGEST_RESET_TARGET", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    print(f"[INGEST] docs_path={docs_path}")
    print(
        f"[INGEST] parser_strategy={AppConfig.UNSTRUCTURED_STRATEGY}, "
        f"languages={AppConfig.UNSTRUCTURED_LANGUAGES}, "
        f"infer_table={AppConfig.UNSTRUCTURED_INFER_TABLE_STRUCTURE}"
    )
    print(
        f"[INGEST] parse_workers={parse_workers}, write_batch_size={write_batch_size}, "
        f"embed_batch_size={os.getenv('EMBED_BATCH_SIZE', '16')}"
    )
    print(f"[INGEST] reset_target={reset_target}")

    if reset_target:
        print("[INGEST] resetting target index/collection...")
        es = ResourceFactory.get_es_client()
        if es.indices.exists(index=AppConfig.ES_INDEX):
            es.indices.delete(index=AppConfig.ES_INDEX)
            print(f"[INGEST] deleted ES index: {AppConfig.ES_INDEX}")
        else:
            print(f"[INGEST] ES index not found, skip delete: {AppConfig.ES_INDEX}")

        connections.connect(
            "default",
            host=AppConfig.MILVUS_HOST,
            port=AppConfig.MILVUS_PORT,
        )
        if utility.has_collection(AppConfig.MILVUS_COLLECTION):
            utility.drop_collection(AppConfig.MILVUS_COLLECTION)
            print(f"[INGEST] dropped Milvus collection: {AppConfig.MILVUS_COLLECTION}")
        else:
            print(
                f"[INGEST] Milvus collection not found, skip drop: {AppConfig.MILVUS_COLLECTION}"
            )

    print("[INGEST] creating collections...")
    ResourceFactory.get_milvus_collection()
    ResourceFactory.get_milvus_ltm_collection()
    print("[INGEST] collections ready")

    files = _discover_files(docs_path)
    if not files:
        print("[INGEST] no supported files found, stop")
        return
    print(f"[INGEST] discovered files={len(files)}")

    parsed_files: list[ParsedFile] = []
    failed_files: list[tuple[str, str]] = []

    if parse_workers <= 1:
        for i, path in enumerate(files, 1):
            print(f"[PARSE {i}/{len(files)}] start: {path}")
            try:
                parsed = _parse_and_chunk_file(path)
                parsed_files.append(parsed)
                print(
                    f"[PARSE {i}/{len(files)}] done: chunks={len(parsed.chunks)}, "
                    f"table_chunks={parsed.table_chunks}, numeric_chunks={parsed.numeric_chunks}, "
                    f"elapsed={parsed.elapsed_sec}s"
                )
            except Exception as e:
                failed_files.append((path, str(e)))
                print(f"[PARSE {i}/{len(files)}] failed: {path} | {e}")
    else:
        print("[PARSE] running in process pool...")
        with ProcessPoolExecutor(max_workers=parse_workers) as pool:
            fut_map = {pool.submit(_parse_and_chunk_file, p): p for p in files}
            completed = 0
            for fut in as_completed(fut_map):
                completed += 1
                path = fut_map[fut]
                try:
                    parsed = fut.result()
                    parsed_files.append(parsed)
                    print(
                        f"[PARSE {completed}/{len(files)}] done: {path} | "
                        f"chunks={len(parsed.chunks)}, table_chunks={parsed.table_chunks}, "
                        f"numeric_chunks={parsed.numeric_chunks}, elapsed={parsed.elapsed_sec}s"
                    )
                except Exception as e:
                    failed_files.append((path, str(e)))
                    print(f"[PARSE {completed}/{len(files)}] failed: {path} | {e}")

    if not parsed_files:
        print("[INGEST] all files failed during parse/chunk")
        return

    writer = DualWriter()
    parsed_files.sort(key=lambda x: x.path)

    total_rows = 0
    write_buffer: list[dict] = []
    total_expected = sum(len(p.chunks) for p in parsed_files)
    print(f"[WRITE] expected_rows={total_expected}")

    for i, parsed in enumerate(parsed_files, 1):
        file_rows = 0
        for row in _iter_data_rows(parsed):
            write_buffer.append(row)
            file_rows += 1
            if len(write_buffer) >= write_batch_size:
                writer.write_all(write_buffer)
                total_rows += len(write_buffer)
                print(f"[WRITE] flushed batch, total_written={total_rows}")
                write_buffer = []
        print(
            f"[WRITE {i}/{len(parsed_files)}] file={parsed.path} rows={file_rows} "
            f"table_chunks={parsed.table_chunks} numeric_chunks={parsed.numeric_chunks}"
        )

    if write_buffer:
        writer.write_all(write_buffer)
        total_rows += len(write_buffer)
        print(f"[WRITE] final flush, total_written={total_rows}")

    print("[INGEST] rebuild complete")
    print(
        f"[INGEST] summary: parsed_ok={len(parsed_files)}, parsed_failed={len(failed_files)}, "
        f"rows_written={total_rows}, expected_rows={total_expected}"
    )
    if failed_files:
        for p, msg in failed_files:
            print(f"[INGEST][FAILED] {p} | {msg}")


if __name__ == "__main__":
    rebuild()
