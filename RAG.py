import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from MVP.services.LLM import embedding_model

def save_chunks_to_file(filename, chunks, strategy_name):
    """将切分后的内容写入文件，方便查看分界线"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== {strategy_name} 结果报告 ===\n")
        f.write(f"总块数: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks):
            # 兼容 Document 对象和 纯文本字符串
            content = chunk.page_content if hasattr(chunk, 'page_content') else chunk
            f.write(f"--- Chunk {i+1} (长度: {len(content)}) ---\n")
            f.write(content)
            f.write("\n\n" + "="*50 + "\n\n")
    print(f"✅ 已保存至: {filename}")

def test_chunking_strategies(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"--- 正在加载并清洗 PDF: {file_path} ---")
    loader = UnstructuredPDFLoader(file_path, mode="single")
    docs = loader.load()
    full_text = docs[0].page_content
    
    # 1. 递归字符切分 (Recursive)
    print("正在执行: Recursive Character Level Chunking...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_documents(docs)
    save_chunks_to_file("output_recursive.txt", recursive_chunks, "Recursive Strategy")

    # 2. Token 切分 (Token)
    print("正在执行: Token Level Chunking...")
    token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, 
        chunk_overlap=20
    )
    token_chunks = token_splitter.split_text(full_text)
    save_chunks_to_file("output_token.txt", token_chunks, "Token Strategy")

    # 3. 语义切分 (Semantic) - 耗时可能较长
    print("正在执行: Semantic Chunking (这需要调用 Embedding 模型，请稍候)...")
    semantic_splitter = SemanticChunker(
        embedding_model, 
        breakpoint_threshold_type="percentile"
    )
    semantic_chunks = semantic_splitter.split_documents(docs)
    save_chunks_to_file("output_semantic.txt", semantic_chunks, "Semantic Strategy")

    print("\n--- 所有实验已完成，请检查生成的三个 .txt 文件 ---")

if __name__ == "__main__":
    test_chunking_strategies("1.pdf")