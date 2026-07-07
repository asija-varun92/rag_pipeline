from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore
from src.retrievel import RagRetriever
from src.search import RagSearch


def main():
    docs = load_all_documents("data")
    
    # generate embeddings
    embedding_manager = EmbeddingPipeline()
    chunks = embedding_manager.chunk_documents(docs)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_manager.generate_embeddings(texts)

    # store embeddings
    vector_store = VectorStore()
    vector_store.add_documents(chunks, vectors)

    rag_retriever = RagRetriever(vector_store, embedding_manager)
    # search with LLM
    search = RagSearch(rag_retriever)
    output = search.search("what is datalake?")
    print(output)


if __name__ == "__main__":
    main()
