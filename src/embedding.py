import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the embedding generator
        Args:
            model_name (str): Hugging Face model name for sentence transformers. Default is all-MiniLM-L6-v2
            """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks of specified size with overlap"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"[INFO] Splited {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[Any]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
