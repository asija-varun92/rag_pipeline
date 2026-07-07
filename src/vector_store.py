import os
import numpy as np
import chromadb
from typing import List, Any
import uuid

class VectorStore:
    """Handle vector store operations using ChromaDB"""
    def __init__(self, collection_name: str= "pdf_documments", persist_directory: str = "../data/chroma_db"):
        """Initilize the vector store
        Args:
            collection_name (str): Name of the collection in ChromaDB. Default is "pdf_documments"
            persist_directory (str): Directory to persist the ChromaDB database. Default is "../data/chroma_db"
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # create perisist chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # create collection if not exists
            self.collection = self.client.get_or_create_collection(name=self.collection_name, 
                                                                   metadata={"description": "PDF documents embedding for RAG",
                                                                             "hnsw:space": "cosine"})
        
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing collection size: {self.collection.count()}")
        except Exception as e:
            ValueError(f"Error occurred while initializing the vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store
        Args:
            documents (List[Any]): List of langchain documents
            embeddings (np.ndarray): Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")

        print(f"Adding {len(documents)} documents to the vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate a unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            # document text
            documents_texts.append(doc.page_content)

            # embedding
            embeddings_list.append(embedding.tolist())

        try:
            # add to collection
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_texts,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} documents to the vector store.")
            print(f"Total collection size after addition: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error occurred while adding documents to the vector store: {e}")
            raise