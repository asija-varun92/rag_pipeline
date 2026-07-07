from typing import List, Dict, Any

class RagRetriever:
    """Handle query-based retrieval from vector store."""
    def __init__(self, vector_store, embedding_manager):
        """Initialize the RagRetriever with a vector store and an embedding manager.
        
        Args:
            vector_store: An instance of the vector store to retrieve documents from.
            embedding_manager: An instance of the embedding manager to generate embeddings.
        """
        self.vector_store = vector_store
        self.emebedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float =0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query.
        
        Args:
            query: The input query string for which to retrieve documents.
            top_k: The number of top documents to retrieve (default is 5).
            similarity_threshold: The minimum similarity score for a document to be considered relevant (default is 0.0).
        
        Returns:
            A list of retrieved documents that meet the similarity threshold.
        """
        # Generate embedding for the query
        query_embedding = self.emebedding_manager.generate_embeddings([query])[0]

        # Retreive documents from the vector store based on the query embedding
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []
            # Filter results based on the similarity threshold
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadata = results["metadatas"][0]
                distance = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadata, distance)):
                    # convert distance to similarity score
                    similarity_score = 1 - distance
                    
                    if similarity_score >= similarity_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1
                            }
                        )
                print(f"retrieved docs: {len(retrieved_docs)}")
            else:
                print("No documents retrieved for the given query.")
            return retrieved_docs
        except Exception as e:
            print(f"Error occurred during retrieval: {e}")
            return []