import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
from pathlib import Path
import logging

class ChromaDBRetriever:
    """Retrieves relevant document chunks from ChromaDB based on a search phrase."""

    def __init__(self, embedding_model_name: str,
                 collection_name: str,
                 vectordb_dir: str,
                 score_threshold: float = 0.5):

        self.vectordb_path = Path(vectordb_dir)
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.score_threshold = score_threshold  # Minimum similarity score for valid results

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ChromaDBRetriever: embedding_model_name: {embedding_model_name}, collection_name: {collection_name}, score_threshold: {score_threshold}")

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for the input text."""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def query(self, search_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection and returns structured results of relevant chunks.
        """
        embedding_vector = self.embed_text(search_phrase)
        results = self.collection.query(query_embeddings=[embedding_vector], n_results=top_k,
            include=["metadatas", "distances"] # Adding metadatas and distances to query
        )

        # Parse results
        retrieved_docs = []
        for doc_id, metadata, distance in zip(results.get("ids", [[]])[0], results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]):
            if distance < self.score_threshold:
                continue  # Skip low-confidence matches
            
            retrieved_docs.append({
                "id": doc_id,
                "score": round(distance, 4),
                "context": metadata.get("text", ""), # Chunk text
                "source": metadata.get("source", "Unknown"),
                "chunk_index": metadata.get("chunk_index", -1)
            })  

        # Sort by score (lower is better in similarity searches)
        retrieved_docs.sort(key=lambda x: x["score"])

        return retrieved_docs[:top_k]
