import logging
from pathlib import Path
import chromadb
import json
from typing import List

class EmbeddingLoader:
    def __init__(self,
                 cleaned_text_file_list: List[str],
                 cleaned_text_dir: str,
                 embeddings_dir: str,
                 vectordb_dir: str,
                 collection_name: str,
                 batch_size: int = 100): # Increased batch size for efficiency

        self.cleaned_text_file_list = cleaned_text_file_list
        self.cleaned_text_path = Path(cleaned_text_dir)
        self.embeddings_path = Path(embeddings_dir)
        self.vectordb_path = Path(vectordb_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.vectordb_path))
        self.collection = self.client.get_or_create_collection(collection_name)

    def process_files(self):
        """Loads chunks and embeddings and stores them in ChromaDB."""
        for cleaned_chunk_file in self.cleaned_text_file_list:
            original_stem = cleaned_chunk_file.replace('_cleaned_chunks.json', '')
            
            chunk_file_path = self.cleaned_text_path / cleaned_chunk_file
            embedding_file_path = self.embeddings_path / f"{original_stem}_embeddings.json"
            
            if not chunk_file_path.exists() or not embedding_file_path.exists():
                self.logger.warning(f"Missing files for {original_stem}, skipping.")
                continue

            try:
                with open(chunk_file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                with open(embedding_file_path, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)

                if len(chunks) != len(embeddings):
                    self.logger.error(f"Mismatch between chunk count ({len(chunks)}) and embedding count ({len(embeddings)}) for {original_stem}.")
                    continue

                if not chunks:
                    self.logger.warning(f"No chunks found for {original_stem}, skipping.")
                    continue
                
                self.logger.info(f"Storing {len(chunks)} chunks for {original_stem} in ChromaDB...")
                
                # Prepare data for batch insertion
                ids = [f"{original_stem}::chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"text": chunk, "source": original_stem, "chunk_index": i}
                    for i, chunk in enumerate(chunks)
                ]
                
                # Add to collection in batches
                for i in range(0, len(ids), self.batch_size):
                    batch_ids = ids[i:i + self.batch_size]
                    batch_embeddings = embeddings[i:i + self.batch_size]
                    batch_metadatas = metadatas[i:i + self.batch_size]
                    
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )
                
                self.logger.info(f"Stored {original_stem} chunks successfully.")

            except Exception as e:
                self.logger.error(f"Failed to process and store {original_stem}: {e}")