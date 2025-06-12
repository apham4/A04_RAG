from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging
# from pathlib import Path
# from typing import List
# import json

class RAGQueryProcessor:

    def __init__(self,
                 llm_client: LLMClient,
                 retriever: ChromaDBRetriever,
                 use_rag: bool = False):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")

    def query(self, query_text: str):
        """
        Processes the query with optional RAG.
        """
        self.logger.debug(f"Received query: {query_text}")

        context = ""
        if self.use_rag:
            self.logger.info("-"*80)
            self.logger.info("Using RAG pipeline...")
            retrieved_docs = self.retriever.query(query_text)
            if not retrieved_docs:
                logging.info("*** No relevant documents found.")
            else:
                result = retrieved_docs[0]
                context = result.get('context', '')
                logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
                logging.info(f"Score: {result.get('score', 'N/A')}")
                doc_text = result.get('text', '')
                preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
                logging.info(f"Document: {preview_text}")
                logging.info(f"Context: {context}")
            self.logger.info("-" * 80)
            
        contexts = []
        if self.use_rag:
            self.logger.info("-" * 80)
            self.logger.info("Using RAG pipeline...")
            retrieved_docs = self.retriever.query(query_text)
            if not retrieved_docs:
                logging.info("*** No relevant documents found.")
            else:
                logging.info(f"Retrieved {len(retrieved_docs)} relevant chunk(s).")
                
                # Loop through all retrieved documents to build the context
                for i, result in enumerate(retrieved_docs):
                    contexts.append(result.get('context', ''))
                    
                    logging.info(f"--- Chunk {i+1} ---")
                    logging.info(f"ID: {result.get('id', 'N/A')}")
                    logging.info(f"Score: {result.get('score', 'N/A')}")
                    logging.info(f"Source: {result.get('source', 'N/A')}")
                    logging.info(f"Context: {result.get('context', '')}")
            self.logger.info("-" * 80)
            
        # Join all contexts with a separator
        final_context = "\n---\n".join(contexts)

        # Construct structured prompt
        final_prompt = f"""
        You are a specialized AI assistant with expertise in clinical guidelines for allergy and asthma. 
        Your primary function is to provide accurate answers based strictly on the retrieved context. 
        If the context does not contain the answer, you must state "Based on the provided documents, I cannot answer this question." 
        Do not provide medical advice. 

        Context:
        {final_context if final_context else "No relevant context found."}

        Question:
        {query_text}
        """

        self.logger.debug(f"Prompt to LLM: {final_prompt}")

        response = self.llm_client.query(final_prompt)
        self.logger.debug(f"LLM Response: {response}")

        return response


