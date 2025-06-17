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

        # Direct LLM query if RAG is not used
        if not self.use_rag:
            response = self.llm_client.query(query_text)
            self.logger.debug(f"LLM Response: {response}")
            return response
            
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
        Your task is to synthesize all relevant information from the provided context to construct a thorough response. If the context addresses the user's query from multiple perspectives or for different subtopics, you should organize your response to reflect these distinctions. 
        
        Your primary and most important rule is to first determine if the user's questions can be answered directly from the provided documents.
        If the answer is in the context, you must synthesize all relevant information into a comprehensive, detailed, and structured response. Organize your response logically using headings, subheadings, and sections for clarity.
        If the answer is not in the context, you must respond with only the exactly phrase "Based on the provided documents, I cannot answer this question." Do not fabricate information, add explanations, or try to infer an answer.

        All responses must be based strictly on the provided context without making assumptions or offering outside medical advice.
        
        Context:
        {final_context if final_context else "No relevant context found."}

        Question:
        {query_text}
        """

        self.logger.debug(f"Prompt to LLM: {final_prompt}")

        response = self.llm_client.query(final_prompt)
        self.logger.debug(f"RAG Response: {response}")

        return response


