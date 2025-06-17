import pytest
from classes.rag_query_processor import RAGQueryProcessor

@pytest.fixture
def mock_llm_client(mocker):
    """Provides a mocked LLMClient."""
    client = mocker.Mock()
    client.query.return_value = "This is the final answer."
    return client

@pytest.fixture
def mock_retriever(mocker):
    """Provides a mocked ChromaDBRetriever."""
    retriever = mocker.Mock()
    return retriever

def test_processor_with_rag_and_docs_found(mock_llm_client, mock_retriever):
    """
    Tests the RAG-enabled flow where the retriever finds relevant documents.
    """
    # GIVEN that the retriever finds a relevant document
    retrieved_docs = [{"context": "Peanut allergy is deadly.", "score": 0.1}]
    mock_retriever.query.return_value = retrieved_docs
    
    # Instantiate the processor with RAG enabled
    processor = RAGQueryProcessor(llm_client=mock_llm_client, retriever=mock_retriever, use_rag=True)
    
    # WHEN a query is made
    processor.query("How dangerous is peanut allergy?")
    
    # THEN the retriever should have been called
    mock_retriever.query.assert_called_once_with("How dangerous is peanut allergy?")
    
    # AND the LLM client should be called with a prompt containing the retrieved context
    mock_llm_client.query.assert_called_once()
    final_prompt = mock_llm_client.query.call_args[0][0]
    
    assert "Context:\n        Peanut allergy is deadly." in final_prompt, "Expected context not found in the prompt."
    assert "Question:\n        How dangerous is peanut allergy?" in final_prompt, "Expected question not found in the prompt."

def test_processor_with_rag_disabled(mock_llm_client, mock_retriever):
    """
    Tests the flow where RAG is disabled, so the retriever should not be used.
    """
    # Instantiate the processor with RAG disabled
    processor = RAGQueryProcessor(llm_client=mock_llm_client, retriever=mock_retriever, use_rag=False)
    
    # WHEN a query is made
    processor.query("How dangerous is peanut allergy?")
    
    # THEN the retriever should NOT have been called
    mock_retriever.query.assert_not_called()
    
    # AND the LLM client is called with a prompt that has no context
    mock_llm_client.query.assert_called_once()
    final_prompt = mock_llm_client.query.call_args[0][0]
    
    assert final_prompt == "How dangerous is peanut allergy?", "No RAG - expected direct query to LLM."