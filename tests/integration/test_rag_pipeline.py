import pytest
import json
from pathlib import Path
from typing import Any
from main import (
    step01_ingest_documents,
    step02_generate_embeddings,
    step03_store_vectors,
    step04_retrieve_relevant_chunks,
    step05_generate_response,
)
from classes.config_manager import ConfigManager
import io
import sys

class Args:
    """
    Simple argument holder for pipeline steps.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.fixture
def pipeline_environment(tmp_path: Path, mocker):
    """
    Sets up a temporary environment for the full pipeline test.
    """
    # Set up config and directories
    config_data = {
        "log_level": "debug",
        "raw_input_directory": str(tmp_path / "raw_input"),
        "cleaned_text_directory": str(tmp_path / "cleaned_text"),
        "embeddings_directory": str(tmp_path / "embeddings"),
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectordb_directory": str(tmp_path / "vectordb"),
        "collection_name": "test_collection",
        "llm_api_url": "http://fake-url/v1/chat/completions",
        "llm_model_name": "test-llm",
        "retriever_min_score_threshold": "0.5"
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    mocker.patch("main.config", ConfigManager(config_file))

    # Create a sample input file for ingestion
    raw_input_dir = tmp_path / "raw_input"
    raw_input_dir.mkdir()
    sample_file = raw_input_dir / "sample.txt"
    sample_file.write_text("This is a test document about peanut allergy. " * 50)

    return {
        "input_filename": "sample.txt",
        "query": "What is peanut allergy?",
        "config": config_data,
        "tmp_path": tmp_path,
    }

def test_full_pipeline(pipeline_environment: dict[str, Any], mocker):
    """
    Tests the full pipeline with mocks for LLM and retriever.
    """
    # Mock LLMClient and ChromaDBRetriever responses
    mock_llm_response = "This is a mock LLM answer."
    mock_retriever_response = [
        {"context": "Peanut allergy is a serious condition.", "score": 0.9}
    ]
    mocker.patch("classes.llm_client.LLMClient.query", return_value=mock_llm_response)
    mocker.patch("classes.chromadb_retriever.ChromaDBRetriever.query", return_value=mock_retriever_response)

    env = pipeline_environment

    # Step 1: Ingest documents
    step01_ingest_documents(Args(input_filename=env["input_filename"]))
    cleaned_file = Path(env["config"]["cleaned_text_directory"]) / "sample_cleaned_chunks.json"
    assert cleaned_file.exists(), "Cleaned chunks file was not created"
    with open(cleaned_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    assert isinstance(chunks, list) and len(chunks) > 0, "Chunks should be a non-empty list"

    # Step 2: Generate embeddings
    step02_generate_embeddings(Args(input_filename=env["input_filename"]))
    embeddings_file = Path(env["config"]["embeddings_directory"]) / "sample_embeddings.json"
    assert embeddings_file.exists(), "Embeddings file was not created"
    with open(embeddings_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    assert isinstance(embeddings, list) and len(embeddings) == len(chunks), "Embeddings count mismatch."

    # Step 3: Store vectors in vector database
    step03_store_vectors(Args(input_filename=env["input_filename"]))
    vectordb_dir = Path(env["config"]["vectordb_directory"])
    assert vectordb_dir.exists(), "Vector database directory was not created"

    # Step 4: Retrieve relevant chunks for a mocked query
    step04_retrieve_relevant_chunks(Args(query_args=env["query"]))

    # Step 5: Generate response (RAG enabled), capture stdout
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        step05_generate_response(Args(
            input_filename=env["input_filename"],
            query_args=env["query"],
            use_rag=True
        ))
    finally:
        sys.stdout = sys_stdout

    # The mocked LLM response should be in the output
    output = captured_output.getvalue()
    assert mock_llm_response in output

def test_ingest_empty_file(tmp_path: Path, mocker):
    """
    Tests ingesting an empty file. 
    Output file maybe empty or not created.
    """
    # Set up config and directories
    config_data = {
        "log_level": "debug",
        "raw_input_directory": str(tmp_path / "raw_input"),
        "cleaned_text_directory": str(tmp_path / "cleaned_text"),
        "embeddings_directory": str(tmp_path / "embeddings"),
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectordb_directory": str(tmp_path / "vectordb"),
        "collection_name": "test_collection",
        "llm_api_url": "http://fake-url/v1/chat/completions",
        "llm_model_name": "test-llm",
        "retriever_min_score_threshold": "0.5"
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    mocker.patch("main.config", ConfigManager(config_file))

    # Create an empty input file
    raw_input_dir = tmp_path / "raw_input"
    raw_input_dir.mkdir()
    sample_file = raw_input_dir / "empty.txt"
    sample_file.write_text("")

    # Run ingestion step with an empty file
    step01_ingest_documents(Args(input_filename="empty.txt"))
    cleaned_file = Path(config_data["cleaned_text_directory"]) / "empty_cleaned_chunks.json"
    if cleaned_file.exists():
        with open(cleaned_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Output should be an empty list
        assert isinstance(chunks, list)
        assert len(chunks) == 0

def test_embedding_mismatch(tmp_path: Path, mocker):
    """
    Test handling of mismatched chunks and embeddings.
    Should log error or skip.
    """
    # Set up config and directories
    config_data = {
        "log_level": "debug",
        "raw_input_directory": str(tmp_path / "raw_input"),
        "cleaned_text_directory": str(tmp_path / "cleaned_text"),
        "embeddings_directory": str(tmp_path / "embeddings"),
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectordb_directory": str(tmp_path / "vectordb"),
        "collection_name": "test_collection",
        "llm_api_url": "http://fake-url/v1/chat/completions",
        "llm_model_name": "test-llm",
        "retriever_min_score_threshold": "0.5"
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    mocker.patch("main.config", ConfigManager(config_file))

    # Create mismatched chunks and embeddings
    cleaned_dir = tmp_path / "cleaned_text"
    embeddings_dir = tmp_path / "embeddings"
    cleaned_dir.mkdir()
    embeddings_dir.mkdir()
    chunks_file = cleaned_dir / "mismatch_cleaned_chunks.json"
    embeddings_file = embeddings_dir / "mismatch_embeddings.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(["chunk1", "chunk2"], f)
    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump([[0.1, 0.2, 0.3]], f)  # Only one embedding

    # Run store vectors step with mismatched files
    step03_store_vectors(Args(input_filename="mismatch.txt"))
    
    # The vector database directory should still be created
    vectordb_dir = Path(config_data["vectordb_directory"])
    assert vectordb_dir.exists()

def test_llm_connection_error(tmp_path: Path, mocker):
    """
    Test LLMClient connection error handling in the pipeline.
    Should output an error message from the LLM mock.
    """
    # Set up config and directories
    config_data = {
        "log_level": "debug",
        "raw_input_directory": str(tmp_path / "raw_input"),
        "cleaned_text_directory": str(tmp_path / "cleaned_text"),
        "embeddings_directory": str(tmp_path / "embeddings"),
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectordb_directory": str(tmp_path / "vectordb"),
        "collection_name": "test_collection",
        "llm_api_url": "http://localhost:9999/invalid",
        "llm_model_name": "test-llm",
        "retriever_min_score_threshold": "0.5"
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    mocker.patch("main.config", ConfigManager(config_file))

    # Create a sample input file
    raw_input_dir = tmp_path / "raw_input"
    raw_input_dir.mkdir()
    sample_file = raw_input_dir / "sample.txt"
    sample_file.write_text("This is a test document about peanut allergy.")

    # Mock LLMClient to simulate connection error
    mocker.patch("classes.llm_client.LLMClient.query", return_value="Error: Connection failed")

    # Run the pipeline up to response generation step with a sample file
    step01_ingest_documents(Args(input_filename="sample.txt"))
    step02_generate_embeddings(Args(input_filename="sample.txt"))
    step03_store_vectors(Args(input_filename="sample.txt"))
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        step05_generate_response(Args(
            input_filename="sample.txt",
            query_args="What is peanut allergy?",
            use_rag=True
        ))
    finally:
        sys.stdout = sys_stdout

    # The expected error message should be in the output
    output = captured_output.getvalue()
    assert "Error: Connection failed" in output, "Expected error message not found in output"
