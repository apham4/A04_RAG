import pytest
import json
from classes.embedding_preparer import EmbeddingPreparer

@pytest.fixture
def preparer_environment(tmp_path):
    """Sets up a temporary environment for the EmbeddingPreparer."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # GIVEN a sample chunk file
    sample_chunks = ["This is chunk one.", "This is chunk two."]
    chunk_file = input_dir / "doc1_cleaned_chunks.json"
    chunk_file.write_text(json.dumps(sample_chunks))
    
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "chunk_file_name": "doc1_cleaned_chunks.json",
        "sample_chunks": sample_chunks,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }

def test_preparer_generates_and_saves_embeddings(preparer_environment):
    """
    Tests that process_files correctly calls the embedding generator
    and saves the results.
    """
    # Instantiate the preparer
    preparer = EmbeddingPreparer(
        file_list=[preparer_environment["chunk_file_name"]],
        input_dir=preparer_environment["input_dir"],
        output_dir=preparer_environment["output_dir"],
        embedding_model_name=preparer_environment["embedding_model_name"]
    )
    
    # WHEN process_files is called
    preparer.process_files()
    
    # THEN an embeddings file should be created
    expected_output = preparer_environment["output_dir"] / "doc1_embeddings.json"
    assert expected_output.exists(), "Embeddings file was not created"
    
    with open(expected_output, 'r') as f:
        saved_embeddings = json.load(f)
        
    assert len(saved_embeddings) == len(preparer_environment["sample_chunks"]), "Incorrect number of embeddings saved"
    assert len(saved_embeddings[0]) == 384, "Embedding dimension is incorrect"