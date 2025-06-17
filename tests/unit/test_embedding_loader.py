import pytest
import json
from classes.embedding_loader import EmbeddingLoader

@pytest.fixture
def loader_environment(tmp_path):
    """Sets up a temporary environment for the EmbeddingLoader."""
    cleaned_dir = tmp_path / "cleaned"
    embeddings_dir = tmp_path / "embeddings"
    vectordb_dir = tmp_path / "db"
    for d in [cleaned_dir, embeddings_dir, vectordb_dir]:
        d.mkdir()

    # GIVEN sample chunk and embedding files
    chunks = ["This is chunk one.", "This is chunk two."]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    
    (cleaned_dir / "stem_cleaned_chunks.json").write_text(json.dumps(chunks))
    (embeddings_dir / "stem_embeddings.json").write_text(json.dumps(embeddings))
    
    return {
        "cleaned_dir": cleaned_dir,
        "embeddings_dir": embeddings_dir,
        "vectordb_dir": vectordb_dir,
        "file_list": ["stem_cleaned_chunks.json"]
    }

def test_loader_processes_files_and_adds_to_db(mocker, loader_environment):
    """
    Tests that the loader correctly prepares and adds data to a mocked ChromaDB.
    """
    env = loader_environment
    
    # GIVEN a mocked ChromaDB client and collection
    mock_collection = mocker.Mock()
    mock_client = mocker.patch('chromadb.PersistentClient').return_value
    mock_client.get_or_create_collection.return_value = mock_collection

    # Instantiate the loader
    loader = EmbeddingLoader(
        cleaned_text_file_list=env["file_list"],
        cleaned_text_dir=env["cleaned_dir"],
        embeddings_dir=env["embeddings_dir"],
        vectordb_dir=env["vectordb_dir"],
        collection_name="test_collection"
    )
    
    # WHEN process_files is called
    loader.process_files()

    # THEN the collection's `add` method should be called with correctly formatted data
    mock_collection.add.assert_called_once()
    
    # Inspect the arguments passed to the mock
    call_args = mock_collection.add.call_args.kwargs
    
    expected_ids = ["stem::chunk_0", "stem::chunk_1"]
    expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    expected_metadatas = [
        {"text": "This is chunk one.", "source": "stem", "chunk_index": 0},
        {"text": "This is chunk two.", "source": "stem", "chunk_index": 1}
    ]
    
    assert call_args['ids'] == expected_ids
    assert call_args['embeddings'] == expected_embeddings
    assert call_args['metadatas'] == expected_metadatas