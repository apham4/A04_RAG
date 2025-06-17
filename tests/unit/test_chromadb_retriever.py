import pytest
from classes.chromadb_retriever import ChromaDBRetriever

@pytest.fixture
def mocked_retriever(mocker):
    """
    Provides a ChromaDBRetriever instance with all external dependencies mocked.
    """
    # Mock the ChromaDB client and collection
    mock_collection = mocker.Mock()
    mocker.patch("chromadb.PersistentClient").return_value.get_or_create_collection.return_value = mock_collection
    
    # Instantiate the retriever (it will now use the mocks)
    retriever = ChromaDBRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="fake-collection",
        vectordb_dir="/fake/dir"
    )
    # Attach the mock collection to the instance for later use in tests
    retriever.collection = mock_collection
    return retriever

def test_retriever_query(mocked_retriever):
    """
    Tests that the retriever correctly queries the DB and parses the results.
    """
    # GIVEN a mocked response from ChromaDB
    mock_db_results = {
        "ids": [["doc1", "doc2"]],
        "metadatas": [[
            {"text": "This is context one.", "source": "source1"},
            {"text": "This is context two.", "source": "source2"}
        ]],
        "distances": [[0.25, 0.8]] # one above, one below threshold
    }
    mocked_retriever.collection.query.return_value = mock_db_results
    mocked_retriever.score_threshold = 0.5 # Set threshold for the test
    
    # WHEN a query is performed
    results = mocked_retriever.query("some search phrase", top_k=2)
    
    # THEN only the result passing the threshold should be returned
    assert len(results) == 1
    assert results[0]["id"] == "doc2"
    assert results[0]["score"] == 0.8
    assert results[0]["context"] == "This is context two."