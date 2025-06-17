import pytest
import requests
import json
from classes.llm_client import LLMClient

@pytest.fixture
def llm_client():
    """Returns an instance of LLMClient for testing."""
    return LLMClient(llm_api_url="http://fake-url/v1/chat/completions", llm_model_name="test-llm")

def test_llm_client_query_success(mocker, llm_client):
    """
    Tests a successful query where the API returns a valid response.
    """
    # GIVEN a mocked successful API response
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    expected_content = "This is the LLM's answer."
    api_json_response = {
        "choices": [{
            "message": {
                "content": expected_content
            }
        }]
    }
    mock_response.json.return_value = api_json_response
    mocker.patch("requests.post", return_value=mock_response)
    
    # WHEN the query method is called
    prompt = "What is RAG?"
    response = llm_client.query(prompt)
    
    # THEN the response should be the content from the mocked API call
    assert response == expected_content
    # AND requests.post should have been called with the correct data
    requests.post.assert_called_once()
    call_args = requests.post.call_args
    sent_payload = json.loads(call_args.kwargs['data'])
    assert sent_payload['messages'][0]['content'] == prompt

def test_llm_client_query_api_error(mocker, llm_client):
    """
    Tests how the client handles a network error from the requests library.
    """
    # GIVEN that the requests.post call will raise an exception
    mocker.patch("requests.post", side_effect=requests.exceptions.RequestException("Network Error"))
    
    # WHEN the query method is called
    response = llm_client.query("This will fail")
    
    # THEN a user-friendly error message should be returned
    assert response == "Error: Could not connect to the LLM."