import pytest
import json
from pathlib import Path
from classes.document_ingestor import DocumentIngestor

@pytest.fixture
def ingestor_environment(tmp_path):
    """A fixture to set up temporary directories and a DocumentIngestor instance."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # We can use a real model for tokenization as it's a local operation
    ingestor = DocumentIngestor(
        file_list=[],
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return ingestor, input_dir, output_dir

def test_extract_text_from_pdf(mocker, ingestor_environment):
    """Tests that text is correctly extracted from a mocked PDF."""
    ingestor, _, _ = ingestor_environment
    
    # GIVEN a mock for pdfplumber
    mock_pdf_page1 = mocker.Mock()
    mock_pdf_page1.extract_text.return_value = "This is the first page."
    mock_pdf_page2 = mocker.Mock()
    mock_pdf_page2.extract_text.return_value = "This is the second page."
    
    mock_pdf = mocker.MagicMock()
    mock_pdf.pages = [mock_pdf_page1, mock_pdf_page2]
    
    # Configure the 'with' statement context manager
    mocker.patch('pdfplumber.open', return_value=mocker.MagicMock(__enter__=mocker.MagicMock(return_value=mock_pdf), __exit__=mocker.MagicMock()))

    # WHEN text is extracted from a fake PDF path
    extracted_text = ingestor._extract_text_from_pdf("fake.pdf")
    
    # THEN the text from all pages should be concatenated
    assert "This is the first page." in extracted_text
    assert "This is the second page." in extracted_text
    assert extracted_text == "This is the first page.\nThis is the second page."

def test_process_files_chunks_and_saves_correctly(ingestor_environment):
    """
    Tests the main process_files method to ensure it reads a file,
    chunks the content, and saves it as a JSON file.
    """
    ingestor, input_dir, output_dir = ingestor_environment
    
    # GIVEN a sample text file in the input directory
    long_text = "This is a sentence. " * 100 # Create text long enough to be chunked
    sample_file = input_dir / "sample.txt"
    sample_file.write_text(long_text)
    
    ingestor.file_list = ["sample.txt"]
    
    # WHEN process_files is called
    ingestor.process_files()
    
    # THEN a JSON file with chunks should be created in the output directory
    expected_output = output_dir / "sample_cleaned_chunks.json"
    assert expected_output.exists(), "Chunk file was not created"
    
    with open(expected_output, 'r') as f:
        data = json.load(f)
    
    assert isinstance(data, list), "Output should be a list of chunks"
    assert len(data) > 1, "Expected the text to be split into multiple chunks"
    assert data[0].startswith("this is a sentence."), "Chunk content is incorrect"