import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from main import (
    step01_ingest_documents,
    step02_generate_embeddings,
    step03_store_vectors,
    step04_retrieve_relevant_chunks,
    step05_generate_response
)

# A simple class to mock the argparse arguments object
# Separating this out as its own script to avoid having a process locking access to ChromaDB files
class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, name):
        return None

def main():
    if len(sys.argv) < 2:
        print("Error: Missing use_rag argument.")
        sys.exit(1)
        
    use_rag = sys.argv[1]
    
    test_query = "What is peanut allergy?"
    print(f"Subprocess: Starting pipeline for benchmark with test query: {test_query}")
    
    if use_rag:
        # Run Step 1
        step01_ingest_documents(MockArgs(input_filename="all"))
    
        # Run Step 2
        step02_generate_embeddings(MockArgs(input_filename="all"))
    
        # Run Step 3
        step03_store_vectors(MockArgs(input_filename="all"))
    
    # Run Step 4+5
    # step04_retrieve_relevant_chunks(MockArgs(input_filename="all", use_rag=use_rag, query_args=test_query))
    step05_generate_response(MockArgs(input_filename="all", use_rag=use_rag, query_args=test_query))
    
    print(f"Subprocess: Finished pipeline.")

if __name__ == "__main__":
    main()