# turn off chromadb sending stats back
export ANONYMIZED_TELEMETRY=False

BASEDIR="C:\Users\phamq\OneDrive\Desktop\school stuff\691\A04\RAG_project\A04_RAG"


# ----------------------------------
#  Step 01: Ingest: parse pdf or text files into cleaned text
# ----------------------------------
# python "$BASEDIR/main.py" step01_ingest --input_filename  "P1 - Group 1 - Proposal.pdf"
python "$BASEDIR/main.py" step01_ingest --input_filename all

# ----------------------------------
#  Step 02: Generate Embeddings from the cleaned text files
# ----------------------------------
# python "$BASEDIR/main.py" step02_generate_embeddings --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
python "$BASEDIR/main.py" step02_generate_embeddings --input_filename all

# ----------------------------------
#  Step 03: Store the cleaned text and embeddings in a vector db
# ----------------------------------
# python "$BASEDIR/main.py" step03_store_vectors  --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
python "$BASEDIR/main.py" step03_store_vectors --input_filename  all

# ----------------------------------
#  Step 04: Retrieve chunks of text and similarity scores
# ----------------------------------
python "$BASEDIR/main.py" step04_retrieve_chunks --query_args "Tell me about peanut allergies"

# ----------------------------------
#  Step 05: Run LLM Queries with and without RAG
#  	If the parameter "--use_rag" is not provided, RAG is not performed
# ----------------------------------
QUERY="Tell me about peanut allergies"
# python "$BASEDIR/main.py" step05_generate_response  --query_args "$QUERY"
python "$BASEDIR/main.py" step05_generate_response  --query_args "$QUERY"  --use_rag
