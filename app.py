import streamlit as st
import boto3
import os
import tempfile
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import toml

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
try:
    secrets = toml.load(SECRETS_FILE_PATH)
    S3_BUCKET = "kalika-rag"  # Ensure this matches the indexer script
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"  # Base path (no trailing slash)
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Model will be downloaded from Hugging Face
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"  # Or other suitable Gemini model
    GEMINI_API_KEY = secrets["gemini_api_key"]
except FileNotFoundError:
    st.error(f"Secrets file not found at {SECRETS_FILE_PATH}. App cannot run.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret key in {SECRETS_FILE_PATH}: {e}. App cannot run.")
    st.stop()

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize S3 client ---
@st.cache_resource  # Cache S3 client resource across reruns
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        # Quick check to verify credentials
        s3.list_buckets()
        logging.info("S3 client initialized successfully.")
        return s3
    except Exception as e:
        logging.error(f"Error initializing S3 client: {str(e)}")
        st.error(f"Failed to connect to S3. Check AWS credentials and permissions. Error: {e}")
        return None

s3_client = get_s3_client()

# --- Initialize Embeddings Model ---
@st.cache_resource  # Cache embeddings model
def get_embeddings_model():
    try:
        # Model will be downloaded from Hugging Face Hub if not cached
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,  # Downloads BAAI/bge-base-en-v1.5 automatically
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available on deployment server
            encode_kwargs={'normalize_embeddings': True}  # Match indexer settings
        )
        logging.info(f"Embeddings model {EMBEDDING_MODEL} loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings model {EMBEDDING_MODEL}. Error: {e}")
        logging.error(f"Failed to load embeddings model: {e}")
        return None

embeddings = get_embeddings_model()

# --- Initialize Gemini LLM ---
@st.cache_resource  # Cache LLM model
def get_gemini_model():
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,  # Lower temperature for more factual answers based on context
            convert_system_message_to_human=True  # Good practice for some models
        )
        logging.info(f"Gemini model {GEMINI_MODEL} initialized.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini model {GEMINI_MODEL}. Check API Key. Error: {e}")
        logging.error(f"Failed to initialize Gemini model: {e}")
        return None

gemini_model = get_gemini_model()

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)  # Cache the loaded index for 1 hour
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    """
    Downloads the FAISS index files (index.faiss, index.pkl) from S3
    to a temporary local directory and loads them.
    Uses Streamlit's caching.
    Requires allow_dangerous_deserialization=True for FAISS.load_local.
    """
    if not _s3_client or not _embeddings:
        st.error("S3 client or embeddings model not initialized. Cannot load index.")
        return None

    # Define the specific file keys
    s3_index_key = f"{prefix}.faiss"
    s3_pkl_key = f"{prefix}.pkl"

    try:
        # Create a temporary directory that persists for the cached resource
        temp_dir = tempfile.mkdtemp()
        local_index_path = os.path.join(temp_dir, "index.faiss")
        local_pkl_path = os.path.join(temp_dir, "index.pkl")

        logging.info(f"Attempting to download index from s3://{bucket}/{prefix}")

        # Download the files
        _s3_client.download_file(bucket, s3_index_key, local_index_path)
        _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
        logging.info(f"Successfully downloaded index files to {temp_dir}")

        # Load the FAISS index from the temporary directory
        vector_store = FAISS.load_local(
            temp_dir,
            _embeddings,
            allow_dangerous_deserialization=True  # Required for loading
        )
        logging.info("FAISS index loaded successfully into memory.")
        return vector_store

    except _s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            st.error(f"FAISS index files not found at s3://{bucket}/{prefix}. Please run the indexing script.")
            logging.error(f"FAISS index files not found at s3://{bucket}/{prefix}.")
        else:
            st.error(f"Error downloading FAISS index from S3: {e}")
            logging.error(f"S3 ClientError downloading index: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None

# --- Querying Functions ---
def query_faiss_index(vector_store, query_text, k=30, use_mmr=False):
    """
    Query the FAISS index. Returns a list of LangChain Document objects.
    k: Number of results to return.
    use_mmr: Set to True to use Maximal Marginal Relevance for diverse results.
    """
    if not vector_store:
        return []
    try:
        search_kwargs = {'k': k}
        if use_mmr:
            results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k*4)
        else:
            results = vector_store.similarity_search(query_text, k=k)

        logging.info(f"Retrieved {len(results)} chunks using {'mmr' if use_mmr else 'similarity'} search for query: '{query_text}'")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS index: {str(e)}")
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    """
    Generate a response using the Gemini LLM, providing retrieved documents as context.
    """
    if not llm:
        return "LLM model is not available."

    if retrieved_docs:
        # Prepare context
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based only on the provided context documents.
        Synthesize the information from all the context sections below to provide a comprehensive and accurate answer to the user's query.
        If the answer cannot be fully found in the context, state clearly what information is available and what is missing. Do not make assumptions or use external knowledge.

        Context Documents:
        ---
        {context}
        ---
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        logging.info(f"Generating response for query: '{query_text}' with {len(retrieved_docs)} context chunks.")
    else:
        messages = [
            SystemMessage(content="You are an AI assistant. No relevant context documents were found for the user's query about Proforma Invoices."),
            HumanMessage(content=query_text)
        ]
        logging.info(f"Generating response for query: '{query_text}' without context documents.")

    try:
        ai_response: AIMessage = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response from LLM: {e}")
        logging.error(f"LLM invocation error: {e}", exc_info=True)
        return "Sorry, I encountered an error while generating the response."

# --- Streamlit UI ---
st.title("📄 Proforma Invoice Query Assistant")
st.markdown("Ask questions about the proforma invoices processed from email attachments.")

# Load resources only if prerequisites are met
if not s3_client or not embeddings or not gemini_model:
    st.error("Application cannot start due to initialization errors. Check logs.")
    st.stop()

# Load FAISS index (cached)
with st.spinner("Loading knowledge base index from S3... Please wait."):
    vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)

if vector_store:
    st.success("Knowledge base index loaded successfully!")
else:
    st.error("Failed to load the knowledge base index. Querying is disabled.")
    st.stop()

# Input area
st.markdown("---")
query_text = st.text_input("Enter your query:", placeholder="e.g., What is the total amount for invoice [filename]? or List all products in [filename].")

# Default query parameters
k_results = 25  # Default K value
use_mmr_search = False  # Default search type

if query_text:
    # 1. Query FAISS index
    retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

    # 2. Generate LLM response
    with st.spinner("Thinking..."):
        response = generate_llm_response(gemini_model, query_text, retrieved_docs)

    # 3. Display response
    st.markdown("### Response:")
    st.markdown(response)
    st.markdown("---")

    # Optional: Display retrieved context
    if retrieved_docs:
        with st.expander("Show Retrieved Context Snippets"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"*Snippet {i+1} (Source: {doc.metadata.get('source', 'N/A') if hasattr(doc, 'metadata') else 'N/A'})*")
                st.text_area(f"snippet_{i}", doc.page_content, height=150, key=f"snippet_{i}")
    else:
        st.info("No relevant snippets were found in the knowledge base for this query.")