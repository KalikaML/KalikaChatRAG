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
import hashlib

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"

try:
    secrets = toml.load(SECRETS_FILE_PATH)
    # Core application settings
    S3_BUCKET = "kalika-rag"  # Ensure this matches the indexer script
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"  # Base path (no trailing slash)
    MODEL_DIRECTORY = "bge-base-en-v1.5"
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"  # Or other suitable Gemini model
    GEMINI_API_KEY = secrets["gemini_api_key"]

    # Authentication credentials
    # Using standard Streamlit secrets approach
    if "credentials" in secrets:
        CREDENTIALS = secrets["credentials"]["usernames"]
        # st.write(CREDENTIALS)
    else:
        # Default credentials if not in secrets (for development)
        CREDENTIALS = {
            "user1": {
                "name": "User",
                "email": "user@example.com",
                "password": hashlib.sha256("user@123".encode()).hexdigest()
            }
        }

except FileNotFoundError:
    st.error(f"Secrets file not found at {SECRETS_FILE_PATH}. App cannot run.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret key in {SECRETS_FILE_PATH}: {e}. App cannot run.")
    st.stop()

# --- Authentication Functions ---
def verify_password(username, password):
    """Verify the password for a given username"""
    if username not in CREDENTIALS:
        logging.warning(f"Login attempt for non-existent user: {username}")
        return False

    # Get the pre-hashed password stored in the credentials
    stored_hashed_password = CREDENTIALS[username]["password"]

    # Hash the password *provided by the user during login*
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()

    # --- Debugging Output (Optional - remove/log in production) ---
    # st.write(f"Username: {username}")
    # st.write(f"Stored Hash: {stored_hashed_password}")
    st.write(f"Input Hash (from typed password): {input_password_hash}") # Keep requested debug output
    # ---

    # Compare the hash of the input password with the stored hash
    is_match = input_password_hash == stored_hashed_password
    if not is_match:
        logging.warning(f"Password mismatch for user: {username}")
    return is_match

def get_user_info(username):
    """Get user info for a given username"""
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None


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


# --- Initialize Embeddings Model ---
@st.cache_resource  # Cache embeddings model
def get_embeddings_model():
    """
    Loads the HuggingFace embeddings model from a local directory.
    Sets the TRANSFORMERS_CACHE environment variable to help the library find the files.
    """
    model_path = MODEL_DIRECTORY  # Relative path to the local model directory

    # --- Crucial Step: Check if local model directory exists ---
    if not os.path.isdir(model_path):
        st.error(f"Local model directory not found at '{os.path.abspath(model_path)}'. "
                 f"Please ensure the directory '{MODEL_DIRECTORY}' exists in the same "
                 f"location as the Streamlit script or provide the correct path.")
        logging.error(f"Model directory {model_path} not found.")
        return None

    try:
        # --- Set TRANSFORMERS_CACHE to the *parent* directory ---
        # This tells the library to look inside the current working directory
        # for the folder named MODEL_DIRECTORY.
        # Use abspath('.') to get the absolute path of the current working directory.
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        logging.info(f"Set TRANSFORMERS_CACHE to: {cache_dir}")

        # Now, initialize HuggingFaceEmbeddings. It should use the environment
        # variable along with local_files_only=True to find the model_path.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,  # Use the directory name as the model identifier
            model_kwargs={
                'device': 'cpu',  # Use CPU if no GPU available/configured
                "local_files_only": True  # Crucial: prevent download attempts
            },
            encode_kwargs={'normalize_embeddings': True}  # Recommended for BGE models
        )

        # Perform a dummy embedding to ensure the model loads correctly
        _ = embeddings.embed_query("Test query")

        logging.info(f"Embeddings model '{model_path}' loaded successfully from local directory.")
        return embeddings

    except Exception as e:
        st.error(f"Failed to load embeddings model from '{model_path}'. Error: {e}")
        logging.error(f"Failed to load embeddings model from '{model_path}': {e}", exc_info=True)
        # Provide more specific guidance based on the error if possible
        if "ConnectionError" in str(e) or "offline" in str(e):
            st.error("It seems the application tried to connect to Hugging Face Hub. "
                     "Ensure 'local_files_only=True' is correctly set and the "
                     f"'{MODEL_DIRECTORY}' contains all necessary model files (config.json, pytorch_model.bin/model.safetensors, tokenizer files etc.).")
        elif "snapshot" in str(e):
            st.error(
                f"The library could not find the model files within the expected structure in '{os.path.abspath(model_path)}' "
                f"even with TRANSFORMERS_CACHE set. Please verify the contents of the '{MODEL_DIRECTORY}' directory.")
        return None


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


# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)  # Cache the loaded index for 1 hour
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    """
    Downloads the FAISS index files (index.faiss, index.pkl) from S3
    to a temporary local directory and loads them.
    Uses Streamlit's caching.
    Requires allow_dangerous_deserialization=True for FAISS.load_local.
    """
    # --- Added Check: Ensure embeddings model is loaded ---
    if not _embeddings:
        st.error("Embeddings model failed to load. Cannot load FAISS index.")
        logging.error("Attempted to load FAISS index, but embeddings are not available.")
        return None
    # --- End Added Check ---

    if not _s3_client:
        st.error("S3 client not initialized. Cannot load index.")
        return None

    # Define the specific file keys based on how FAISS saves (no explicit folder)
    s3_index_key = f"{prefix}.faiss"
    s3_pkl_key = f"{prefix}.pkl"

    try:
        # Create a temporary directory that persists for the cached resource
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")  # FAISS expects these specific names
            local_pkl_path = os.path.join(temp_dir, "index.pkl")  # when loading from a directory

            logging.info(f"Attempting to download index from s3://{bucket}/{prefix} (.faiss and .pkl)")

            # Download the files
            _s3_client.download_file(bucket, s3_index_key, local_index_path)
            _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
            logging.info(f"Successfully downloaded index files to {temp_dir}")

            # Load the FAISS index from the temporary directory
            # Pass the directory path, not individual file paths
            vector_store = FAISS.load_local(
                folder_path=temp_dir,  # Pass the directory path
                embeddings=_embeddings,
                allow_dangerous_deserialization=True  # *** Absolutely required ***
            )
            logging.info("FAISS index loaded successfully into memory.")
            return vector_store  # The vector store is returned, temp_dir is cleaned up automatically after 'with' block

    except _s3_client.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            st.error(
                f"FAISS index files ({s3_index_key}, {s3_pkl_key}) not found in s3://{bucket}/. Please run the indexing script.")
            logging.error(f"FAISS index files not found at s3://{bucket}/{prefix}.")
        elif error_code == 'NoSuchBucket':
            st.error(f"S3 bucket '{bucket}' not found. Please check the S3_BUCKET name.")
            logging.error(f"S3 bucket '{bucket}' not found.")
        elif error_code in ['NoCredentialsError', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
            st.error(f"AWS S3 Authentication Error: {e}. Please check your AWS credentials in secrets.toml.")
            logging.error(f"S3 Authentication Error: {e}")
        else:
            st.error(f"Error downloading FAISS index from S3: {e}")
            logging.error(f"S3 ClientError downloading index: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        # Check if it's a deserialization issue
        if "Pickle" in str(e) or "deserialization" in str(e):
            st.error("Potential deserialization issue with the FAISS index (.pkl file). "
                     "Ensure the index was created with the same version of LangChain/FAISS "
                     "and that 'allow_dangerous_deserialization=True' is set.")
        return None


# --- Querying Functions ---
def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
    """
    Query the FAISS index. Returns a list of LangChain Document objects.
    k: Number of results to return. Start with 5-10.
    use_mmr: Set to True to use Maximal Marginal Relevance for potentially more diverse results.
    """
    if not vector_store:
        logging.warning("query_faiss_index called but vector_store is None.")
        return []
    try:
        search_kwargs = {'k': k}
        search_type = 'similarity'
        if use_mmr:
            search_type = 'mmr'
            # MMR specific defaults, can be tuned:
            # search_kwargs['fetch_k'] = 20 # Fetch more initially for MMR to select from
            # search_kwargs['lambda_mult'] = 0.5 # 0.5 balances similarity and diversity

        logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")

        # Direct similarity search:
        if use_mmr:
            # Increase fetch_k for MMR to have more documents to choose from for diversity
            results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        else:
            results = vector_store.similarity_search(query_text, k=k)

        logging.info(f"Retrieved {len(results)} chunks using {search_type} search.")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
        return []


def generate_llm_response(llm, query_text, retrieved_docs):
    """
    Generate a response using the Gemini LLM, providing retrieved documents as context.
    """
    if not llm:
        logging.error("generate_llm_response called but llm is None.")
        return "LLM model is not available."

    if retrieved_docs:
        # --- Context Preparation ---
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])  # Separate chunks clearly
        context_sources = ", ".join(list(set(doc.metadata.get('source', 'N/A') for doc in retrieved_docs if hasattr(doc,
                                                                                                                    'metadata') and 'source' in doc.metadata)))  # Get unique sources
        context_log_msg = f"Context from sources: {context_sources}" if context_sources else "Context from retrieved chunks."

        # --- Enhanced System Prompt ---
        system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based *only* on the provided context documents.
        Analyze the user's query and the following context documents carefully.
        Synthesize the information from the relevant parts of the context to provide a comprehensive and accurate answer.
        Quote relevant snippets or data points from the context to support your answer where appropriate.
        If the answer cannot be fully found within the provided context, clearly state what information is available and specify what parts of the query cannot be answered based on the context.
        Do not make assumptions or use any external knowledge outside of the provided context.

        Context Documents:
        ---
        {context}
        ---
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        logging.info(
            f"Generating response for query: '{query_text}' with {len(retrieved_docs)} context chunks. {context_log_msg}")

    else:
        # Handle case where no relevant documents were found
        messages = [
            SystemMessage(content="You are an AI assistant answering questions about Proforma Invoices. "
                                  "No relevant context documents were found in the knowledge base for the user's query."),
            HumanMessage(
                content=query_text + "\n\nSince no relevant documents were found, please state that you cannot answer the question based on the available knowledge base.")
        ]
        logging.info(f"Generating response for query: '{query_text}' without context documents.")

    try:
        ai_response: AIMessage = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response from LLM: {e}")
        logging.error(f"LLM invocation error: {e}", exc_info=True)
        return "Sorry, I encountered an error while generating the response."


# --- Login Page ---
def login_page():
    st.title("📄 Proforma Invoice Assistant - Login")

    # Center the login form with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Login to Access the System")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login", use_container_width=True)

        if login_button:
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = get_user_info(username)
                st.success("Login successful!")
                st.rerun()  # Refresh the page
            else:
                st.error("Invalid username or password")


# --- Main Application ---
def main_app():
    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()  # Refresh the page after logout

    # Main app UI
    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown("Ask questions about the proforma invoices processed from email attachments.")

    # Initialize resources
    s3_client = get_s3_client()
    embeddings = get_embeddings_model()
    gemini_model = get_gemini_model()

    # --- Resource Loading and Status ---
    s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
    embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
    gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"

    with st.status("Initializing resources...", expanded=False) as status_container:
        st.write(s3_status)
        st.write(embeddings_status)
        st.write(gemini_status)
        # Check if core components failed
        if not s3_client or not embeddings or not gemini_model:
            st.error("Core components failed to initialize. Application cannot proceed. Check logs for details.")
            status_container.update(label="Initialization Failed!", state="error")
            st.stop()  # Stop execution if core components fail
        else:
            # Load FAISS index only if core components are okay
            st.write("Loading Knowledge Base Index...")
            vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
            if vector_store:
                st.write("✅ Knowledge Base Index Loaded")
                status_container.update(label="Initialization Complete!", state="complete", expanded=False)
            else:
                st.write("❌ Knowledge Base Index Failed to Load")
                status_container.update(label="Initialization Failed!", state="error")
                st.error(
                    "Failed to load the knowledge base index. Querying is disabled. Check S3 path and permissions.")
                st.stop()  # Stop execution if index loading fails

    # --- Query Interface ---
    st.markdown("---")
    query_text = st.text_input("Enter your query:",
                               placeholder="e.g., What is the total amount for invoice [filename]? or List all products in [filename].",
                               key="query_input",  # Add key for potential state management
                               disabled=not vector_store)  # Disable input if index failed

    # Using fixed settings for simplicity:
    k_results = 15  # Increased default K value for potentially better context
    use_mmr_search = False  # Default search type

    if query_text and vector_store:  # Ensure vector_store is available
        # 1. Query FAISS index
        with st.spinner(f"Searching knowledge base for relevant info (k={k_results}, MMR={use_mmr_search})..."):
            retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

        # 2. Generate LLM response
        with st.spinner("🧠 Synthesizing answer using retrieved context..."):
            response = generate_llm_response(gemini_model, query_text, retrieved_docs)

        # 3. Display response
        st.markdown("### Response:")
        st.markdown(response)  # Use markdown for better formatting if the LLM provides it
        st.markdown("---")

        # 4. Optional: Display retrieved context
        if retrieved_docs:
            with st.expander("🔍 Show Retrieved Context Snippets"):
                st.markdown(f"Retrieved {len(retrieved_docs)} snippets:")
                for i, doc in enumerate(retrieved_docs):
                    # Try to get filename from metadata
                    source_info = "Unknown Source"
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"  # Default to N/A if 'source' key missing
                        # You could add more metadata here if available, e.g., page number
                        # source_info += f", Page: {doc.metadata.get('page', 'N/A')}"

                    st.text_area(
                        label=f"**Snippet {i + 1}** ({source_info})",  # Use label instead of markdown in text_area
                        value=doc.page_content,  # Use value for text_area content
                        height=150,
                        key=f"snippet_{i}",
                        disabled=True  # Make text area read-only
                    )
        else:
            st.info("No relevant snippets were found in the knowledge base for this query.")

    elif query_text and not vector_store:
        st.error("Cannot process query because the knowledge base index is not loaded.")


# --- Main Entry Point ---
def main():
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None

    # Show login page or main app based on authentication status
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()
