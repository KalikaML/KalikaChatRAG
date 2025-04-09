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
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"

try:
    secrets = toml.load(SECRETS_FILE_PATH)
    # Core application settings
    S3_BUCKET = "kalika-rag"  # Ensure this matches the indexer script
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"  # Base path (no trailing slash)
    MODEL_DIRECTORY = "BAAI/BAAI-bge-base-en-v1.5"
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
    st.write(f"Input Hash (from typed password): {input_password_hash}")  # Keep requested debug output
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
        ai_response: AIMessage = llm(messages)  # Type hint for clarity
        logging.info("LLM response generated successfully.")
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response from the LLM: {e}")
        logging.error(f"Error generating LLM response: {e}", exc_info=True)
        return "An error occurred while generating the response."


# --- Authentication Decorator ---
def streamlit_auth(func):
    """Authentication decorator for Streamlit apps."""

    def wrapper(*args, **kwargs):
        # Check if authentication is already done
        if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
            func(*args, **kwargs)  # Run the decorated function if authenticated
            return  # Exit the wrapper after running func

        # --- Login Form ---
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if verify_password(username, password):
                # --- Authentication successful ---
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                st.session_state['name'] = get_user_info(username)

                st.success(f"Welcome, {st.session_state['name']}!")

                # Refresh the page to run the main app
                st.rerun()  # Important: rerun the script to show the authenticated view
            else:
                # --- Authentication failed ---
                st.error("Invalid username or password")
                st.session_state['authentication_status'] = False  # Ensure it's explicitly set
        else:
            # Initial state: not authenticated
            st.session_state['authentication_status'] = False
            st.stop()  # Halt execution until login

    return wrapper


# =============================================================================
# New Chat History Functionality (Add BEFORE existing Streamlit UI code)
# =============================================================================

def initialize_chat_session():
    """Initialize or reset the current chat session"""
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = {
            'id': str(uuid4()),
            'messages': [],
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    if 'chats' not in st.session_state:
        st.session_state.chats = []

    if 'follow_up_questions' not in st.session_state:
        st.session_state.follow_up_questions = []


# =============================================================================
# Main Application with Authentication
# =============================================================================

@streamlit_auth
def main():
    s3_client = get_s3_client()
    embeddings_model = get_embeddings_model()
    llm = get_gemini_model()
    vector_store = download_and_load_faiss_index(s3_client, embeddings_model, S3_BUCKET, S3_PROFORMA_INDEX_PATH)

    # --- New Chat History Sidebar ---
    with st.sidebar:
        st.header("Chat History")

        # New Chat Button
        if st.button("➕ New Chat"):
            # Archive current chat if not empty
            if st.session_state.get('current_chat') and len(st.session_state.current_chat['messages']) > 0:
                st.session_state.chats.append(st.session_state.current_chat)
            initialize_chat_session()
            st.rerun()

        # Display chat history
        if 'chats' in st.session_state:
            for chat in reversed(st.session_state.chats):
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    if chat['messages']:  # Check if chat has messages
                        first_message = chat['messages'][0]['content']
                        chat_title = f"💬 {first_message[:30]}..."
                    else:
                        chat_title = "💬 Empty Chat"
                    if st.button(chat_title, key=chat['id']):
                        st.session_state.current_chat = chat
                        st.rerun()
                with col2:
                    st.caption(chat['created_at'][-8:])

        st.markdown("---")

    # --- Main Chat Interface ---
    st.title("Proforma Invoice Assistant")

    # Initialize chat session
    initialize_chat_session()

    # Display chat messages
    if 'current_chat' in st.session_state and st.session_state.current_chat:
        for msg in st.session_state.current_chat['messages']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Display follow-up questions if available
                if msg.get("follow_ups"):
                    st.markdown("**Follow-up Questions:**")
                    for question in msg["follow_ups"]:
                        if st.button(question, key=f"followup_{question[:20]}"):
                            # Handle follow-up question click
                            st.session_state.user_input = question
                            user_query = question  # Assign question to user_query
                            retrieved_docs = query_faiss_index(vector_store, user_query)
                            ai_response = generate_llm_response(llm, user_query, retrieved_docs)

                            follow_up_prompt = SystemMessage(content=f"""
                                Generate 3 concise follow-up questions that a user might ask after this response. 
                                Return only the questions as a bullet point list, nothing else.
                                Response: {ai_response}
                            """)
                            follow_ups = llm([follow_up_prompt]).content.split("\n")[:3]

                            st.session_state.current_chat['messages'].extend([
                                {"role": "user", "content": user_query},
                                {"role": "assistant", "content": ai_response, "follow_ups": follow_ups}
                            ])
                            st.rerun()

    # --- Existing query input and processing ---
    if user_query := st.chat_input("Ask about proforma invoices..."):
        retrieved_docs = query_faiss_index(vector_store, user_query)
        ai_response = generate_llm_response(llm, user_query, retrieved_docs)

        # --- New Follow-up Question Generation ---
        follow_up_prompt = SystemMessage(content=f"""
            Generate 3 concise follow-up questions that a user might ask after this response. 
            Return only the questions as a bullet point list, nothing else.
            Response: {ai_response}
        """)
        follow_ups = llm([follow_up_prompt]).content.split("\n")[:3]

        # Store message with follow-ups
        st.session_state.current_chat['messages'].extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response, "follow_ups": follow_ups}
        ])

        # Store follow-ups in session state
        st.session_state.follow_up_questions = follow_ups

        # Rerun to update UI
        st.rerun()


# --- Run the app ---
if __name__ == "__main__":
    main()
