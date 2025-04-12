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
    S3_BUCKET = "kalika-rag"
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
    MODEL_DIRECTORY = "BAAI/BAAI-bge-base-en-v1.5"
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"
    GEMINI_API_KEY = secrets["gemini_api_key"]

    # Authentication credentials
    if "credentials" in secrets:
        CREDENTIALS = secrets["credentials"]["usernames"]
    else:
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

    stored_hashed_password = CREDENTIALS[username]["password"]
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()
    is_match = input_password_hash == stored_hashed_password
    if not is_match:
        logging.warning(f"Password mismatch for user: {username}")
    return is_match

def get_user_info(username):
    """Get user info for a given username"""
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None

# --- Initialize S3 client ---
@st.cache_resource
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        s3.list_buckets()
        logging.info("S3 client initialized successfully.")
        return s3
    except Exception as e:
        logging.error(f"Error initializing S3 client: {str(e)}")
        st.error(f"Failed to connect to S3. Check AWS credentials and permissions. Error: {e}")
        return None

# --- Initialize Embeddings Model ---
@st.cache_resource
def get_embeddings_model():
    model_path = MODEL_DIRECTORY
    if not os.path.isdir(model_path):
        st.error(f"Local model directory not found at '{os.path.abspath(model_path)}'. "
                 f"Please ensure the directory '{MODEL_DIRECTORY}' exists.")
        logging.error(f"Model directory {model_path} not found.")
        return None

    try:
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        logging.info(f"Set TRANSFORMERS_CACHE to: {cache_dir}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': 'cpu',
                "local_files_only": True
            },
            encode_kwargs={'normalize_embeddings': True}
        )

        _ = embeddings.embed_query("Test query")
        logging.info(f"Embeddings model '{model_path}' loaded successfully.")
        return embeddings

    except Exception as e:
        st.error(f"Failed to load embeddings model from '{model_path}'. Error: {e}")
        logging.error(f"Failed to load embeddings model: {e}", exc_info=True)
        return None

# --- Initialize Gemini LLM ---
@st.cache_resource
def get_gemini_model():
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        logging.info(f"Gemini model {GEMINI_MODEL} initialized.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini model {GEMINI_MODEL}. Check API Key. Error: {e}")
        logging.error(f"Failed to initialize Gemini model: {e}")
        return None

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _embeddings:
        st.error("Embeddings model failed to load. Cannot load FAISS index.")
        logging.error("Attempted to load FAISS index, but embeddings are not available.")
        return None
    if not _s3_client:
        st.error("S3 client not initialized. Cannot load index.")
        return None

    s3_index_key = f"{prefix}.faiss"
    s3_pkl_key = f"{prefix}.pkl"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")
            local_pkl_path = os.path.join(temp_dir, "index.pkl")
            logging.info(f"Downloading index from s3://{bucket}/{prefix}")
            _s3_client.download_file(bucket, s3_index_key, local_index_path)
            _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
            logging.info(f"Downloaded index files to {temp_dir}")
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("FAISS index loaded successfully.")
            return vector_store

    except _s3_client.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        st.error(f"Error downloading FAISS index: {e}")
        logging.error(f"S3 ClientError: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None

# --- Querying Functions ---
def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
    if not vector_store:
        logging.warning("query_faiss_index called but vector_store is None.")
        return []
    try:
        search_kwargs = {'k': k}
        search_type = 'similarity'
        if use_mmr:
            search_type = 'mmr'
        logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")
        if use_mmr:
            results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        else:
            results = vector_store.similarity_search(query_text, k=k)
        logging.info(f"Retrieved {len(results)} chunks.")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        logging.error("generate_follow_up_questions called but llm is None.")
        return []
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
    try:
        follow_up_prompt = f"""Based on the user query and response, generate 3 specific, relevant follow-up questions.
        Questions should be clear, concise, and related to invoices.

        Previous Query: {query_text}

        Response: {response_text}

        {f'Context (excerpt): {context[:500]}...' if context else ''}

        Provide three follow-up questions in a list format, one per line.
        """
        messages = [
            SystemMessage(content="You are an AI assistant generating follow-up questions about invoices."),
            HumanMessage(content=follow_up_prompt)
        ]
        ai_response: AIMessage = llm.invoke(messages)
        questions = []
        for line in ai_response.content.strip().split('\n'):
            clean_line = line.strip().lstrip("0123456789.-*• ").strip()
            if clean_line and '?' in clean_line:
                questions.append(clean_line)
        return questions[:3]
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}", exc_info=True)
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    if not llm:
        logging.error("generate_llm_response called but llm is None.")
        return "LLM model is not available."
    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""You are an AI assistant answering questions about Proforma Invoices based only on the provided context.
        Answer accurately using the context below. Quote relevant snippets if applicable.
        If the answer is not in the context, state what is available and what cannot be answered.

        Context:
        ---
        {context}
        ---
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        logging.info(f"Generating response for query: '{query_text}' with {len(retrieved_docs)} chunks.")
    else:
        messages = [
            SystemMessage(content="You are an AI assistant. No context documents were found."),
            HumanMessage(content=query_text + "\n\nNo relevant documents found. State that you cannot answer based on the knowledge base.")
        ]
        logging.info(f"Generating response for query: '{query_text}' without context.")
    try:
        ai_response: AIMessage = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logging.error(f"LLM error: {e}", exc_info=True)
        return "Error generating response."

# --- Login Page ---
def login_page():
    st.title("📄 Proforma Invoice Assistant - Login")
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
                st.rerun()
            else:
                st.error("Invalid username or password")

# --- Main Application ---
def main_app():
    # Initialize chat sessions
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
        st.session_state.current_chat_id = "chat_1"
        st.session_state.chat_counter = 1
        st.session_state.chat_sessions["chat_1"] = {
            'query_history': [],
            'response_history': [],
            'follow_up_questions': []
        }

    # Sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        st.markdown("### Chat Sessions")
        # Display chat sessions as clickable buttons
        for chat_id in st.session_state.chat_sessions.keys():
            if st.button(f"Chat {chat_id.split('_')[1]}", key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        # New Chat Button
        if st.button("New Chat"):
            st.session_state.chat_counter += 1
            new_chat_id = f"chat_{st.session_state.chat_counter}"
            st.session_state.chat_sessions[new_chat_id] = {
                'query_history': [],
                'response_history': [],
                'follow_up_questions': []
            }
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        # Logout Button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.session_state.chat_sessions = {}
            st.rerun()

    # Main app UI
    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown("Ask questions about proforma invoices processed from email attachments.")

    # Initialize resources
    s3_client = get_s3_client()
    embeddings = get_embeddings_model()
    gemini_model = get_gemini_model()

    # Resource status
    s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
    embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
    gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"

    with st.status("Initializing resources...", expanded=False) as status_container:
        st.write(s3_status)
        st.write(embeddings_status)
        st.write(gemini_status)
        if not s3_client or not embeddings or not gemini_model:
            st.error("Core components failed to initialize.")
            status_container.update(label="Initialization Failed!", state="error")
            st.stop()
        else:
            st.write("Loading Knowledge Base Index...")
            vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
            if vector_store:
                st.write("✅ Knowledge Base Index Loaded")
                status_container.update(label="Initialization Complete!", state="complete")
            else:
                st.write("❌ Knowledge Base Index Failed")
                status_container.update(label="Initialization Failed!", state="error")
                st.error("Failed to load knowledge base index.")
                st.stop()

    # --- Chat Interface ---
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]

    # Ensure chat session has required keys
    if 'query_history' not in current_chat:
        current_chat['query_history'] = []
    if 'response_history' not in current_chat:
        current_chat['response_history'] = []
    if 'follow_up_questions' not in current_chat:
        current_chat['follow_up_questions'] = []

    # Container for chat history
    chat_container = st.container()

    # Display chat history
    with chat_container:
        if current_chat['query_history']:
            for i in range(len(current_chat['query_history'])):
                st.markdown(f"**You:**")
                st.markdown(f"> {current_chat['query_history'][i]}")
                st.markdown(f"**Assistant:**")
                st.markdown(current_chat['response_history'][i])
                st.markdown("---")

    # Handle follow-up question click
    if 'follow_up_clicked' in st.session_state and st.session_state.follow_up_clicked:
        query_text = st.session_state.follow_up_clicked
        st.session_state.follow_up_clicked = None
        process_query(query_text, vector_store, gemini_model, current_chat, chat_container)
    else:
        # Input box for new query
        query_text = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the total amount for invoice [filename]?",
            key=f"query_input_{st.session_state.current_chat_id}",
            disabled=not vector_store
        )
        if query_text:
            process_query(query_text, vector_store, gemini_model, current_chat, chat_container)

    # Display follow-up questions
    if current_chat['follow_up_questions']:
        st.markdown("### Suggested Follow-Up Questions:")
        cols = st.columns(len(current_chat['follow_up_questions']))
        for i, question in enumerate(current_chat['follow_up_questions']):
            if cols[i].button(question, key=f"follow_up_{i}_{st.session_state.current_chat_id}"):
                st.session_state.follow_up_clicked = question
                st.rerun()

def process_query(query_text, vector_store, gemini_model, current_chat, chat_container):
    """Process a user query and append to chat history"""
    k_results = 15
    use_mmr_search = False

    with st.spinner(f"Searching knowledge base..."):
        retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

    with st.spinner("Generating response..."):
        response = generate_llm_response(gemini_model, query_text, retrieved_docs)

    # Append to history
    current_chat['query_history'].append(query_text)
    current_chat['response_history'].append(response)

    # Generate follow-up questions
    with st.spinner("Generating follow-up questions..."):
        follow_up_questions = generate_follow_up_questions(gemini_model, query_text, response, retrieved_docs)
        current_chat['follow_up_questions'] = follow_up_questions

    # Update chat display
    with chat_container:
        st.markdown(f"**You:**")
        st.markdown(f"> {query_text}")
        st.markdown(f"**Assistant:**")
        st.markdown(response)
        st.markdown("---")

# --- Main Entry Point ---
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()