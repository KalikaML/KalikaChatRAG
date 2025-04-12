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
    S3_BUCKET = "kalika-rag"
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
    MODEL_DIRECTORY = "BAAI/BAAI-bge-base-en-v1.5"
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"
    GEMINI_API_KEY = secrets["gemini_api_key"]

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
    if username not in CREDENTIALS:
        logging.warning(f"Login attempt for non-existent user: {username}")
        return False
    stored_hashed_password = CREDENTIALS[username]["password"]
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()
    return input_password_hash == stored_hashed_password

def get_user_info(username):
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
        st.error(f"Failed to connect to S3. Error: {e}")
        return None

# --- Initialize Embeddings Model ---
@st.cache_resource
def get_embeddings_model():
    model_path = MODEL_DIRECTORY
    if not os.path.isdir(model_path):
        st.error(f"Local model directory not found at '{os.path.abspath(model_path)}'.")
        logging.error(f"Model directory {model_path} not found.")
        return None
    try:
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu', "local_files_only": True},
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
        st.error(f"Failed to initialize Gemini model. Error: {e}")
        logging.error(f"Failed to initialize Gemini model: {e}")
        return None

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _embeddings or not _s3_client:
        st.error("Embeddings or S3 client not initialized.")
        return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")
            local_pkl_path = os.path.join(temp_dir, "index.pkl")
            _s3_client.download_file(bucket, f"{prefix}.faiss", local_index_path)
            _s3_client.download_file(bucket, f"{prefix}.pkl", local_pkl_path)
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("FAISS index loaded successfully.")
            return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None

# --- Querying Functions ---
def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
    if not vector_store:
        return []
    try:
        if use_mmr:
            results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        else:
            results = vector_store.similarity_search(query_text, k=k)
        logging.info(f"Retrieved {len(results)} chunks for query: '{query_text}'")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        return []
    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]]) if retrieved_docs else ""
    try:
        follow_up_prompt = f"""Based on the query and response, generate 3 specific follow-up questions about invoices.

        Query: {query_text}
        Response: {response_text}
        {f'Context: {context[:500]}...' if context else ''}

        Provide three questions, one per line.
        """
        messages = [
            SystemMessage(content="Generate follow-up questions about invoices."),
            HumanMessage(content=follow_up_prompt)
        ]
        ai_response: AIMessage = llm.invoke(messages)
        questions = [line.strip().lstrip("0123456789.-*• ").strip() for line in ai_response.content.strip().split('\n')]
        questions = [q for q in questions if q and '?' in q][:3]
        return questions
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}", exc_info=True)
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    if not llm:
        return "LLM model is not available."
    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""Answer questions about Proforma Invoices using only the provided context.
        Context:
        ---
        {context}
        ---
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
    else:
        messages = [
            SystemMessage(content="No context documents found."),
            HumanMessage(content=query_text + "\n\nNo documents found. State that you cannot answer.")
        ]
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
        st.markdown("### Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
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
        st.session_state.current_chat_id = None
        st.session_state.chat_counter = 0

    # Create a new chat if none exists or after clicking New Chat
    if not st.session_state.current_chat_id or 'new_chat_triggered' in st.session_state:
        st.session_state.chat_counter += 1
        new_chat_id = f"chat_{st.session_state.chat_counter}"
        st.session_state.chat_sessions[new_chat_id] = {
            'query_history': [],
            'response_history': [],
            'follow_up_questions': []
        }
        st.session_state.current_chat_id = new_chat_id
        if 'new_chat_triggered' in st.session_state:
            del st.session_state.new_chat_triggered

    # Sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        st.markdown("### Chat History")
        for chat_id in sorted(st.session_state.chat_sessions.keys()):
            if st.button(f"Chat {chat_id.split('_')[1]}", key=f"chat_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.session_state.query_input = ""  # Clear input
                st.rerun()
        if st.button("New Chat"):
            st.session_state.new_chat_triggered = True
            st.session_state.query_input = ""  # Clear input
            st.rerun()
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.session_state.chat_sessions = {}
            st.rerun()

    # Main UI
    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown("Ask questions about proforma invoices.")

    # Initialize resources
    s3_client = get_s3_client()
    embeddings = get_embeddings_model()
    gemini_model = get_gemini_model()
    vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)

    if not (s3_client and embeddings and gemini_model and vector_store):
        st.error("Failed to initialize resources.")
        st.stop()

    # Chat interface
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for i in range(len(current_chat.get('query_history', []))):
            st.markdown(f"**You:**")
            st.markdown(f"> {current_chat['query_history'][i]}")
            st.markdown(f"**Assistant:**")
            st.markdown(current_chat['response_history'][i])
            st.markdown("---")

    # Handle follow-up question
    if 'follow_up_query' in st.session_state and st.session_state.follow_up_query:
        query_text = st.session_state.follow_up_query
        process_query(query_text, vector_store, gemini_model, current_chat, chat_container)
        st.session_state.follow_up_query = None  # Clear after processing
        st.rerun()  # Refresh to show input box

    # Query input
    query_text = st.text_input(
        "Enter your query:",
        placeholder="e.g., What is the total amount for invoice [filename]?",
        key=f"query_input_{st.session_state.current_chat_id}",
        value=st.session_state.get('query_input', ''),
        disabled=not vector_store
    )
    st.session_state.query_input = ""  # Reset after capturing

    if query_text:
        process_query(query_text, vector_store, gemini_model, current_chat, chat_container)

    # Display follow-up questions
    if current_chat.get('follow_up_questions'):
        st.markdown("### Suggested Follow-Up Questions:")
        cols = st.columns(min(len(current_chat['follow_up_questions']), 3))
        for i, question in enumerate(current_chat['follow_up_questions']):
            if cols[i % len(cols)].button(question, key=f"follow_up_{i}_{st.session_state.current_chat_id}"):
                st.session_state.follow_up_query = question
                st.rerun()

def process_query(query_text, vector_store, gemini_model, current_chat, chat_container):
    k_results = 15
    use_mmr_search = False

    with st.spinner("Searching..."):
        retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

    with st.spinner("Generating response..."):
        response = generate_llm_response(gemini_model, query_text, retrieved_docs)

    # Update chat history
    current_chat.setdefault('query_history', []).append(query_text)
    current_chat.setdefault('response_history', []).append(response)

    # Generate follow-up questions
    with st.spinner("Generating follow-up questions..."):
        follow_up_questions = generate_follow_up_questions(gemini_model, query_text, response, retrieved_docs)
        current_chat['follow_up_questions'] = follow_up_questions

    # Display new message
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