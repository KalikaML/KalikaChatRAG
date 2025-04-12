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
        st.error(f"Local model directory not found at '{os.path.abspath(model_path)}'.")
        logging.error(f"Model directory {model_path} not found.")
        return None

    try:
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        logging.info(f"Set TRANSFORMERS_CACHE to: {cache_dir}")

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
        st.error(f"Failed to initialize Gemini model {GEMINI_MODEL}. Error: {e}")
        logging.error(f"Failed to initialize Gemini model: {e}")
        return None

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _embeddings:
        st.error("Embeddings model failed to load. Cannot load FAISS index.")
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
        logging.warning("query_faiss_index called but vector_store is None.")
        return []
    try:
        search_kwargs = {'k': k}
        search_type = 'similarity'
        if use_mmr:
            search_type = 'mmr'
        logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")
        results = vector_store.similarity_search(query_text, k=k) if not use_mmr else \
                 vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        logging.info(f"Retrieved {len(results)} chunks.")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {e}")
        logging.error(f"Error querying FAISS index: {e}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        logging.error("generate_follow_up_questions called but llm is None.")
        return []
    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]]) if retrieved_docs else ""
    try:
        follow_up_prompt = f"""Based on the user query and response, generate 3 specific, relevant follow-up questions.
        Query: {query_text}
        Response: {response_text}
        {f'Context: {context[:500]}...' if context else ''}
        Provide three questions in a list, one per line, related to invoices."""
        messages = [
            SystemMessage(content="You are an AI assistant generating follow-up questions about invoices."),
            HumanMessage(content=follow_up_prompt)
        ]
        ai_response = llm.invoke(messages)
        questions = [line.strip().lstrip("0123456789.-*• ") for line in ai_response.content.strip().split('\n')
                    if line.strip() and '?' in line]
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
        system_prompt = f"""You are an AI assistant answering questions about Proforma Invoices using only the provided context.
        Context:
        ---
        {context}
        ---
        Answer the query accurately, quoting relevant snippets if needed. If the answer isn't fully in the context, state what's available."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
    else:
        messages = [
            SystemMessage(content="No context documents found for the query."),
            HumanMessage(content=query_text + "\n\nState that you cannot answer based on the available knowledge base.")
        ]
    try:
        ai_response = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logging.error(f"LLM error: {e}", exc_info=True)
        return "Error generating response."

# --- Login Page ---
def login_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        .login-container {
            background-color: #1e2025;
            padding: 32px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            max-width: 400px;
            margin: 80px auto;
            color: #d9dce1;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput > div > input {
            background-color: #282a30;
            color: #d9dce1;
            border: 1px solid #3a3c43;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Inter', sans-serif;
        }
        .stButton > button {
            background-color: #4f74e3;
            color: #ffffff;
            border-radius: 6px;
            padding: 12px;
            width: 100%;
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        .stButton > button:hover {
            background-color: #3758c9;
        }
        .stError { color: #f87171; font-family: 'Inter', sans-serif; }
        .stSuccess { color: #2dd4bf; font-family: 'Inter', sans-serif; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='color: #ffffff; font-weight: 600; margin-bottom: 24px; text-align: center;'>Proforma Invoice Assistant</h2>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login"):
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = get_user_info(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Application ---
def main_app():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        .stApp {
            background-color: #121317;
            font-family: 'Inter', sans-serif;
            color: #d9dce1;
        }
        .sidebar .sidebar-content {
            background-color: #1b1d22;
            padding: 24px;
            box-shadow: 2px 0 12px rgba(0,0,0,0.25);
            color: #d9dce1;
        }
        .chat-container {
            background-color: #1e2025;
            border-radius: 8px;
            padding: 32px;
            margin: 24px auto;
            max-width: 960px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .chat-message {
            margin-bottom: 24px;
            line-height: 1.6;
        }
        .chat-message strong {
            color: #ffffff;
            font-weight: 600;
        }
        .chat-message > p, .chat-message > div {
            color: #d9dce1;
        }
        .stTextInput > div > input {
            background-color: #282a30;
            color: #d9dce1;
            border: 1px solid #3a3c43;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        .stButton > button {
            background-color: #4f74e3;
            color: #ffffff;
            border-radius: 6px;
            padding: 10px;
            margin: 4px 0;
            border: none;
            width: 100%;
            text-align: left;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .stButton > button:hover {
            background-color: #3758c9;
        }
        .sidebar-button-container, .follow-up-container {
            background-color: #282a30;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .logout-button {
            float: right;
            background-color: #e11d48;
            padding: 8px 16px;
            font-size: 13px;
        }
        .logout-button:hover {
            background-color: #be123c;
        }
        h1 {
            color: #ffffff;
            text-align: center;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        h3 {
            color: #8b919e;
            font-size: 15px;
            font-weight: 500;
            margin-bottom: 16px;
        }
        hr {
            border-color: #3a3c43;
            margin: 16px 0;
        }
        .stSpinner > div {
            border-color: #4f74e3 transparent transparent transparent;
        }
        .stStatus > div {
            background-color: #282a30;
            color: #d9dce1;
            border: 1px solid #3a3c43;
            border-radius: 6px;
            padding: 12px;
            font-size: 14px;
        }
        .stError { color: #f87171; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"<p style='color: #d9dce1; font-size: 15px; font-weight: 500; margin-bottom: 12px;'>Welcome, {st.session_state.name}</p>", unsafe_allow_html=True)
        st.markdown('<div style="overflow: hidden;">', unsafe_allow_html=True)
        if st.button("Logout", key="logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
            st.session_state.current_chat_id = "chat_1"
            st.session_state.chat_counter = 1
            st.session_state.chat_sessions["chat_1"] = {
                'query_history': [],
                'response_history': [],
                'follow_up_questions': []
            }

        with st.container():
            st.markdown('<div class="sidebar-button-container">', unsafe_allow_html=True)
            if st.button("New Chat", key="new_chat"):
                st.session_state.chat_counter += 1
                new_chat_id = f"chat_{st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_chat_id] = {
                    'query_history': [],
                    'response_history': [],
                    'follow_up_questions': []
                }
                st.session_state.current_chat_id = new_chat_id
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
        if 'follow_up_questions' in current_chat and current_chat['follow_up_questions']:
            with st.container():
                st.markdown('<div class="follow-up-container">', unsafe_allow_html=True)
                st.markdown("<h3>Suggested Questions</h3>", unsafe_allow_html=True)
                for i, question in enumerate(current_chat['follow_up_questions']):
                    if st.button(question, key=f"follow_up_{i}_{st.session_state.current_chat_id}"):
                        st.session_state.follow_up_clicked = question
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # Main UI
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.title("📄 Proforma Invoice Query Assistant")
        st.markdown("Query details about proforma invoices from email attachments.")

        # Resources
        s3_client = get_s3_client()
        embeddings = get_embeddings_model()
        gemini_model = get_gemini_model()

        s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
        embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
        gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"

        with st.status("Initializing resources...", expanded=False) as status:
            st.write(s3_status)
            st.write(embeddings_status)
            st.write(gemini_status)
            if not s3_client or not embeddings or not gemini_model:
                st.error("Core components failed to initialize.")
                status.update(label="Initialization Failed!", state="error")
                st.stop()
            else:
                st.write("Loading Knowledge Base...")
                vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
                if vector_store:
                    st.write("✅ Knowledge Base Loaded")
                    status.update(label="Initialization Complete!", state="complete")
                else:
                    st.error("Failed to load knowledge base.")
                    status.update(label="Initialization Failed!", state="error")
                    st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)

        if 'query_history' not in current_chat:
            current_chat['query_history'] = []
        if 'response_history' not in current_chat:
            current_chat['response_history'] = []

        # Chat history
        if current_chat['query_history']:
            for i in range(len(current_chat['query_history'])):
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**Question:**")
                st.markdown(f"> {current_chat['query_history'][i]}")
                st.markdown(f"**Answer:**")
                st.markdown(current_chat['response_history'][i])
                st.markdown("<hr>")
                st.markdown('</div>', unsafe_allow_html=True)

        # Query input
        if 'follow_up_clicked' in st.session_state and st.session_state.follow_up_clicked:
            query_text = st.session_state.follow_up_clicked
            st.session_state.follow_up_clicked = None
        else:
            query_text = st.text_input(
                "Enter your query:",
                placeholder="e.g., What is the total amount for invoice [filename]?",
                key="query_input",
                value='',
                disabled=not vector_store
            )

        k_results = 15
        use_mmr_search = False

        if query_text and vector_store:
            with st.spinner(f"Searching knowledge base (k={k_results})..."):
                retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

            with st.spinner("Generating answer..."):
                response = generate_llm_response(gemini_model, query_text, retrieved_docs)

            current_chat['query_history'].append(query_text)
            current_chat['response_history'].append(response)

            with st.spinner("Generating suggestions..."):
                follow_up_questions = generate_follow_up_questions(gemini_model, query_text, response, retrieved_docs)
                current_chat['follow_up_questions'] = follow_up_questions

            st.markdown('<div class="chat-message">', unsafe_allow_html=True)
            st.markdown("### Response:")
            st.markdown(response)
            st.markdown("<hr>")
            st.markdown('</div>', unsafe_allow_html=True)

        elif query_text and not vector_store:
            st.error("Cannot process query: knowledge base not loaded.")

        st.markdown('</div>', unsafe_allow_html=True)

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