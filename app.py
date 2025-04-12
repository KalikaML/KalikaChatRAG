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
    is_match = input_password_hash == stored_hashed_password
    if not is_match:
        logging.warning(f"Password mismatch for user: {username}")
    return is_match

def get_user_info(username):
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None

# --- Initialize S3 Client ---
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
        st.error(f"Model directory not found at '{os.path.abspath(model_path)}'.")
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
        st.error(f"Failed to load embeddings model. Error: {e}")
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
        st.error("Cannot load FAISS index: missing embeddings or S3 client.")
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
def query_faiss_index(vector_store, query_text, k=15, use_mmr=False):
    if not vector_store:
        logging.warning("No vector store available.")
        return []
    try:
        logging.info(f"Querying FAISS index with query: '{query_text}'")
        results = vector_store.similarity_search(query_text, k=k) if not use_mmr else \
                 vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        logging.info(f"Retrieved {len(results)} documents.")
        return results
    except Exception as e:
        st.error(f"Error querying index: {e}")
        logging.error(f"Error querying FAISS index: {e}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        logging.error("No LLM available for follow-up questions.")
        return []
    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]]) if retrieved_docs else ""
    try:
        prompt = f"""Based on the query and response, generate 3 relevant follow-up questions about invoices.
        Query: {query_text}
        Response: {response_text}
        {f'Context: {context[:500]}...' if context else ''}
        Provide three questions, one per line."""
        messages = [
            SystemMessage(content="Generate follow-up questions about invoices."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        questions = [line.strip().lstrip("0123456789.-*• ") for line in response.content.strip().split('\n')
                    if line.strip() and '?' in line]
        logging.info(f"Generated {len(questions)} follow-up questions.")
        return questions[:3]
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}", exc_info=True)
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    if not llm:
        logging.error("No LLM available.")
        return "LLM unavailable."
    try:
        if retrieved_docs:
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            prompt = f"""Answer the query about Proforma Invoices using only the provided context.
            Context:
            {context}
            Query: {query_text}
            If the answer isn't fully in the context, state what's available and note limitations."""
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=query_text)
            ]
        else:
            messages = [
                SystemMessage(content="No context documents found."),
                HumanMessage(content=f"{query_text}\n\nState that no answer is possible due to lack of context.")
            ]
        response = llm.invoke(messages)
        logging.info(f"Generated response for query: '{query_text}'")
        return response.content
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
            background-color: #1c1e24;
            padding: 32px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            max-width: 400px;
            margin: 80px auto;
            color: #d4d7dd;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput > div > input {
            background-color: #24262d;
            color: #d4d7dd;
            border: 1px solid #35373f;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        .stButton > button {
            background-color: #5b7bf7;
            color: #ffffff;
            border-radius: 6px;
            padding: 12px;
            width: 100%;
            border: none;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 14px;
            transition: background-color 0.2s;
        }
        .stButton > button:hover {
            background-color: #435fd7;
        }
        .stError { color: #f87171; font-family: 'Inter', sans-serif; }
        .stSuccess { color: #34d399; font-family: 'Inter', sans-serif; }
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
            background-color: #0f1115;
            font-family: 'Inter', sans-serif;
            color: #d4d7dd;
        }
        .sidebar .sidebar-content {
            background-color: #181a20;
            padding: 24px;
            box-shadow: 2px 0 12px rgba(0,0,0,0.2);
            color: #d4d7dd;
        }
        .chat-container {
            background-color: #1c1e24;
            border-radius: 8px;
            padding: 32px;
            margin: 24px auto;
            max-width: 960px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
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
            color: #d4d7dd;
        }
        .stTextInput > div > input {
            background-color: #24262d;
            color: #d4d7dd;
            border: 1px solid #35373f;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }
        .stButton > button {
            background-color: #5b7bf7;
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
            background-color: #435fd7;
        }
        .sidebar-button-container, .follow-up-container {
            background-color: #24262d;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 16px;
        }
        .logout-button {
            float: right;
            background-color: #dc2626;
            padding: 8px 16px;
            font-size: 13px;
        }
        .logout-button:hover {
            background-color: #b91c1c;
        }
        h1 {
            color: #ffffff;
            text-align: center;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        h3 {
            color: #858b98;
            font-size: 15px;
            font-weight: 500;
            margin-bottom: 12px;
        }
        hr {
            border-color: #35373f;
            margin: 16px 0;
        }
        .stSpinner > div {
            border-color: #5b7bf7 transparent transparent transparent;
        }
        .stStatus > div {
            background-color: #24262d;
            color: #d4d7dd;
            border: 1px solid #35373f;
            border-radius: 6px;
            padding: 12px;
            font-size: 14px;
        }
        .stError { color: #f87171; }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"<p style='color: #d4d7dd; font-size: 15px; font-weight: 500; margin-bottom: 12px;'>Welcome, {st.session_state.name}</p>", unsafe_allow_html=True)
        st.markdown('<div style="overflow: hidden;">', unsafe_allow_html=True)
        if st.button("Logout", key="logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.session_state.chat_sessions = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

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

        # New Chat Button
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

        # Follow-Up Questions
        current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
        if current_chat.get('follow_up_questions'):
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
            if not all([s3_client, embeddings, gemini_model]):
                st.error("Core components failed to initialize.")
                status.update(label="Initialization Failed!", state="error")
                st.stop()
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

        # Ensure chat session keys exist
        current_chat.setdefault('query_history', [])
        current_chat.setdefault('response_history', [])
        current_chat.setdefault('follow_up_questions', [])

        # Display chat history
        for i, (query, response) in enumerate(zip(current_chat['query_history'], current_chat['response_history'])):
            st.markdown('<div class="chat-message">', unsafe_allow_html=True)
            st.markdown(f"**Question {i+1}:**")
            st.markdown(f"> {query}")
            st.markdown(f"**Answer:**")
            st.markdown(response)
            st.markdown("<hr>")
            st.markdown('</div>', unsafe_allow_html=True)

        # Query input
        query_text = st.session_state.get('follow_up_clicked', '')
        if query_text:
            st.session_state.follow_up_clicked = None
        query_input = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the total amount for invoice [filename]?",
            key="query_input",
            value=query_text,
            disabled=not vector_store
        )

        if query_input and vector_store:
            with st.spinner("Searching knowledge base..."):
                retrieved_docs = query_faiss_index(vector_store, query_input, k=15, use_mmr=False)
            with st.spinner("Generating answer..."):
                response = generate_llm_response(gemini_model, query_input, retrieved_docs)
            with st.spinner("Generating suggestions..."):
                follow_up_questions = generate_follow_up_questions(gemini_model, query_input, response, retrieved_docs)

            # Update chat session
            current_chat['query_history'].append(query_input)
            current_chat['response_history'].append(response)
            current_chat['follow_up_questions'] = follow_up_questions

            # Display new response
            st.markdown('<div class="chat-message">', unsafe_allow_html=True)
            st.markdown(f"**Question {len(current_chat['query_history'])}:**")
            st.markdown(f"> {query_input}")
            st.markdown(f"**Answer:**")
            st.markdown(response)
            st.markdown("<hr>")
            st.markdown('</div>', unsafe_allow_html=True)

            st.rerun()  # Refresh to show follow-up questions
        elif query_input and not vector_store:
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