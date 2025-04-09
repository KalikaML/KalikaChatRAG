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
from datetime import datetime

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
        return False
    stored_hashed_password = CREDENTIALS[username]["password"]
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()
    return input_password_hash == stored_hashed_password


def get_user_info(username):
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None


# --- Resource Initialization ---
@st.cache_resource
def get_s3_client():
    try:
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        s3.list_buckets()
        return s3
    except Exception as e:
        st.error(f"Failed to connect to S3: {e}")
        return None


@st.cache_resource
def get_embeddings_model():
    if not os.path.isdir(MODEL_DIRECTORY):
        st.error(f"Model directory not found: {os.path.abspath(MODEL_DIRECTORY)}")
        return None
    try:
        os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('.')
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_DIRECTORY,
            model_kwargs={'device': 'cpu', "local_files_only": True},
            encode_kwargs={'normalize_embeddings': True}
        )
        embeddings.embed_query("Test query")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        return None


@st.cache_resource
def get_gemini_model():
    try:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        return None


@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _s3_client or not _embeddings:
        return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")
            local_pkl_path = os.path.join(temp_dir, "index.pkl")
            _s3_client.download_file(bucket, f"{prefix}.faiss", local_index_path)
            _s3_client.download_file(bucket, f"{prefix}.pkl", local_pkl_path)
            return FAISS.load_local(
                folder_path=temp_dir,
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None


# --- Query Functions ---
def query_faiss_index(vector_store, query_text, k=15, use_mmr=False):
    if not vector_store:
        return []
    try:
        if use_mmr:
            return vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        return vector_store.similarity_search(query_text, k=k)
    except Exception as e:
        st.error(f"Error querying FAISS index: {e}")
        return []


def generate_llm_response(llm, query_text, retrieved_docs, chat_history=None):
    if not llm:
        return "LLM model not available."

    messages = []
    if chat_history:
        messages.extend(chat_history)

    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""You are a professional assistant specializing in Proforma Invoices. 
        Use only the provided context and chat history to answer.
        Context:
        ---
        {context}
        ---"""
        messages.extend([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ])
    else:
        messages.append(HumanMessage(content=query_text + "\n\nNo context found."))

    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"Error generating response: {e}"


def generate_followup_questions(query_text, response):
    return [
        f"What details support '{query_text}'?",
        f"Can you provide more specifics about '{query_text}'?",
        f"What else should I know about '{query_text}'?"
    ]


# --- UI Components ---
def login_page():
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("🔒 Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button", use_container_width=True):
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = get_user_info(username)
                st.session_state.chat_history = {}
                st.session_state.current_chat = "Chat 1"
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown('</div>', unsafe_allow_html=True)


def main_app():
    # Custom CSS for professional look
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            max-width: 80%;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: auto;
        }
        .ai-message {
            background-color: #f5f5f5;
            margin-right: auto;
        }
        .followup-btn {
            margin: 0.2rem;
            padding: 0.3rem 0.8rem;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Proforma Assistant")
        st.write(f"Welcome, {st.session_state.name}")

        # Chat history management
        if st.button("New Chat", key="new_chat"):
            chat_num = len(st.session_state.chat_history) + 1
            st.session_state.current_chat = f"Chat {chat_num}"
            st.session_state.chat_history[st.session_state.current_chat] = []
            st.rerun()

        st.subheader("Chat History")
        for chat_name in st.session_state.chat_history.keys():
            if st.button(chat_name, key=f"hist_{chat_name}"):
                st.session_state.current_chat = chat_name

        if st.button("Logout", key="logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("📄 Proforma Invoice Assistant")

        # Initialize resources
        s3_client = get_s3_client()
        embeddings = get_embeddings_model()
        gemini_model = get_gemini_model()
        vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)

        # Chat display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history.get(st.session_state.current_chat, []):
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>',
                                unsafe_allow_html=True)

        # Input form
        with st.form(key="query_form", clear_on_submit=True):
            query_text = st.text_area("Enter your query:", height=100)
            submit_button = st.form_submit_button("Send")

        if submit_button and query_text and vector_store:
            # Add user message to history
            st.session_state.chat_history[st.session_state.current_chat].append({
                "role": "user",
                "content": query_text,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

            # Process query
            with st.spinner("Processing..."):
                retrieved_docs = query_faiss_index(vector_store, query_text)
                response = generate_llm_response(
                    gemini_model,
                    query_text,
                    retrieved_docs,
                    [(SystemMessage(content=msg["content"]) if msg["role"] == "system"
                      else HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"]))
                     for msg in st.session_state.chat_history[st.session_state.current_chat]]
                )

                # Add AI response to history
                st.session_state.chat_history[st.session_state.current_chat].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            st.rerun()

    with col2:
        st.subheader("Follow-up Questions")
        if st.session_state.chat_history.get(st.session_state.current_chat):
            last_query = next(
                (msg["content"] for msg in reversed(st.session_state.chat_history[st.session_state.current_chat])
                 if msg["role"] == "user"), "")
            last_response = next(
                (msg["content"] for msg in reversed(st.session_state.chat_history[st.session_state.current_chat])
                 if msg["role"] == "assistant"), "")
            if last_query:
                for question in generate_followup_questions(last_query, last_response):
                    if st.button(question, key=f"followup_{question}", help="Click to ask"):
                        st.session_state.chat_history[st.session_state.current_chat].append({
                            "role": "user",
                            "content": question,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.rerun()


# --- Main Entry Point ---
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
        st.session_state.chat_history = {}
        st.session_state.current_chat = "Chat 1"

    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()