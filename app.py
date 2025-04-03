import streamlit as st
import boto3
import faiss
import numpy as np
from google.generativeai import GenerativeModel
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer

# --- Secrets and Configuration ---
S3_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
S3_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME "]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]

# S3 Paths for FAISS Indexes (replace with your actual paths)
PROFORMA_FAISS_PATH = "faiss_indexes/proforma_faiss_index/"  # e.g., "faiss/proforma_index.faiss"
PO_FAISS_PATH = "faiss_indexes/po_faiss_index/"  # e.g., "faiss/po_index.faiss"

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# Initialize Gemini model
gemini_model = GenerativeModel("gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Hugging Face embeddings model
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HF_TOKEN)


# --- FAISS Index Loading Functions ---
def load_faiss_index(s3_key):
    local_path = f"/tmp/{os.path.basename(s3_key)}"
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    index = faiss.read_index(local_path)
    return index


def get_faiss_metadata(s3_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix)
    files = response.get("Contents", [])
    total_count = len(files)
    recent_file = max(files, key=lambda x: x["LastModified"], default=None)
    return total_count, recent_file["LastModified"] if recent_file else None


# --- RAG Logic ---
def query_rag(index, query, top_k=3):
    query_embedding = embeddings_model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    # Placeholder: Replace with actual document retrieval logic
    retrieved_docs = "Retrieved context from FAISS (replace with actual docs)"
    response = gemini_model.generate_content(f"Query: {query}\nContext: {retrieved_docs}").text
    return response


# --- Streamlit App ---
st.set_page_config(page_title="Sales Team RAG Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Stylish UI
st.markdown("""
    <style>
    /* General Styling */
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stApp { background-color: #f5f7fa; color: #333; }
    h1 { color: #2c3e50; font-size: 2.5em; font-weight: 600; }
    h2 { color: #34495e; font-size: 1.8em; }

    /* Sidebar */
    .css-1d391kg { background-color: #34495e; color: white; }
    .stRadio > label { color: white; font-size: 1.1em; }
    .stButton > button { background-color: #3498db; color: white; border-radius: 8px; }
    .stButton > button:hover { background-color: #2980b9; }

    /* Chat Messages */
    .chat-message { padding: 15px; margin: 10px 0; border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .chat-message.user { background-color: #3498db; color: white; }
    .chat-message.assistant { background-color: #ecf0f1; color: #2c3e50; }

    /* Input Box */
    .stTextInput > div > input { border: 2px solid #3498db; border-radius: 8px; padding: 10px; }

    /* Stats Section */
    .stats-box { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>Control Panel</h2>", unsafe_allow_html=True)
    mode = st.radio("Select Mode", ["Proforma", "PO"], key="mode_select")

    # Load FAISS index based on mode
    if mode == "Proforma":
        faiss_index = load_faiss_index(PROFORMA_FAISS_PATH)
        s3_prefix = os.path.dirname(PROFORMA_FAISS_PATH) + "/"
    else:
        faiss_index = load_faiss_index(PO_FAISS_PATH)
        s3_prefix = os.path.dirname(PO_FAISS_PATH) + "/"

    # FAISS Index Stats
    st.markdown("<h2>Index Stats</h2>", unsafe_allow_html=True)
    total_files, last_updated = get_faiss_metadata(s3_prefix)
    st.markdown(f"""
        <div class='stats-box'>
            <p><strong>Total Files:</strong> {total_files}</p>
            <p><strong>Last Updated:</strong> {last_updated.strftime('%Y-%m-%d %H:%M:%S') if last_updated else 'N/A'}</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat History
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Chat 1": []}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Chat 1"

    st.markdown("<h2>Chat History</h2>", unsafe_allow_html=True)
    if st.button("New Chat"):
        new_chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_chat_id] = []
        st.session_state.current_chat = new_chat_id

    for chat_id in st.session_state.chat_sessions.keys():
        if st.button(chat_id, key=chat_id):
            st.session_state.current_chat = chat_id

# Main Interface
st.markdown(f"<h1>Sales Team RAG Assistant - {mode} Mode</h1>", unsafe_allow_html=True)
st.write("Ask anything about Proforma or PO documents with ease!")

# Chat Container
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_sessions[st.session_state.current_chat]:
        with st.chat_message(message["role"]):
            st.markdown(f"<div class='chat-message {message['role']}'>{message['content']}</div>",
                        unsafe_allow_html=True)

# User Input
user_input = st.chat_input("Type your question here (e.g., 'What’s the status of PO #123?')")
if user_input:
    with chat_container:
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message user'>{user_input}</div>", unsafe_allow_html=True)

    response = query_rag(faiss_index, user_input)
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)

    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": user_input})
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": response})