import streamlit as st
import boto3
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import threading
import time
import logging
import google.generativeai as genai

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load secrets from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_key=AWS_SECRET_KEY,
)

# Initialize embeddings - use cache for better performance
@st.cache_resource
def get_embeddings():
    logger.info(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

embeddings = get_embeddings()

# Gemini LLM implementation
class GeminiLLM:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        logger.info("Initialized Gemini LLM")

    def generate(self, prompt, temperature=0.7, max_length=None):
        generation_config = {"temperature": temperature}
        if max_length:
            generation_config["max_output_tokens"] = max_length
        try:
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {str(e)}")
            return f"Error: Unable to generate response - {str(e)}"

@st.cache_resource
def get_llm():
    return GeminiLLM(api_key=GEMINI_API_KEY)

gemini_llm = get_llm()

# Enhanced prompt template
prompt_template = PromptTemplate(
    input_variables=["documents", "question", "doc_count"],
    template="""
    You are an assistant designed to support a sales team. Using the provided information from {doc_count} documents (including proforma invoices and purchase orders), answer the user's question with accurate, concise, and actionable details in a well-structured bullet-point format.

    Make your response as comprehensive as needed to fully address the query - don't artificially limit length.

    Information from documents: {documents}
    Question: {question}

    Important: Your response must ONLY include the answer in bullet points. Do NOT include:
    - The documents or source information you used
    - Any preamble or introduction to your answer
    - Any mention of the context you're referencing
    - This instruction itself

    Respond directly with bullet points:
    - [Relevant detail addressing the user's question]
    - [Additional relevant detail, if applicable]
    - [Further relevant detail, if applicable]
    """
)

# Load FAISS index from S3 with logging
@st.cache_resource
def load_faiss_index_from_s3(index_path):
    logger.info(f"Loading FAISS index from {index_path}")
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            try:
                logger.info(f"Downloading {s3_key} to {local_path}")
                s3_client.download_file(S3_BUCKET, s3_key, local_path)
            except Exception as e:
                logger.error(f"Failed to download {s3_key}: {str(e)}")
                raise
        try:
            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Successfully loaded FAISS index from {index_path}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise

# Preload indexes
@st.cache_resource
def get_all_indexes():
    proforma_index = load_faiss_index_from_s3(PROFORMA_INDEX_PATH)
    po_index = load_faiss_index_from_s3(PO_INDEX_PATH)
    # Test the indexes
    test_query = "test query"
    proforma_docs = proforma_index.similarity_search(test_query, k=1)
    po_docs = po_index.similarity_search(test_query, k=1)
    logger.info(f"Proforma test result: {len(proforma_docs)} docs retrieved")
    logger.info(f"PO test result: {len(po_docs)} docs retrieved")
    return {
        "Proforma Invoices": proforma_index,
        "Purchase Orders": po_index,
        "Combined": None
    }

# Count new files in S3 folder
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    if 'Contents' not in response:
        return 0
    new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
    return new_files

# Retrieve documents from combined indexes
def retrieve_from_all_indexes(query, k=10):
    proforma_index = st.session_state.all_indexes["Proforma Invoices"]
    po_index = st.session_state.all_indexes["Purchase Orders"]
    proforma_docs = proforma_index.similarity_search(query, k=k // 2)
    po_docs = po_index.similarity_search(query, k=k // 2)
    all_docs = proforma_docs + po_docs
    logger.info(f"Retrieved {len(all_docs)} docs for query '{query}':")
    for i, doc in enumerate(all_docs):
        logger.info(f"Doc {i+1}: {doc.page_content[:100]}...")
    return all_docs

# Background thread for file count updates
def background_refresh():
    while True:
        try:
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time.sleep(300)  # Refresh every 5 minutes
        except Exception as e:
            logger.error(f"Error in background refresh: {e}")
            time.sleep(60)

# Custom CSS
def load_css():
    return """
        <style>
        .stApp { background-color: #1E1E1E; color: #E0E0E0; }
        .chat-message { padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 16px; }
        .user-message { background-color: #2D2D2D; color: #BB86FC; text-align: right; border: 1px solid #BB86FC; }
        .bot-message { background-color: #333333; color: #E0E0E0; text-align: left; border: 1px solid #03DAC6; }
        .sidebar .sidebar-content { background-color: #252525; padding: 20px; color: #E0E0E0; }
        .context-panel { background-color: #252525; padding: 15px; border-radius: 8px; border: 1px solid #555555; margin-top: 10px; }
        .context-title { color: #03DAC6; font-weight: bold; margin-bottom: 10px; }
        .context-item { background-color: #333333; padding: 10px; border-radius: 5px; margin-bottom: 8px; font-size: 14px; border-left: 3px solid #BB86FC; }
        .document-source { font-size: 12px; color: #BB86FC; margin-bottom: 5px; }
        .stTextInput > div > div > input { background-color: #333333; color: #E0E0E0; border: 1px solid #555555; border-radius: 5px; }
        .stButton > button { background-color: #BB86FC; color: #1E1E1E; border: none; border-radius: 5px; }
        .stButton > button:hover { background-color: #03DAC6; color: #1E1E1E; }
        h1, h2, h3 { color: #BB86FC; }
        .stSpinner > div > div { color: #03DAC6; }
        .loading-message { background-color: #2D2D2D; color: #03DAC6; padding: 10px; border-radius: 5px; text-align: center; margin: 20px 0; }
        .doc-stats { background-color: #252525; padding: 8px 12px; border-radius: 4px; display: inline-block; margin-right: 10px; border-left: 3px solid #03DAC6; }
        </style>
    """

# Main Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.markdown(load_css(), unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "context_documents" not in st.session_state:
        st.session_state.context_documents = []
    if "indexes_loaded" not in st.session_state:
        st.session_state.indexes_loaded = False
    if "proforma_new" not in st.session_state:
        st.session_state.proforma_new = 0
    if "po_new" not in st.session_state:
        st.session_state.po_new = 0
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if "background_thread_started" not in st.session_state:
        st.session_state.background_thread_started = False
    if "response_length" not in st.session_state:
        st.session_state.response_length = "Auto"

    # Start background thread
    if not st.session_state.background_thread_started:
        thread = threading.Thread(target=background_refresh, daemon=True)
        thread.start()
        st.session_state.background_thread_started = True

    # Two-column layout
    col1, col2 = st.columns([7, 3])

    with col1:
        st.title("RAG Chatbot")
        st.write("Ask anything related to proforma invoices and purchase orders.")

        # Load indexes
        if not st.session_state.indexes_loaded:
            with st.spinner("Loading FAISS indexes..."):
                st.session_state.all_indexes = get_all_indexes()
                st.session_state.indexes_loaded = True

        # User input
        user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")

        input_col1, input_col2, input_col3 = st.columns([2, 2, 1])
        with input_col1:
            option = st.selectbox("Data Source", ["Combined (Both Sources)", "Proforma Invoices Only", "Purchase Orders Only"])
        with input_col2:
            doc_count = st.slider("Number of documents to retrieve", min_value=3, max_value=50, value=10)
        with input_col3:
            response_length = st.selectbox("Response Length", ["Auto", "Concise", "Detailed"])
            st.session_state.response_length = response_length

        if st.button("Send") and user_input:
            st.session_state.chat_history.append(("user", user_input))
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="loading-message">Retrieving documents and generating response...</div>',
                unsafe_allow_html=True
            )

            try:
                # Retrieve documents
                if option == "Combined (Both Sources)":
                    documents = retrieve_from_all_indexes(user_input, k=doc_count)
                elif option == "Proforma Invoices Only":
                    vector_store = st.session_state.all_indexes["Proforma Invoices"]
                    documents = vector_store.similarity_search(user_input, k=doc_count)
                else:
                    vector_store = st.session_state.all_indexes["Purchase Orders"]
                    documents = vector_store.similarity_search(user_input, k=doc_count)

                st.session_state.context_documents = documents

                # Set max_length based on response length
                max_length = None
                if response_length == "Concise":
                    max_length = 300
                elif response_length == "Detailed":
                    max_length = 2000

                # Generate response
                prompt_instance = prompt_template.format(
                    documents=[doc.page_content for doc in documents],
                    question=user_input,
                    doc_count=len(documents)
                )
                response = gemini_llm.generate(prompt_instance, max_length=max_length)

                st.session_state.chat_history.append(("bot_metadata", f"Using {len(documents)} documents"))
                st.session_state.chat_history.append(("bot", response))
                response_placeholder.empty()
                st.experimental_rerun()

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.session_state.chat_history.append(("error", error_message))
                response_placeholder.empty()
                st.experimental_rerun()

        # Display chat history
        for sender, message in reversed(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot_metadata":
                st.markdown(f'<div class="doc-stats">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot":
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "error":
                st.error(message)

    with col2:
        st.markdown("<h3>Retrieved Context</h3>", unsafe_allow_html=True)
        if st.session_state.context_documents:
            st.markdown('<div class="context-panel">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="context-title">Source Documents ({len(st.session_state.context_documents)})</div>',
                unsafe_allow_html=True)
            for i, doc in enumerate(st.session_state.context_documents):
                source_type = "Unknown"
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    if PROFORMA_FOLDER.lower() in doc.metadata['source'].lower():
                        source_type = "Proforma"
                    elif PO_FOLDER.lower() in doc.metadata['source'].lower():
                        source_type = "PO"
                st.markdown(
                    f'<div class="document-source">Document {i + 1} - {source_type}</div>' +
                    f'<div class="context-item">{doc.page_content}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="context-panel">No context retrieved yet. Ask a question to see relevant documents.</div>',
                unsafe_allow_html=True)

        st.markdown("<h3>Stats</h3>", unsafe_allow_html=True)
        st.write(f"Proforma Invoices: {st.session_state.proforma_new} new files")
        st.write(f"Purchase Orders: {st.session_state.po_new} new files")
        st.write(f"Last Updated: {st.session_state.last_updated}")
        if st.button("Refresh Stats"):
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.experimental_rerun()
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.context_documents = []
            st.experimental_rerun()

# Utility to recreate FAISS index (run separately if needed)
def recreate_faiss_index(folder_prefix, index_path):
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader(f"s3://{S3_BUCKET}/{folder_prefix}")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
        for file_name in ["index.faiss", "index.pkl"]:
            s3_client.upload_file(
                os.path.join(temp_dir, file_name),
                S3_BUCKET,
                f"{index_path}{file_name}"
            )
    logger.info(f"Recreated FAISS index for {folder_prefix} at {index_path}")

if __name__ == "__main__":
    main()
    # Uncomment to recreate indexes if needed (run separately)
    # recreate_faiss_index(PROFORMA_FOLDER, PROFORMA_INDEX_PATH)
    # recreate_faiss_index(PO_FOLDER, PO_INDEX_PATH)