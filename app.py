import streamlit as st
import boto3
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from datetime import datetime
import threading
import time

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Load secrets from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)


# Initialize embeddings - use cache for better performance
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )


embeddings = get_embeddings()


# Initialize Gemini LLM with proper API implementation
class GeminiLLM:
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize the client properly here
        # This is a placeholder - replace with actual initialization
        # Example: from google.generativeai import GenerativeModel
        #          self.model = GenerativeModel("gemini-1.5-pro")

    def generate(self, prompt, temperature=0.7, max_length=512):
        # Replace with actual implementation
        # Example:
        # response = self.model.generate_content(
        #     prompt,
        #     generation_config={
        #         "temperature": temperature,
        #         "max_output_tokens": max_length,
        #     }
        # )
        # return response.text
        return f"Generated response for: {prompt}"  # Placeholder response


# Cache the LLM to avoid reinitializing
@st.cache_resource
def get_llm():
    return GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])


gemini_llm = get_llm()

# Prompt template for sales team queries
prompt_template = PromptTemplate(
    input_variables=["documents", "question"],
    template="""
    You are an assistant designed to support a sales team. Using the provided information from proforma invoices and purchase orders, answer the user's question with accurate, concise, and actionable details in a well-structured bullet-point format.

    Information: {documents}
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
    (Include as many bullet points as necessary to fully answer the question)
    """
)


# Function to load FAISS index from S3 - cached for performance
@st.cache_resource
def load_faiss_index_from_s3(index_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# Preload both indexes at startup to avoid loading delays
@st.cache_resource
def get_all_indexes():
    proforma_index = load_faiss_index_from_s3(PROFORMA_INDEX_PATH)
    po_index = load_faiss_index_from_s3(PO_INDEX_PATH)
    return {
        "Proforma Invoices": proforma_index,
        "Purchase Orders": po_index
    }


# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    if 'Contents' not in response:
        return 0
    new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
    return new_files


# Background thread to periodically refresh file counts
def background_refresh():
    while True:
        st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
        st.session_state.po_new = count_new_files(PO_FOLDER)
        st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time.sleep(300)  # Refresh every 5 minutes


# Custom styling
def load_css():
    return """
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .chat-message {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            font-size: 16px;
        }
        .user-message {
            background-color: #2D2D2D;
            color: #BB86FC;
            text-align: right;
            border: 1px solid #BB86FC;
        }
        .bot-message {
            background-color: #333333;
            color: #E0E0E0;
            text-align: left;
            border: 1px solid #03DAC6;
        }
        .sidebar .sidebar-content {
            background-color: #252525;
            padding: 20px;
            color: #E0E0E0;
        }
        .context-panel {
            background-color: #252525;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #555555;
            margin-top: 10px;
        }
        .context-title {
            color: #03DAC6;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .context-item {
            background-color: #333333;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 8px;
            font-size: 14px;
            border-left: 3px solid #BB86FC;
        }
        .stTextInput > div > div > input {
            background-color: #333333;
            color: #E0E0E0;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        .stButton > button {
            background-color: #BB86FC;
            color: #1E1E1E;
            border: none;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #03DAC6;
            color: #1E1E1E;
        }
        h1, h2, h3 {
            color: #BB86FC;
        }
        .stSpinner > div > div {
            color: #03DAC6;
        }
        .loading-message {
            background-color: #2D2D2D;
            color: #03DAC6;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 20px 0;
        }
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

    # Start background thread for file count updates
    if not st.session_state.background_thread_started:
        thread = threading.Thread(target=background_refresh, daemon=True)
        thread.start()
        st.session_state.background_thread_started = True

    # Create a two-column layout
    col1, col2 = st.columns([7, 3])

    # Main chat interface in the left column
    with col1:
        st.title("RAG Chatbot")
        st.write("Ask anything based on the selected data source.")

        # Load indexes in the background if not already loaded
        if not st.session_state.indexes_loaded:
            with st.spinner("Loading FAISS indexes..."):
                st.session_state.all_indexes = get_all_indexes()
                st.session_state.indexes_loaded = True

        user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")

        option = st.session_state.get("current_option", "Proforma Invoices")

        if st.button("Send") and user_input:
            # Append user question to chat history immediately for better UX
            st.session_state.chat_history.append(("user", user_input))

            # Create a placeholder for the bot's response
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="loading-message">Generating response...</div>',
                unsafe_allow_html=True
            )

            # Get the vector store for the selected data source
            vector_store = st.session_state.all_indexes[option]
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant documents

            # Retrieve documents
            documents = retriever.retrieve(user_input)
            st.session_state.context_documents = documents

            # Generate response using the documents
            prompt_instance = prompt_template.format(documents=documents, question=user_input)
            response = gemini_llm.generate(prompt_instance)

            # Update chat history with bot response
            st.session_state.chat_history.append(("bot", response))

            # Clear the placeholder
            response_placeholder.empty()

            # Force a rerun to update the UI
            st.experimental_rerun()

        # Display chat history
        for sender, message in reversed(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)

    # Right panel for context
    with col2:
        st.markdown("<h3>Retrieved Context</h3>", unsafe_allow_html=True)

        # Display the context documents
        if st.session_state.context_documents:
            st.markdown('<div class="context-panel">', unsafe_allow_html=True)
            st.markdown('<div class="context-title">Source Documents</div>', unsafe_allow_html=True)
            for i, doc in enumerate(st.session_state.context_documents):
                st.markdown(f'<div class="context-item">Document {i + 1}:<br>{doc.page_content}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="context-panel">No context retrieved yet. Ask a question to see relevant documents.</div>',
                unsafe_allow_html=True)

        # Options and stats in the right panel below context
        st.markdown("<h3>Options</h3>", unsafe_allow_html=True)
        option = st.radio("Select Data Source", ("Proforma Invoices", "Purchase Orders"))
        st.session_state.current_option = option

        st.markdown("<h3>Stats</h3>", unsafe_allow_html=True)
        st.write(f"Proforma Invoices: {st.session_state.proforma_new} new files")
        st.write(f"Purchase Orders: {st.session_state.po_new} new files")
        st.write(f"Last Updated: {st.session_state.last_updated}")

        # Add a refresh button for stats
        if st.button("Refresh Stats"):
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.experimental_rerun()


if __name__ == "__main__":
    main()