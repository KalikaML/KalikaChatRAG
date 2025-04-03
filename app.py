import streamlit as st
import boto3
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import toml
from datetime import datetime

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Replace with your preferred model
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Load secrets from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize LLM (HuggingFace Hub)
llm = HuggingFaceHub(
    repo_id=LLM_MODEL,
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.
    Context: {context}
    Question: {question}
    Answer:
    """
)

# Function to load FAISS index from S3
def load_faiss_index_from_s3(index_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    if 'Contents' not in response:
        return 0
    new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
    return new_files

# Main Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")

    # Custom CSS for black theme
    st.markdown("""
        <style>
        /* General page styling */
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        /* Chat message styling */
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
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #252525;
            padding: 20px;
            color: #E0E0E0;
        }
        /* Input field */
        .stTextInput > div > div > input {
            background-color: #333333;
            color: #E0E0E0;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        /* Button styling */
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
        /* Titles and headers */
        h1, h2, h3 {
            color: #BB86FC;
        }
        /* Spinner text */
        .stSpinner > div > div {
            color: #03DAC6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for options and stats
    with st.sidebar:
        st.title("Chatbot Options")
        option = st.radio("Select Data Source", ("Proforma Invoices", "Purchase Orders"))
        st.subheader("New Files Processed")
        proforma_new = count_new_files(PROFORMA_FOLDER)
        po_new = count_new_files(PO_FOLDER)
        st.write(f"Proforma Invoices: {proforma_new} new files")
        st.write(f"Purchase Orders: {po_new} new files")
        st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Chat interface
    st.title("RAG Chatbot")
    st.write("Ask anything based on the selected data source!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_option" not in st.session_state:
        st.session_state.current_option = None

    # Load FAISS index based on selected option
    if option != st.session_state.current_option:
        st.session_state.current_option = option
        index_path = PROFORMA_INDEX_PATH if option == "Proforma Invoices" else PO_INDEX_PATH
        with st.spinner(f"Loading {option} FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(index_path)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )

    # Chat input
    user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            response = st.session_state.qa_chain.run(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()