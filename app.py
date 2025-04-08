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

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
try:
    secrets = toml.load(SECRETS_FILE_PATH)
    S3_BUCKET = "kalika-rag"
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Model name from Hugging Face
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"
    GEMINI_API_KEY = secrets["gemini_api_key"]
except (FileNotFoundError, KeyError) as e:
    st.error(f"Configuration error: {str(e)}")
    st.stop()


# --- Initialize Resources with Caching ---
@st.cache_resource
def init_resources():
    """Initialize all required resources with proper error handling"""
    resources = {}

    try:
        # 1. S3 Client
        resources['s3'] = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        resources['s3'].list_buckets()

        # 2. Embeddings Model (BGE)
        resources['embeddings'] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 3. Gemini Model
        resources['gemini'] = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )

        return resources

    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        logging.exception("Resource initialization error")
        st.stop()


resources = init_resources()


# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def load_faiss_index(s3_client, embeddings):
    """Enhanced index loading with retries and validation"""
    try:
        temp_dir = tempfile.mkdtemp()

        # Download index files
        s3_client.download_file(S3_BUCKET, f"{S3_PROFORMA_INDEX_PATH}.faiss",
                                os.path.join(temp_dir, "index.faiss"))
        s3_client.download_file(S3_BUCKET, f"{S3_PROFORMA_INDEX_PATH}.pkl",
                                os.path.join(temp_dir, "index.pkl"))

        # Load index with safety checks
        return FAISS.load_local(
            temp_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Index loading failed: {str(e)}")
        logging.exception("FAISS index error")
        st.stop()


# --- UI and Application Flow ---
st.title("📄 Proforma Invoice Query Assistant")
st.markdown("Ask questions about proforma invoices processed from email attachments.")

# Load FAISS index
with st.spinner("Loading knowledge base..."):
    vector_store = load_faiss_index(resources['s3'], resources['embeddings'])

query_text = st.text_input("Enter your query:",
                           placeholder="e.g., What's the total amount for invoice X?")

if query_text:
    # Retrieve documents and generate response
    documents = vector_store.similarity_search(query_text, k=25)

    with st.spinner("Generating response..."):
        response = resources['gemini'].invoke([
            SystemMessage(content=f"Context: {[doc.page_content for doc in documents]}"),
            HumanMessage(content=query_text)
        ])

    st.markdown("### Response:")
    st.markdown(response.content)
