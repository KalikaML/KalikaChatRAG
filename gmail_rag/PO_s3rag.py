import streamlit as st
import boto3
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# AWS S3 Configuration
S3_BUCKET_NAME = "kalika-rag"
S3_FAISS_INDEX_PATH = "faiss_indexes/po_faiss_index"

# Load secrets from Streamlit
AWS_ACCESS_KEY = st.secrets["access_key_id"]
AWS_SECRET_KEY = st.secrets["secret_access_key"]

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)


@st.cache_resource
def load_faiss_index_from_s3():
    """Loads a FAISS index from S3."""
    try:
        # Download FAISS index files from S3
        s3_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_FAISS_INDEX_PATH)
        if 'Contents' in s3_objects:
            index_files = {obj['Key']: obj['Size'] for obj in s3_objects['Contents']}
            if any(file.endswith(".faiss") for file in index_files) and any(
                    file.endswith(".pkl") for file in index_files):
                # Create a temporary directory to store index files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for s3_key in index_files.keys():
                        filename = os.path.basename(s3_key)
                        local_path = os.path.join(temp_dir, filename)
                        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)  #load directly only from s3 or shedule in low picktime

                    st.info("Loading existing PO FAISS index from S3...")
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)  # Add this part
    except Exception as e:
        st.error(f"Error loading FAISS index from S3: {e}")
    return None


def query_po_rag(query):
    """Queries the RAG model for PO Dump Data."""
    vector_store = load_faiss_index_from_s3()
    if not vector_store:
        st.warning("Index not found. Make sure the index exists in S3.")
        return "Index not found. Make sure the index exists in S3."

    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama2:latest")  # Using Llama2
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return chain.run(query)


# Streamlit UI
st.title("Chatbot for PO Dump Analysis")

st.header("Ask Questions About PO Data")
query = st.text_input("Enter your query:")
if query:
    answer = query_po_rag(query)
    st.write("Answer:", answer)

