import streamlit as st
import boto3
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# AWS S3 Configuration
S3_BUCKET_NAME = "kalika-rag"
FAISS_INDEX_PATH = "faiss_indexes/proforma_faiss_index"

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
        # Create a temporary directory to store index files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download FAISS index files from S3
            s3_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=FAISS_INDEX_PATH)
            if 'Contents' in s3_objects:
                index_files = {obj['Key']: obj['Size'] for obj in s3_objects['Contents']}

                # Check if index.faiss and index.pkl exist
                if any(file.endswith("index.faiss") for file in index_files.keys()) and \
                        any(file.endswith("index.pkl") for file in index_files.keys()):

                    # Download both index.faiss and index.pkl
                    for s3_key in index_files.keys():
                        filename = os.path.basename(s3_key)
                        local_path = os.path.join(temp_dir, filename)
                        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)

                    st.info("Loading existing Proforma FAISS index from S3...")
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    return FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
                else:
                    st.error("Incomplete FAISS index files in S3 (missing .faiss or .pkl).")
                    return None
            else:
                st.warning("No FAISS index files found in S3.")
                return None
    except Exception as e:
        st.error(f"Error loading FAISS index from S3: {e}")
    return None


def query_proforma_rag(query):
    """Queries the RAG model for Proforma Invoice Data."""
    vector_store = load_faiss_index_from_s3()
    if not vector_store:
        st.warning("Index not found. Make sure the index exists in S3.")
        return "Index not found. Make sure the index exists in S3."

    retriever = vector_store.as_retriever()
    # llm = Ollama(model="llama2:latest")  # Using Llama2
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"max_new_tokens": 512})
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return chain.run(query)


# Streamlit UI
st.title("Chatbot for Proforma Invoice Analysis")

st.header("Ask Questions About Proforma Invoices")
query = st.text_input("Enter your query:")
if query:
    answer = query_proforma_rag(query)
    st.write("Answer:", answer)

