import os
import logging
import boto3
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader, errors
from langchain_core.documents import Document  # Import Document

# Configuration
S3_BUCKET = "kalika-rag"
S3_FAISS_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
LOCAL_FAISS_DIR = "local_faiss_index"
LOCAL_FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_DIR, "proforma_faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize S3 client (for fetching FAISS index only)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
)

def get_faiss_index():
    """Load FAISS index from local storage."""
    try:
        return FAISS.load_local(LOCAL_FAISS_INDEX_PATH, embeddings)
    except Exception as e:
        logging.error(f"Failed to load FAISS index locally: {e}")
        return None

def fetch_faiss_index_from_s3():
    """Fetch FAISS index from S3 bucket."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "faiss_index")
            s3_client.download_file(S3_BUCKET, S3_FAISS_INDEX_PATH, temp_file_path)
            os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)
            os.rename(temp_file_path, LOCAL_FAISS_INDEX_PATH)
            logging.info(f"FAISS index downloaded from S3 and saved locally.")
            return True
    except Exception as e:
        logging.error(f"Failed to download FAISS index from S3: {e}")
        return False

