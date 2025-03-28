'''import os
import logging
import boto3
import toml
import tempfile
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader, errors
from email_processor import fetch_proforma_emails
from s3_uploader import upload_to_s3
import streamlit as st

# Configuration
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FAISS_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
LOCAL_FAISS_DIR = "local_faiss_index"
LOCAL_FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_DIR, "proforma_faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load secrets
#secrets = toml.load(SECRETS_FILE_PATH)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["access_key_id"],
    aws_secret_access_key=st.secrets["secret_access_key"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Global FAISS index and stats
faiss_index = None
indexed_files = set()  # Track filenames of indexed PDFs
faiss_source = "Not initialized"
newly_indexed_files = []  # Track filenames of newly indexed PDFs


def process_pdf_content(file_content, filename):
    """Extract and chunk text from valid PDF bytes, adding filename as metadata."""
    text = ""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"PDF processing error for {filename}: {str(e)}")
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    # Add metadata to each chunk
    return [{"text": chunk, "metadata": {"source": filename}} for chunk in chunks]


def fetch_faiss_index_from_s3():
    """Fetch FAISS index from S3 and store it locally."""
    global faiss_source, faiss_index, indexed_files
    try:
        s3_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_FAISS_INDEX_PATH)
        if 'Contents' not in s3_objects:
            logging.warning("No FAISS index files found in S3.")
            return False

        index_files = [obj['Key'] for obj in s3_objects['Contents']]
        faiss_file = any(file.endswith("index.faiss") for file in index_files)
        pkl_file = any(file.endswith("index.pkl") for file in index_files)

        if not (faiss_file and pkl_file):
            logging.error("Incomplete FAISS index files in S3 (missing .faiss or .pkl).")
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            for s3_key in index_files:
                filename = os.path.basename(s3_key)
                local_path = os.path.join(temp_dir, filename)
                s3_client.download_file(S3_BUCKET, s3_key, local_path)

            # Load the index from the temporary directory
            faiss_index = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            faiss_source = "S3 (initial fetch)"

            # Update indexed_files based on S3
            indexed_files = set()
            for obj in s3_objects['Contents']:
                if obj['Key'].endswith(".pdf"):
                    indexed_files.add(obj['Key'])

            logging.info("FAISS index fetched from S3 and stored locally.")
            return True

    except Exception as e:
        logging.error(f"Error fetching FAISS index from S3: {e}")
        return False


def initialize_faiss_index_from_local(temp_dir=None):
    """Initialize or load the FAISS index locally, optionally from a temp directory."""
    global faiss_index, indexed_files, faiss_source
    try:
        target_path = temp_dir if temp_dir else LOCAL_FAISS_INDEX_PATH

        if os.path.exists(target_path) and \
                os.path.isfile(os.path.join(target_path, "index.faiss")) and \
                os.path.isfile(os.path.join(target_path, "index.pkl")):
            logging.info(f"Loading FAISS index from {target_path}...")
            faiss_index = FAISS.load_local(target_path, embeddings,
                                            allow_dangerous_deserialization=True)
            faiss_source = "Local storage"

        else:
            logging.warning("No local FAISS index found. It will be created on the next update.")
            faiss_index = None
            faiss_source = "Not initialized"

    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        faiss_index = None


def update_faiss_index_from_emails():
    """Fetch emails, upload PDFs to S3, and update the local FAISS index."""
    global faiss_index, indexed_files, faiss_source, newly_indexed_files
    newly_indexed_files = []  # Reset the list of newly indexed files
    try:
        pdf_files = fetch_proforma_emails()
        if not pdf_files:
            logging.info("No new PDFs to process.")
            return []

        valid_pdfs = upload_to_s3(pdf_files)
        if not valid_pdfs:
            logging.info("No new valid PDFs to index.")
            return []

        new_documents = []
        for filename, file_content in valid_pdfs:
            s3_key = f"proforma_invoice/{filename}"  # Construct S3 key
            if s3_key not in indexed_files:  # Use S3 key for checking
                chunks = process_pdf_content(file_content, filename)
                if chunks:
                    new_documents.extend([chunk["text"] for chunk in chunks])
                    indexed_files.add(s3_key)  # Store S3 key in indexed_files
                    newly_indexed_files.append(filename)  # Add filename to the list of newly indexed files

        if new_documents:
            # Ensure faiss_index is initialized
            if faiss_index is None:
                initialize_faiss_index_from_local()
            if faiss_index is None:
                faiss_index = FAISS.from_texts(new_documents, embeddings, metadatas=[{"source": f} for f in newly_indexed_files])
                faiss_source = "Local (created from emails)"
            else:
                new_vector_store = FAISS.from_texts(new_documents, embeddings, metadatas=[{"source": f} for f in newly_indexed_files])
                faiss_index.merge_from(new_vector_store)
                faiss_source = "Local (updated from emails)"

            faiss_index.save_local(LOCAL_FAISS_INDEX_PATH)
            logging.info(f"FAISS index updated with {len(new_documents)} new chunks.")
            return newly_indexed_files  # Return the list of newly indexed files
        else:
            logging.info("No new documents to add to FAISS index.")
            return []

    except Exception as e:
        logging.error(f"FAISS index update failed: {str(e)}")
        return []


def get_faiss_index():
    """Return the current FAISS index, initializing if necessary."""
    global faiss_index
    if faiss_index is None:
        initialize_faiss_index_from_local()
    return faiss_index


def get_faiss_stats():
    """Return statistics about the FAISS index."""
    global indexed_files, faiss_source
    return {
        "indexed_files": len(indexed_files),
        "faiss_source": faiss_source
    }
'''

import os
import logging
import boto3
import toml
import tempfile
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader, errors
from email_processor import fetch_proforma_emails
from s3_uploader import upload_to_s3
from langchain_core.documents import Document  # Import Document

# Configuration
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FAISS_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
LOCAL_FAISS_DIR = "local_faiss_index"
LOCAL_FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_DIR, "proforma_faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load secrets
#secrets = toml.load(SECRETS_FILE_PATH)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["access_key_id"],
    aws_secret_access_key=st.secrets["secret_access_key"],
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Global FAISS index and stats
faiss_index = None
indexed_files = set()  # Track filenames of indexed PDFs
faiss_source = "Not initialized"
newly_indexed_files = []  # Track filenames of newly indexed PDFs


def process_pdf_content(file_content, filename):
    """Extract and chunk text from valid PDF bytes, adding filename as metadata."""
    text = ""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"PDF processing error for {filename}: {str(e)}")
        return []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    # Add metadata to each chunk
    return [{"text": chunk, "metadata": {"source": filename}} for chunk in chunks]


def fetch_faiss_index_from_s3():
    """Fetch FAISS index from S3 and store it locally."""
    global faiss_source, faiss_index, indexed_files
    try:
        s3_objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_FAISS_INDEX_PATH)
        if 'Contents' not in s3_objects:
            logging.warning("No FAISS index files found in S3.")
            return False

        index_files = [obj['Key'] for obj in s3_objects['Contents']]
        faiss_file = any(file.endswith("index.faiss") for file in index_files)
        pkl_file = any(file.endswith("index.pkl") for file in index_files)

        if not (faiss_file and pkl_file):
            logging.error("Incomplete FAISS index files in S3 (missing .faiss or .pkl).")
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            for s3_key in index_files:
                filename = os.path.basename(s3_key)
                local_path = os.path.join(temp_dir, filename)
                s3_client.download_file(S3_BUCKET, s3_key, local_path)

            # Load the index from the temporary directory
            faiss_index = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
            faiss_source = "S3 (initial fetch)"

            # Update indexed_files based on S3
            indexed_files = set()
            for obj in s3_objects['Contents']:
                if obj['Key'].endswith(".pdf"):
                    indexed_files.add(obj['Key'])

            logging.info("FAISS index fetched from S3 and stored locally.")
            return True

    except Exception as e:
        logging.error(f"Error fetching FAISS index from S3: {e}")
        return False


def initialize_faiss_index_from_local(temp_dir=None):
    """Initialize or load the FAISS index locally, optionally from a temp directory."""
    global faiss_index, indexed_files, faiss_source
    try:
        target_path = temp_dir if temp_dir else LOCAL_FAISS_INDEX_PATH

        if os.path.exists(target_path) and \
                os.path.isfile(os.path.join(target_path, "index.faiss")) and \
                os.path.isfile(os.path.join(target_path, "index.pkl")):
            logging.info(f"Loading FAISS index from {target_path}...")
            faiss_index = FAISS.load_local(target_path, embeddings,
                                            allow_dangerous_deserialization=True)
            faiss_source = "Local storage"

        else:
            logging.warning("No local FAISS index found. It will be created on the next update.")
            faiss_index = None
            faiss_source = "Not initialized"

    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        faiss_index = None


def update_faiss_index_from_emails():
    """Fetch emails, upload PDFs to S3, and update the local FAISS index."""
    global faiss_index, indexed_files, faiss_source, newly_indexed_files
    newly_indexed_files = []  # Reset the list of newly indexed files
    try:
        pdf_files = fetch_proforma_emails()
        if not pdf_files:
            logging.info("No new PDFs to process.")
            return []

        valid_pdfs = upload_to_s3(pdf_files)
        if not valid_pdfs:
            logging.info("No new valid PDFs to index.")
            return []

        new_documents = []
        for filename, file_content in valid_pdfs:
            s3_key = f"proforma_invoice/{filename}"  # Construct S3 key
            if s3_key not in indexed_files:  # Use S3 key for checking
                chunks = process_pdf_content(file_content, filename)
                if chunks:
                    new_documents.extend([chunk["text"] for chunk in chunks])
                    indexed_files.add(s3_key)  # Store S3 key in indexed_files
                    newly_indexed_files.append(filename)  # Add filename to the list of newly indexed files

        if new_documents:
            # Ensure faiss_index is initialized
            if faiss_index is None:
                initialize_faiss_index_from_local()
            if faiss_index is None:
                faiss_index = FAISS.from_texts(new_documents, embeddings, metadatas=[{"source": f} for f in newly_indexed_files])
                faiss_source = "Local (created from emails)"
            else:
                new_vector_store = FAISS.from_texts(new_documents, embeddings, metadatas=[{"source": f} for f in newly_indexed_files])
                faiss_index.merge_from(new_vector_store)
                faiss_source = "Local (updated from emails)"

            faiss_index.save_local(LOCAL_FAISS_INDEX_PATH)
            logging.info(f"FAISS index updated with {len(new_documents)} new chunks.")
            return newly_indexed_files  # Return the list of newly indexed files
        else:
            logging.info("No new documents to add to FAISS index.")
            return []

    except Exception as e:
        logging.error(f"FAISS index update failed: {str(e)}")
        return []


def get_faiss_index():
    """Return the current FAISS index, initializing if necessary."""
    global faiss_index
    if faiss_index is None:
        initialize_faiss_index_from_local()
    return faiss_index


def get_faiss_stats():
    """Return statistics about the FAISS index."""
    global indexed_files, faiss_source
    return {
        "indexed_files": len(indexed_files),
        "faiss_source": faiss_source
    }

