'''import os
import logging
import boto3
import toml
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader, errors
import io
from email_processor import fetch_proforma_emails
from s3_uploader import upload_to_s3
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter # Or whatever
import pickle

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
    global faiss_source
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

            initialize_faiss_index_from_local(temp_dir)
            faiss_source = "S3 (initial fetch)"
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
            faiss_index = FAISS.load_local(target_path, embeddings, allow_dangerous_deserialization=True)
            # Assume all files in S3 were indexed initially; refine this if metadata exists
            if not temp_dir:
                faiss_source = "Local storage"
            # For simplicity, we'll count indexed files later during updates
            if temp_dir:
                os.makedirs(LOCAL_FAISS_INDEX_PATH, exist_ok=True)
                for filename in ["index.faiss", "index.pkl"]:
                    os.rename(
                        os.path.join(temp_dir, filename),
                        os.path.join(LOCAL_FAISS_INDEX_PATH, filename)
                    )
        else:
            logging.warning("No local FAISS index found. It will be created on the next update.")
            faiss_index = None
            faiss_source = "Not initialized"
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        faiss_index = None

def update_faiss_index_from_emails():
    """
    Updates the FAISS index by fetching new proforma invoice emails,
    uploading them to S3, and indexing them.
    Returns the count of new files added and their names.
    """
    from email_processor import fetch_proforma_emails
    from s3_uploader import upload_to_s3

    new_files_count = 0
    new_file_names = []
    all_documents = []  # To store all documents for indexing

    # 1. Fetch new proforma invoice emails
    pdf_files = fetch_proforma_emails()

    # 2. Upload new PDFs to S3
    valid_pdf_files = upload_to_s3(pdf_files)

    # 3. Index new PDFs and update FAISS index
    if valid_pdf_files:
        new_files_count = len(valid_pdf_files)
        new_file_names = [file[0] for file in valid_pdf_files]  # Extract filenames

        # Load documents and split them
        for filename, file_content in valid_pdf_files:
            with open("temp.pdf", "wb") as f:
                f.write(file_content)  # Temporarily write to a file
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            all_documents.extend(documents)  # Add to the list of all documents

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # If the index already exists, load it and add new documents
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings)  # Assuming your index is saved locally
            vector_store.add_documents(texts)
            print("Existing index found, adding new documents.")
        except:
            # If the index does not exist, create a new one
            vector_store = FAISS.from_documents(texts, embeddings)
            print("No existing index found, creating a new one.")

        # Save the updated FAISS index locally
        vector_store.save_local("faiss_index")

        print(f"FAISS index updated. {new_files_count} new files added.")

    # Upload the index to s3
    # upload_faiss_index_to_s3(faiss_index, temp_dir)
    return new_files_count, new_file_names

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
    }'''
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader, errors
import io
from s3_uploader import upload_to_s3, download_faiss_index_from_s3, get_s3_file_count

# Configuration
LOCAL_FAISS_DIR = "local_faiss_index"
LOCAL_FAISS_INDEX_PATH = os.path.join(LOCAL_FAISS_DIR, "proforma_faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

# Global FAISS index
faiss_index = None
last_updated_file_count = 0  # Track the number of files when the index was last updated
new_files_count = 0

def process_pdf_content(file_content):
    """Extract and chunk text from valid PDF bytes."""
    text = ""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def initialize_faiss_index():
    """Initialize or load the FAISS index locally from S3 if available."""
    global faiss_index, last_updated_file_count
    try:
        # Download FAISS index from S3
        if download_faiss_index_from_s3(LOCAL_FAISS_INDEX_PATH):
            logging.info("FAISS index downloaded from S3.")
            faiss_index = FAISS.load_local(LOCAL_FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("FAISS index loaded from local.")

            # Set last updated file count based on S3 file count
            last_updated_file_count = get_s3_file_count()
            logging.info(f"Initial S3 file count: {last_updated_file_count}")
        else:
            logging.warning("No FAISS index found in S3. It will be created on the next update.")
            faiss_index = None
            last_updated_file_count = 0

    except Exception as e:
        logging.error(f"Error initializing FAISS index: {e}")
        faiss_index = None
        last_updated_file_count = 0

def update_faiss_index():
    """Check for new files in S3 and update the FAISS index."""
    global faiss_index, last_updated_file_count, new_files_count
    from s3_uploader import get_s3_file_count  # Import here to avoid circular dependency
    from email_processor import fetch_proforma_emails

    try:
        # Get the current file count in S3
        current_file_count = get_s3_file_count()
        logging.info(f"Current S3 file count: {current_file_count}")

        # Compare with the last updated file count
        if current_file_count > last_updated_file_count:
            new_files_count = current_file_count - last_updated_file_count
            logging.info(f"New files found: {new_files_count}")

            # Fetch PDFs from emails
            pdf_files = fetch_proforma_emails()
            # Upload new PDFs to S3 and get valid PDFs for indexing
            # You might need to adjust this part based on your logic for determining "new" files
            valid_pdfs = upload_to_s3(pdf_files)

            # Extract text and create documents
            texts = []
            for filename, file_content in valid_pdfs:
                pdf_texts = process_pdf_content(file_content)
                texts.extend(pdf_texts)

            if not texts:
                logging.info("No new content to index.")
                return 0  # Return 0 if no updates were made

            # Create a new FAISS index or update the existing one
            if faiss_index is None:
                logging.info("Creating new FAISS index.")
                faiss_index = FAISS.from_texts(texts, embeddings)
            else:
                logging.info("Updating existing FAISS index.")
                new_faiss = FAISS.from_texts(texts, embeddings)
                faiss_index.merge_from(new_faiss)

            # Save the updated FAISS index locally
            os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)
            faiss_index.save_local(LOCAL_FAISS_INDEX_PATH)
            logging.info(f"FAISS index updated and saved to {LOCAL_FAISS_INDEX_PATH}")

            # Upload the updated FAISS index to S3
            if upload_faiss_index_to_s3(LOCAL_FAISS_INDEX_PATH):
                logging.info("FAISS index uploaded to S3.")
            else:
                logging.error("Failed to upload FAISS index to S3.")

            # Update the last updated file count
            last_updated_file_count = current_file_count

            return new_files_count  # Return the number of new files indexed

        else:
            logging.info("No new files found in S3.")
            return 0  # Return 0 if no updates were made

    except Exception as e:
        logging.error(f"FAISS index update failed: {e}")
        return 0

def get_faiss_index():
    """Return the current FAISS index."""
    global faiss_index
    return faiss_index

def get_last_updated_file_count():
    """Return the number of files when the FAISS index was last updated."""
    global last_updated_file_count
    return last_updated_file_count

def get_new_files_count():
    global new_files_count
    return new_files_count
