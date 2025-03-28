'''

import boto3
import os
import logging
import toml
from PyPDF2 import PdfReader, errors
import io
import streamlit as st

# Configuration
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"

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

def is_valid_pdf(content):
    """Verify if content is a valid PDF."""
    try:
        PdfReader(io.BytesIO(content))
        return True
    except (errors.PdfReadError, ValueError, TypeError):
        return False

def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 check error: {e}")
        return False

def upload_to_s3(pdf_files):
    """Upload new PDFs to S3 and return valid PDFs for indexing."""
    valid_pdfs = []
    for filename, file_content in pdf_files:
        try:
            if not is_valid_pdf(file_content):
                logging.warning(f"Skipping invalid PDF: {filename}")
                continue
            key = f"{S3_FOLDER}{filename}"
            if not file_exists_in_s3(S3_BUCKET, key):
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=file_content,
                    ContentType='application/pdf'
                )
                logging.info(f"Uploaded to S3: {key}")
                valid_pdfs.append((filename, file_content))
            else:
                logging.info(f"Skipping existing file: {key}")
        except Exception as e:
            logging.error(f"Upload failed for {filename}: {e}")
    return valid_pdfs

def get_s3_file_count():
    """Count the total number of PDF files in S3 under the proforma_invoice folder."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FOLDER)
        total_files = 0
        for page in pages:
            total_files += len([obj for obj in page.get('Contents', []) if obj['Key'].lower().endswith('.pdf')])
        return total_files
    except Exception as e:
        logging.error(f"Error counting S3 files: {e}")
        return 0'''
import boto3
import os
import logging
import toml
from PyPDF2 import PdfReader, errors
import io
import streamlit as st

# Configuration
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"

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

def is_valid_pdf(content):
    """Verify if content is a valid PDF."""
    try:
        PdfReader(io.BytesIO(content))
        return True
    except (errors.PdfReadError, ValueError, TypeError):
        return False

def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 check error: {e}")
        return False

def upload_to_s3(pdf_files):
    """Upload new PDFs to S3 and return valid PDFs for indexing."""
    valid_pdfs = []
    for filename, file_content in pdf_files:
        try:
            if not is_valid_pdf(file_content):
                logging.warning(f"Skipping invalid PDF: {filename}")
                continue
            key = f"{S3_FOLDER}{filename}"
            if not file_exists_in_s3(S3_BUCKET, key):
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=file_content,
                    ContentType='application/pdf'
                )
                logging.info(f"Uploaded to S3: {key}")
                valid_pdfs.append((filename, file_content))
            else:
                logging.info(f"Skipping existing file: {key}")
        except Exception as e:
            logging.error(f"Upload failed for {filename}: {e}")
    return valid_pdfs

def get_s3_file_count():
    """Count the total number of PDF files in S3 under the proforma_invoice folder."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FOLDER)
        total_files = 0
        for page in pages:
            total_files += len([obj for obj in page.get('Contents', []) if obj['Key'].lower().endswith('.pdf')])
        return total_files
    except Exception as e:
        logging.error(f"Error counting S3 files: {e}")
        return 0

def download_faiss_index_from_s3(local_faiss_path):
    """Download the FAISS index from S3."""
    faiss_index_key = os.path.join(S3_FOLDER, "proforma_faiss_index/index.faiss")
    faiss_pkl_key = os.path.join(S3_FOLDER, "proforma_faiss_index/index.pkl")

    try:
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_faiss_path), exist_ok=True)

        # Download index.faiss
        s3_client.download_file(S3_BUCKET, faiss_index_key, os.path.join(local_faiss_path, "index.faiss"))
        logging.info(f"Downloaded index.faiss from S3 to {local_faiss_path}")

        # Download index.pkl
        s3_client.download_file(S3_BUCKET, faiss_pkl_key, os.path.join(local_faiss_path, "index.pkl"))
        logging.info(f"Downloaded index.pkl from S3 to {local_faiss_path}")

        return True
    except Exception as e:
        logging.error(f"Error downloading FAISS index from S3: {e}")
        return False

def upload_faiss_index_to_s3(local_faiss_path):
    """Upload the FAISS index to S3."""
    faiss_index_key = os.path.join(S3_FOLDER, "proforma_faiss_index/index.faiss")
    faiss_pkl_key = os.path.join(S3_FOLDER, "proforma_faiss_index/index.pkl")

    try:
        # Upload index.faiss
        s3_client.upload_file(os.path.join(local_faiss_path, "index.faiss"), S3_BUCKET, faiss_index_key)
        logging.info(f"Uploaded index.faiss to S3: {faiss_index_key}")

        # Upload index.pkl
        s3_client.upload_file(os.path.join(local_faiss_path, "index.pkl"), S3_BUCKET, faiss_pkl_key)
        logging.info(f"Uploaded index.pkl to S3: {faiss_pkl_key}")

        return True
    except Exception as e:
        logging.error(f"Error uploading FAISS index to S3: {e}")
        return False
