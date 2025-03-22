'''import boto3
import os
import logging
import toml
from PyPDF2 import PdfReader, errors
import io

# Configuration
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"

# Load secrets
secrets = toml.load(SECRETS_FILE_PATH)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets["access_key_id"],
    aws_secret_access_key=secrets["secret_access_key"],
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
import boto3
import os
import logging
import toml
from PyPDF2 import PdfReader, errors
import io

# Configuration
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"

# Load secrets
secrets = toml.load(SECRETS_FILE_PATH)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets["access_key_id"],
    aws_secret_access_key=secrets["secret_access_key"],
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
    return valid_pdfs'''

import boto3
import os
import logging
import toml
from PyPDF2 import PdfReader, errors
import io

# Configuration
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"

# Load secrets
secrets = toml.load(SECRETS_FILE_PATH)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets["access_key_id"],
    aws_secret_access_key=secrets["secret_access_key"],
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