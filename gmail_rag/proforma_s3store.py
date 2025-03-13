import imaplib
import email
import boto3
import os
import logging
import io
import tempfile
from email.header import decode_header
import toml
from PyPDF2 import PdfReader, errors  # Fixed import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Configuration constants
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
IMAP_SERVER = "imap.gmail.com"
S3_BUCKET = "kalika-rag"
S3_FOLDER = "proforma_invoice/"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load secrets from secrets.toml
secrets = toml.load(SECRETS_FILE_PATH)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets["access_key_id"],
    aws_secret_access_key=secrets["secret_access_key"],
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


def clean_filename(filename):
    """Sanitize filename while preserving original extension if valid."""
    try:
        decoded_name = decode_header(filename)[0][0]
        if isinstance(decoded_name, bytes):
            filename = decoded_name.decode(errors='ignore')
        else:
            filename = str(decoded_name)
    except:
        filename = "proforma_invoice"

    # Split filename and extension
    name, ext = os.path.splitext(filename)
    cleaned_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

    # Preserve extension only if it's .pdf
    return f"{cleaned_name}.pdf" if ext.lower() == '.pdf' else f"{cleaned_name}.pdf"


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


def upload_to_s3(file_content, bucket, key):
    """Upload file content directly to S3."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType='application/pdf'
        )
        logging.info(f"Uploaded to S3: {key}")
        return True
    except Exception as e:
        logging.error(f"Upload failed for {key}: {e}")
        return False


def process_pdf_content(file_content):
    """Extract and chunk text from valid PDF bytes."""
    text = ""
    try:
        if not is_valid_pdf(file_content):
            raise errors.PdfReadError("Invalid PDF structure")

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


def create_faiss_index():
    """Create and upload FAISS index with proper error handling."""
    try:
        documents = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FOLDER)

        for page in pages:
            for obj in page.get('Contents', []):
                if obj['Key'].lower().endswith('.pdf'):
                    try:
                        response = s3_client.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                        file_content = response['Body'].read()

                        if not is_valid_pdf(file_content):
                            logging.warning(f"Skipping invalid PDF: {obj['Key']}")
                            continue

                        chunks = process_pdf_content(file_content)
                        if chunks:
                            documents.extend(chunks)
                    except Exception as e:
                        logging.error(f"Error processing {obj['Key']}: {str(e)}")

        if not documents:
            logging.warning("No valid PDF documents found to index")
            return

        # Create temporary directory for FAISS files
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "proforma_faiss_index")

            # Create and save FAISS index
            vector_store = FAISS.from_texts(documents, embeddings)
            vector_store.save_local(index_path)

            # Upload index files
            for file_name in ["index.faiss", "index.pkl"]:
                local_path = os.path.join(index_path, file_name)
                s3_key = f"{S3_PROFORMA_INDEX_PATH}{file_name}"

                with open(local_path, "rb") as f:
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=s3_key,
                        Body=f
                    )

        logging.info(f"FAISS index updated with {len(documents)} chunks from {len(documents)} documents")

    except Exception as e:
        logging.error(f"FAISS index creation failed: {str(e)}")
        raise


def process_proforma_emails():
    """Process emails with improved PDF validation and error handling."""
    try:
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(secrets["gmail_uname"], secrets["gmail_pwd"])
            logging.info("Email authentication successful")

            mail.select("inbox")
            status, email_ids = mail.search(None, '(SUBJECT "Proforma Invoice")')

            if status != "OK":
                logging.warning("No matching emails found")
                return

            processed_files = 0
            for e_id in email_ids[0].split()[-10:]:
                try:
                    status, msg_data = mail.fetch(e_id, "(RFC822)")
                    if status != "OK":
                        continue

                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            for part in msg.walk():
                                if part.get_content_maintype() == 'multipart':
                                    continue

                                if part.get_filename() and part.get_content_type() == 'application/pdf':
                                    filename = clean_filename(part.get_filename())
                                    file_content = part.get_payload(decode=True)

                                    if not file_content:
                                        logging.warning(f"Skipping empty attachment: {filename}")
                                        continue

                                    if not is_valid_pdf(file_content):
                                        logging.warning(f"Skipping invalid PDF: {filename}")
                                        continue

                                    key = f"{S3_FOLDER}{filename}"
                                    if not file_exists_in_s3(S3_BUCKET, key):
                                        if upload_to_s3(file_content, S3_BUCKET, key):
                                            processed_files += 1
                                    else:
                                        logging.info(f"Skipping existing file: {key}")
                except Exception as e:
                    logging.error(f"Error processing email {e_id}: {str(e)}")

            logging.info(f"Processing complete. Uploaded {processed_files} new valid PDFs.")

    except Exception as e:
        logging.error(f"Email processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    process_proforma_emails()
    create_faiss_index()

