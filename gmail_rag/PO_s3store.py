import imaplib
import email
import boto3
import os
import logging
import io
import tempfile
import pandas as pd
from email.header import decode_header
import toml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Configuration constants
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
IMAP_SERVER = "imap.gmail.com"
S3_BUCKET = "kalika-rag"
PO_DUMP_FOLDER = "PO_Dump/"
S3_FAISS_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load secrets from secrets.toml
secrets = toml.load(SECRETS_FILE_PATH)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Email and S3 credentials
EMAIL_ACCOUNT = secrets["gmail_uname"]
EMAIL_PASSWORD = secrets["gmail_pwd"]
AWS_ACCESS_KEY = secrets["access_key_id"]
AWS_SECRET_KEY = secrets["secret_access_key"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

def clean_filename(filename):
    """Sanitize filename while preserving original extension if valid."""
    try:
        print("filename:",filename)
        decoded_name = decode_header(filename)[0][0]
        if isinstance(decoded_name, bytes):
            filename = decoded_name.decode(errors="ignore")
        else:
            filename = str(decoded_name)
    except:
        print("in except filename:", filename)
        filename = "po_dump"

    # Split filename and extension
    name, ext = os.path.splitext(filename)
    cleaned_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)

    return f"{cleaned_name}{ext}"  # Keep the original extension


def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        logging.error(f"S3 check error: {e}")
        return False


def upload_to_s3(file_content, bucket, key, content_type):
    """Upload file content directly to S3."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType=content_type
        )
        logging.info(f"Uploaded to S3: {key}")
        return True
    except Exception as e:
        logging.error(f"Upload failed for {key}: {e}")
        return False


def process_excel_content(file_content):
    """Extract and chunk text from an Excel file."""
    text = ""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        text = df.to_string(index=False)  # Convert DataFrame to string
    except Exception as e:
        logging.error(f"Excel processing error: {str(e)}")
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_faiss_index():
    """Create and upload FAISS index for PO Dumps."""
    try:
        documents = []
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PO_DUMP_FOLDER)

        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith((".xlsx", ".xls")):
                    try:
                        response = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                        file_content = response["Body"].read()

                        chunks = process_excel_content(file_content)
                        if chunks:
                            documents.extend(chunks)
                    except Exception as e:
                        logging.error(f"Error processing {obj['Key']}: {str(e)}")

        if not documents:
            logging.warning("No valid Excel documents found to index")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "po_faiss_index")
            vector_store = FAISS.from_texts(documents, embeddings)
            vector_store.save_local(index_path)

            for file_name in ["index.faiss", "index.pkl"]:
                local_path = os.path.join(index_path, file_name)
                s3_key = f"{S3_FAISS_INDEX_PATH}{file_name}"

                with open(local_path, "rb") as f:
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=s3_key,
                        Body=f
                    )

        logging.info(f"PO FAISS index updated with {len(documents)} chunks")

    except Exception as e:
        logging.error(f"PO FAISS index creation failed: {str(e)}")
        raise


def process_po_emails():
    """Process PO Order emails and upload Excel attachments to S3."""
    try:
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            logging.info("Successfully authenticated with email server")

            mail.select("inbox")
            status, email_ids = mail.search(None, '(SUBJECT "PO Order")')

            if status != "OK":
                logging.warning("No emails found with matching subject")
                return

            processed_files = 0
            for e_id in email_ids[0].split()[-40:]:  # Process last 10 emails
                try:
                    status, msg_data = mail.fetch(e_id, "(RFC822)")
                    if status != "OK":
                        continue

                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            for part in msg.walk():
                                if part.get_content_maintype() == "multipart":
                                    continue

                                if part.get_filename() and part.get_content_type() in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                                    filename = clean_filename(part.get_filename())
                                    print("final file::", filename)
                                    file_content = part.get_payload(decode=True)

                                    if not file_content:
                                        logging.warning(f"Skipping empty attachment: {filename}")
                                        continue

                                    key = f"{PO_DUMP_FOLDER}{filename}"
                                    if not file_exists_in_s3(S3_BUCKET, key):
                                        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if filename.endswith(".xlsx") else "application/vnd.ms-excel"
                                        if upload_to_s3(file_content, S3_BUCKET, key, content_type):
                                            processed_files += 1
                                    else:
                                        logging.info(f"Skipping existing file: {key}")
                except Exception as e:
                    logging.error(f"Error processing email {e_id}: {str(e)}")

            logging.info(f"Processing complete. Uploaded {processed_files} new Excel files.")

    except Exception as e:
        logging.error(f"Email processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    process_po_emails()
    create_faiss_index()


