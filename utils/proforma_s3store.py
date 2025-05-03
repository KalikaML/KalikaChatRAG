
import imaplib
import email
import boto3
import os
import logging
import io
import tempfile
from email.header import decode_header
import toml
from PyPDF2 import PdfReader, errors
# Ensure necessary imports for local models (usually handled by langchain_huggingface)
# Make sure sentence-transformers and torch/tensorflow/flax are installed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Configuration constants
SECRETS_FILE_PATH = ".streamlit/secrets.toml"

# --- MODIFIED: Path to your LOCAL embedding model ---
# >>> IMPORTANT: Update this path to the actual location of your ge-base-en-v1.5 model directory <<<
LOCAL_EMBEDDING_MODEL_PATH = "bge-base-en-v1.5"
# e.g., "C:/ai_models/ge-base-en-v1.5" or "/home/user/models/ge-base-en-v1.5"
# ---

IMAP_SERVER = "imap.gmail.com"
S3_BUCKET = "kalika-rag"
PO_DUMP_FOLDER = "proforma_invoice/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"

# Load secrets from secrets.toml
try:
    secrets = toml.load(SECRETS_FILE_PATH)
except FileNotFoundError:
    logging.error(f"Secrets file not found at: {SECRETS_FILE_PATH}")
    raise
except Exception as e:
    logging.error(f"Error loading secrets file: {e}")
    raise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Email and S3 credentials
try:
    EMAIL_ACCOUNT = secrets["gmail_uname"]
    EMAIL_PASSWORD = secrets["gmail_pwd"]
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]
except KeyError as e:
    logging.error(f"Missing key in secrets file: {e}")
    raise

# Initialize S3 client
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    # Verify connection by listing buckets (optional, remove if not needed)
    s3_client.list_buckets()
    logging.info("Successfully initialized S3 client.")
except Exception as e:
    logging.error(f"Failed to initialize S3 client: {e}")
    raise


# Initialize embeddings model using the local path
try:
    # Check if the local path exists before initializing
    if not os.path.isdir(LOCAL_EMBEDDING_MODEL_PATH):
         raise FileNotFoundError(f"Local model directory not found: {LOCAL_EMBEDDING_MODEL_PATH}")

    logging.info(f"Loading local embedding model from: {LOCAL_EMBEDDING_MODEL_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL_PATH, # Use the local path
        model_kwargs={'device': 'cpu'}, # Keep using CPU or change to 'cuda' if preferred and available
        encode_kwargs={'normalize_embeddings': False} # Check if ge-base-en-v1.5 benefits from normalization
    )
    logging.info("Successfully initialized local embeddings model.")
except FileNotFoundError as e:
    logging.error(e)
    raise
except Exception as e:
    logging.error(f"Failed to load local embedding model: {e}")
    logging.error("Ensure 'sentence-transformers' and necessary backends (like 'torch') are installed, and the model path is correct.")
    raise


def clean_filename(filename):
    """Sanitize filename while preserving original extension if valid."""
    try:
        # Decode header to handle potential encoding issues
        decoded_parts = decode_header(filename)
        decoded_name = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                # If no encoding detected, try default encodings or ignore errors
                decoded_name += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                decoded_name += str(part)
        filename = decoded_name.strip() if decoded_name else "po_document"

    except Exception as e:
        logging.warning(f"Could not decode filename '{filename}', using default. Error: {e}")
        filename = "po_document"

    # Split filename and extension
    name, ext = os.path.splitext(filename)

    # Remove or replace invalid characters for filenames
    # Allows letters, numbers, underscore, hyphen, and space (replace space with underscore)
    cleaned_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name.replace(' ', '_'))
    # Remove leading/trailing underscores and reduce multiple consecutive underscores
    cleaned_name = '_'.join(filter(None, cleaned_name.split('_')))


    # Preserve extension only if it's .pdf (case-insensitive)
    if ext.lower() == '.pdf':
        return f"{cleaned_name}.pdf"
    else:
        # If not PDF, decide how to handle (e.g., keep original ext, discard, add default)
        # Here, we discard the non-pdf extension for simplicity in this script's context
        logging.warning(f"Original file '{filename}' was not a PDF. Saving sanitized name without extension: {cleaned_name}")
        # Or return f"{cleaned_name}{ext}" # to keep original extension
        return cleaned_name # Current behavior: discard non-pdf extension


def is_valid_pdf(content):
    """Verify if content is a valid PDF by attempting to read it."""
    if not content:
        logging.warning("PDF content is empty.")
        return False
    try:
        # Use BytesIO to treat the byte content like a file
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        # Check if it has pages; an empty or severely corrupted PDF might not
        if not reader.pages:
             logging.warning("PDF is valid but contains no pages.")
             # Decide if this is acceptable; here we consider it valid structure-wise
             # return False # Uncomment if 0 pages means invalid for your use case
        return True
    except errors.PdfReadError as e:
        logging.warning(f"Invalid PDF structure detected: {e}")
        return False
    except Exception as e: # Catch other potential issues like TypeError
        logging.warning(f"Error validating PDF content: {e}")
        return False


def file_exists_in_s3(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        # If the error code is 404 (Not Found), then the file doesn't exist.
        if e.response['Error']['Code'] == '404':
            return False
        # Log other S3 errors
        logging.error(f"S3 check error for key '{key}': {e}")
        return False # Assume failure on other errors
    except Exception as e:
        logging.error(f"Unexpected error checking S3 for key '{key}': {e}")
        return False


def upload_to_s3(file_content, bucket, key):
    """Upload file content directly to S3."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType='application/pdf' # Assuming content is always PDF here
        )
        logging.info(f"Uploaded to S3: s3://{bucket}/{key}")
        return True
    except Exception as e:
        logging.error(f"Upload failed for S3 key '{key}': {e}")
        return False


def process_pdf_content(file_content, filename_for_logging=""):
    """Extract and chunk text from valid PDF bytes."""
    text = ""
    pdf_log_name = f"'{filename_for_logging}' " if filename_for_logging else ""
    try:
        # Basic validation check (optional, as is_valid_pdf is called before)
        if not file_content:
             logging.warning(f"Skipping processing for {pdf_log_name}due to empty content.")
             return []

        pdf_file = io.BytesIO(file_content)
        try:
            reader = PdfReader(pdf_file)
            if not reader.pages:
                 logging.warning(f"No pages found in PDF {pdf_log_name}. Skipping text extraction.")
                 return []

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Add newline between pages
                    else:
                        logging.warning(f"Page {i+1} in PDF {pdf_log_name}had no extractable text.")
                except Exception as page_e:
                    logging.error(f"Error extracting text from page {i+1} in PDF {pdf_log_name}: {page_e}")
                    continue # Skip problematic page

        except errors.PdfReadError as pdf_e:
            logging.error(f"Failed to read PDF structure {pdf_log_name}: {pdf_e}")
            return [] # Cannot process if structure is unreadable
        except Exception as e:
            logging.error(f"Unexpected error during PDF reading {pdf_log_name}: {e}")
            return []

    except Exception as e: # Catch errors related to BytesIO or other setup
        logging.error(f"PDF processing setup error for {pdf_log_name}: {e}")
        return []

    if not text.strip():
        logging.warning(f"No text extracted from PDF {pdf_log_name}.")
        return []

    # Use CharacterTextSplitter as defined
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    try:
        chunks = text_splitter.split_text(text)
        logging.info(f"Split PDF {pdf_log_name} into {len(chunks)} chunks.")
        return chunks
    except Exception as split_e:
        logging.error(f"Error splitting text from PDF {pdf_log_name}: {split_e}")
        return [] # Return empty list if splitting fails


def process_po_emails():
    """Process PO emails, upload valid PDF attachments to S3, avoiding duplicates."""
    processed_files = 0
    skipped_duplicates = 0
    skipped_invalid = 0
    skipped_empty = 0
    email_count = 0
    total_attachments = 0

    try:
        logging.info("Connecting to IMAP server...")
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            logging.info("Successfully authenticated with email server.")

            mail.select("inbox")
            # Consider making the search criteria more robust if needed
            search_criteria = 'OR OR (SUBJECT "PO Order") (SUBJECT "Purchase Order") (SUBJECT "PO Dump")'
            status, email_ids_bytes = mail.search(None, search_criteria)

            if status != "OK":
                logging.error("Failed to search emails.")
                return

            email_id_list = email_ids_bytes[0].split()
            if not email_id_list:
                logging.info("No emails found matching the search criteria.")
                return

            # Process a subset or all emails (e.g., last 10 or all)
            # Process up to the last 50 emails for demonstration, adjust as needed
            process_limit = 50
            emails_to_process = email_id_list[-process_limit:]
            logging.info(f"Found {len(email_id_list)} matching emails. Processing the last {len(emails_to_process)}.")

            for e_id in emails_to_process:
                email_count += 1
                try:
                    status, msg_data = mail.fetch(e_id, "(RFC822)")
                    if status != "OK":
                        logging.warning(f"Failed to fetch email ID {e_id.decode()}. Status: {status}")
                        continue

                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            subject = "(No Subject)"
                            try:
                                subj_hdr = decode_header(msg["subject"])[0]
                                subject = subj_hdr[0].decode(subj_hdr[1] or 'utf-8') if isinstance(subj_hdr[0], bytes) else subj_hdr[0]
                            except Exception:
                                logging.warning(f"Could not decode subject for email ID {e_id.decode()}")

                            logging.info(f"Processing email ID {e_id.decode()} - Subject: '{subject}'")

                            for part in msg.walk():
                                # Skip non-attachment parts
                                if part.get_content_maintype() == 'multipart' or part.get('Content-Disposition') is None:
                                    continue

                                filename = part.get_filename()
                                if filename:
                                    total_attachments += 1
                                    cleaned_filename = clean_filename(filename)
                                    # Ensure the cleaned filename has a .pdf extension if it's supposed to be a PDF
                                    if part.get_content_type() == 'application/pdf' and not cleaned_filename.lower().endswith('.pdf'):
                                         cleaned_filename += ".pdf" # Append if missing after cleaning

                                    if part.get_content_type() == 'application/pdf':
                                        file_content = part.get_payload(decode=True)

                                        if not file_content:
                                            logging.warning(f"Skipping empty attachment: '{filename}' (cleaned: '{cleaned_filename}')")
                                            skipped_empty += 1
                                            continue

                                        if not is_valid_pdf(file_content):
                                            logging.warning(f"Skipping invalid PDF: '{filename}' (cleaned: '{cleaned_filename}')")
                                            skipped_invalid += 1
                                            continue

                                        # Define S3 key using the PO_DUMP_FOLDER
                                        key = f"{PO_DUMP_FOLDER}{cleaned_filename}"

                                        if not file_exists_in_s3(S3_BUCKET, key):
                                            if upload_to_s3(file_content, S3_BUCKET, key):
                                                processed_files += 1
                                            else:
                                                # Upload failed, log is in upload_to_s3
                                                pass
                                        else:
                                            logging.info(f"Skipping duplicate file (already exists in S3): '{key}'")
                                            skipped_duplicates += 1
                                    else:
                                         logging.info(f"Skipping non-PDF attachment: '{filename}' (Content-Type: {part.get_content_type()})")

                except Exception as e:
                    logging.error(f"Error processing email ID {e_id.decode()}: {str(e)}", exc_info=True) # Add traceback

        logging.info("--- Email Processing Summary ---")
        logging.info(f"Emails Scanned: {email_count}")
        logging.info(f"Total Attachments Found: {total_attachments}")
        logging.info(f"New Valid PDFs Uploaded: {processed_files}")
        logging.info(f"Skipped Duplicates (Already in S3): {skipped_duplicates}")
        logging.info(f"Skipped Invalid/Corrupt PDFs: {skipped_invalid}")
        logging.info(f"Skipped Empty Attachments: {skipped_empty}")
        logging.info("---------------------------------")

    except imaplib.IMAP4.error as imap_err:
        logging.error(f"IMAP Error: {imap_err}")
    except Exception as e:
        logging.error(f"Email processing failed unexpectedly: {str(e)}", exc_info=True) # Add traceback
    finally:
        logging.info("Email processing function finished.")


def create_faiss_index_po():
    """Create/Update FAISS index from new PDFs in S3 PO_Dump folder and upload to S3."""
    all_new_chunks = []
    processed_new_pdfs = 0
    uploaded_index_files = 0
    s3_index_exists = False
    existing_vector_store = None

    try:
        # --- Step 1: Check for and Load Existing FAISS Index from S3 ---
        logging.info(f"Checking for existing FAISS index in S3 at: s3://{S3_BUCKET}/{PO_INDEX_PATH}")
        try:
            # Check if index files exist (e.g., index.faiss and index.pkl)
            s3_client.head_object(Bucket=S3_BUCKET, Key=f"{PO_INDEX_PATH}index.faiss")
            s3_client.head_object(Bucket=S3_BUCKET, Key=f"{PO_INDEX_PATH}index.pkl")
            s3_index_exists = True
            logging.info("Existing FAISS index found in S3.")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info("No existing FAISS index found in S3. A new index will be created.")
                s3_index_exists = False
            else:
                logging.warning(f"Could not check for existing S3 index due to error: {e}. Assuming no index exists.")
                s3_index_exists = False

        if s3_index_exists:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    logging.info("Downloading existing index from S3...")
                    s3_client.download_file(S3_BUCKET, f"{PO_INDEX_PATH}index.faiss", os.path.join(temp_dir, "index.faiss"))
                    s3_client.download_file(S3_BUCKET, f"{PO_INDEX_PATH}index.pkl", os.path.join(temp_dir, "index.pkl"))
                    logging.info("Loading existing index into FAISS...")
                    # Load the index allowing potentially unsafe pickle loading (standard for FAISS/Langchain)
                    existing_vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
                    logging.info(f"Successfully loaded existing index with {existing_vector_store.index.ntotal} vectors.")
                except Exception as load_err:
                    logging.error(f"Failed to download or load existing FAISS index: {load_err}. Proceeding to create a new index.", exc_info=True)
                    existing_vector_store = None # Reset if loading failed
                    s3_index_exists = False # Treat as if it doesn't exist

        # --- Step 2: List PDFs in the PO_Dump folder ---
        logging.info(f"Listing objects in S3 bucket '{S3_BUCKET}' with prefix '{PO_DUMP_FOLDER}'")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PO_DUMP_FOLDER)
        pdf_keys_to_process = []

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Process only actual .pdf files, not the folder prefix itself or processed files
                    if key.lower().endswith('.pdf') and not key.lower().endswith('_processed.pdf') and key != PO_DUMP_FOLDER:
                        pdf_keys_to_process.append(key)

        if not pdf_keys_to_process:
            logging.info("No new (unprocessed) PDF files found in S3 PO_Dump folder.")
            if not s3_index_exists:
                 logging.warning("No PDFs found and no existing index. FAISS index cannot be created.")
            else:
                 logging.info("No new PDFs to add to the existing index.")
            return # Exit if no new PDFs and no existing index, or if nothing to add

        logging.info(f"Found {len(pdf_keys_to_process)} new PDF(s) to process.")

        # --- Step 3: Process New PDFs ---
        for key in pdf_keys_to_process:
            try:
                logging.info(f"Processing PDF: {key}")
                pdf_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                pdf_content = pdf_obj['Body'].read()

                if not is_valid_pdf(pdf_content):
                    logging.warning(f"Skipping invalid PDF file from S3: {key}")
                    # Optionally, rename invalid files too to avoid reprocessing
                    invalid_key = key.replace('.pdf', '_invalid.pdf')
                    logging.info(f"Renaming invalid PDF in S3 to: {invalid_key}")
                    s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': key}, Key=invalid_key)
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
                    continue

                # Extract filename for logging within process_pdf_content
                pdf_filename_for_log = os.path.basename(key)
                chunks = process_pdf_content(pdf_content, pdf_filename_for_log)

                if chunks:
                    all_new_chunks.extend(chunks)
                    processed_new_pdfs += 1

                    # Mark the PDF as processed by renaming it in S3
                    processed_key = key.replace('.pdf', '_processed.pdf')
                    logging.info(f"Renaming processed PDF in S3 to: {processed_key}")
                    s3_client.copy_object(
                        Bucket=S3_BUCKET,
                        CopySource={'Bucket': S3_BUCKET, 'Key': key},
                        Key=processed_key
                    )
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
                else:
                    logging.warning(f"No text chunks extracted from {key}. Marking as processed without adding to index.")
                    # Still rename to avoid reprocessing empty/unextractable PDFs
                    processed_key = key.replace('.pdf', '_processed_empty.pdf') # Different suffix?
                    logging.info(f"Renaming empty/unextractable PDF in S3 to: {processed_key}")
                    s3_client.copy_object( Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': key}, Key=processed_key)
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=key)


            except s3_client.exceptions.ClientError as s3_err:
                logging.error(f"S3 error processing file {key}: {s3_err}", exc_info=True)
            except Exception as e:
                logging.error(f"Error processing file {key}: {e}", exc_info=True)


        # --- Step 4: Update or Create FAISS Index ---
        if not all_new_chunks and existing_vector_store is None:
            logging.warning("No valid text chunks extracted from any new PDFs and no existing index. Cannot create or update FAISS index.")
            return

        vector_store_to_save = None
        if existing_vector_store:
            logging.info("Adding new text chunks to the existing FAISS index...")
            if all_new_chunks:
                try:
                    existing_vector_store.add_texts(all_new_chunks)
                    vector_store_to_save = existing_vector_store
                    logging.info(f"Added {len(all_new_chunks)} new chunks. Index now contains {existing_vector_store.index.ntotal} vectors.")
                except Exception as add_err:
                     logging.error(f"Failed to add new texts to existing index: {add_err}. Saving index without new additions.", exc_info=True)
                     vector_store_to_save = existing_vector_store # Save the original loaded index
            else:
                logging.info("No new chunks to add. The existing index remains unchanged.")
                # No need to re-save if nothing changed, unless we want to update timestamp/metadata?
                # For simplicity, we'll skip saving if no changes were made to an existing index.
                logging.info("FAISS index update skipped as no new documents were successfully processed.")
                return

        elif all_new_chunks: # No existing index, but we have new chunks
            logging.info("Creating a new FAISS index from extracted text chunks...")
            try:
                vector_store_to_save = FAISS.from_texts(all_new_chunks, embeddings)
                logging.info(f"Created new FAISS index with {vector_store_to_save.index.ntotal} vectors.")
            except Exception as create_err:
                logging.error(f"Failed to create new FAISS index: {create_err}", exc_info=True)
                return # Cannot proceed if index creation fails
        else:
             # Should have been caught earlier, but as a safeguard:
             logging.error("Internal state error: No existing index and no new chunks, but reached saving stage.")
             return


        # --- Step 5: Save Updated/New Index to S3 ---
        if vector_store_to_save:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    index_path_local = os.path.join(temp_dir, "faiss_index")
                    vector_store_to_save.save_local(index_path_local)
                    logging.info(f"FAISS index saved locally to temporary directory: {index_path_local}")

                    # Upload index files (.faiss, .pkl)
                    for file_name in os.listdir(index_path_local):
                        local_file = os.path.join(index_path_local, file_name)
                        # Ensure target S3 path uses forward slashes
                        s3_key = f"{PO_INDEX_PATH}{file_name}".replace("\\", "/")

                        logging.info(f"Uploading {local_file} to S3 key: {s3_key}")
                        with open(local_file, 'rb') as f:
                            s3_client.put_object(
                                Bucket=S3_BUCKET,
                                Key=s3_key,
                                Body=f
                            )
                            uploaded_index_files += 1
                        logging.info(f"Successfully uploaded: {s3_key}")

                except Exception as save_upload_err:
                     logging.error(f"Error saving index locally or uploading to S3: {save_upload_err}", exc_info=True)
                     # Decide if partial upload is acceptable or if retry logic is needed
                     return # Stop if saving/upload fails

            logging.info(f"--- FAISS Index Creation/Update Summary ---")
            logging.info(f"Processed {processed_new_pdfs} new PDF(s).")
            logging.info(f"Extracted and added/indexed {len(all_new_chunks)} text chunks.")
            logging.info(f"Uploaded {uploaded_index_files} FAISS index file(s) to S3 path: s3://{S3_BUCKET}/{PO_INDEX_PATH}")
            logging.info("------------------------------------------")
        else:
             logging.info("No vector store was generated or updated, skipping save to S3.")


    except s3_client.exceptions.ClientError as s3_err:
         logging.error(f"An S3 ClientError occurred during FAISS index processing: {s3_err}", exc_info=True)
    except Exception as e:
        logging.error(f"FAISS index creation/update failed: {str(e)}", exc_info=True)
    finally:
        logging.info("FAISS index processing function finished.")


if __name__ == "__main__":
    logging.info("Starting script execution...")
    try:
        process_po_emails()  # Fetch PO PDFs from emails and upload new ones to S3
        create_faiss_index_po() # Create/Update FAISS index for PO PDFs in S3
        logging.info("Script execution completed successfully.")
    except Exception as main_err:
         logging.critical(f"Script failed during execution: {main_err}", exc_info=True)
         # Consider exit codes or further alerting here