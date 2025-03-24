import imaplib
import email
import os
import logging
from email.header import decode_header
import toml
import streamlit as st

# Configuration
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
IMAP_SERVER = "imap.gmail.com"

# Load secrets
secrets = toml.load(SECRETS_FILE_PATH)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
    name, ext = os.path.splitext(filename)
    cleaned_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
    return f"{cleaned_name}.pdf" if ext.lower() == '.pdf' else f"{cleaned_name}.pdf"

def fetch_proforma_emails():
    """Fetch PDFs from emails with 'Proforma Invoice' in the subject."""
    pdf_files = []
    try:
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(st.secrets["gmail_uname"], st.secrets["gmail_pwd"])
            logging.info("Email authentication successful")
            mail.select("inbox")
            status, email_ids = mail.search(None, '(SUBJECT "Proforma Invoice")')
            if status != "OK":
                logging.warning("No matching emails found")
                return pdf_files

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
                                    if file_content:
                                        pdf_files.append((filename, file_content))
                except Exception as e:
                    logging.error(f"Error processing email {e_id}: {str(e)}")
            logging.info(f"Fetched {len(pdf_files)} PDFs from emails.")
    except Exception as e:
        logging.error(f"Email processing failed: {str(e)}")
    return pdf_files