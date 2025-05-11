# email_utils.py
import imaplib
import email
from email.header import decode_header
import re
from datetime import datetime, timedelta
from streamlit import secrets
import PyPDF2  # For PDF processing
import io  # For PDF processing
import time

from whatsapp_utils import config  # For default IMAP server
from whatsapp_utils.database_manager import is_email_processed, \
    mark_email_as_processed  # Circular import if this file imports from db_manager too early. Be careful.

# To avoid circular imports, db_conn can be passed to functions here, or status updates can be returned to caller.
# For this structure, let's assume the main email processing orchestrator passes db_conn.

GMAIL_EMAIL = secrets.get("gmail_uname", "your_email@example.com")
GMAIL_PASSWORD = secrets.get("gmail_pwd", "your_password")
IMAP_SERVER_URL = secrets.get("IMAP_SERVER", config.IMAP_SERVER_DEFAULT)


def decode_mail_header_robust(header_value):
    if header_value is None: return ""
    try:
        decoded_parts = decode_header(header_value)
        header_str = ""
        for part_content, charset in decoded_parts:
            if isinstance(part_content, bytes):
                header_str += part_content.decode(charset or "utf-8", "replace")
            else:
                header_str += part_content
        return header_str.strip()
    except Exception as e_dec:
        print(f"EU: Header decode error: {e_dec}. Original: {header_value}")
        return str(header_value)  # Fallback


def extract_text_from_pdf_bytes(pdf_file_bytes):  # Renamed to avoid conflict if st is imported
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file_bytes))
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
        print(f"EU: Error extracting text from PDF: {e}")
        # If Streamlit is used here: from streamlit import error; error(f"Error extracting text from PDF: {str(e)}")
        return ""


def get_email_body_text_content(msg):  # Renamed
    text_parts, html_parts = [], []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type();
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                try:
                    text_parts.append(
                        part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', "replace"))
                except:
                    pass
            elif ctype == 'text/html' and 'attachment' not in cdispo:
                try:
                    html_parts.append(
                        part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', "replace"))
                except:
                    pass
    else:
        if msg.get_content_type() == 'text/plain':
            try:
                text_parts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', "replace"))
            except:
                pass
        elif msg.get_content_type() == 'text/html':
            try:
                html_parts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', "replace"))
            except:
                pass
    if text_parts: return "\n".join(text_parts).strip()
    if html_parts:
        html_content = "\n".join(html_parts).strip()
        text_content = re.sub('<style[^<]+?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
        text_content = re.sub('<script[^<]+?</script>', '', text_content, flags=re.IGNORECASE | re.DOTALL)
        text_content = re.sub('<[^<]+?>', ' ', text_content)
        text_content = re.sub(r'\s+', ' ', text_content)
        return text_content.strip()
    return ""


def fetch_and_prepare_emails(db_conn, subjects_to_monitor, only_recent_days=None):
    """
    Connects to IMAP, searches for UNSEEN emails matching subjects,
    filters out already processed UIDs, and yields new email messages.
    """
    mail = None
    emails_to_yield = []
    print(f"EU: [{datetime.now()}] Starting email fetch cycle...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER_URL)
        mail.login(GMAIL_EMAIL, GMAIL_PASSWORD)
        mail.select("inbox")

        if not subjects_to_monitor: print("EU: No subjects to monitor."); return []

        escaped_subjects = [s.replace("\"", "\\\"").replace("(", "\\(").replace(")", "\\)") for s in
                            subjects_to_monitor]
        base_subject_query = ""
        if len(escaped_subjects) == 1:
            base_subject_query = f'SUBJECT "{escaped_subjects[0]}"'
        elif len(escaped_subjects) > 1:
            base_subject_query = f'(SUBJECT "{escaped_subjects[-1]}")'
            for i in range(len(escaped_subjects) - 2, -1, -1):
                base_subject_query = f'(OR (SUBJECT "{escaped_subjects[i]}") {base_subject_query})'
        else:
            print("EU: No valid subjects after escaping."); return []

        search_parts = ['(UNSEEN'];
        search_parts.append(base_subject_query)
        if only_recent_days and isinstance(only_recent_days, int) and only_recent_days > 0:
            date_since = (datetime.now() - timedelta(days=only_recent_days)).strftime("%d-%b-%Y")
            search_parts.append(f'SINCE "{date_since}"')
        search_parts.append(')')
        full_search_criteria = ' '.join(search_parts)

        print(f"EU: IMAP Search Criteria: {full_search_criteria}")
        status, msg_numbers_data = mail.search(None, full_search_criteria)
        if status != "OK": print(f"EU: IMAP search failed: {status}"); return []
        if not msg_numbers_data[0]: print(f"EU: No UNSEEN emails matching criteria."); return []

        new_email_uids_to_fetch = []
        for num_str in msg_numbers_data[0].split():
            if not num_str.strip(): continue
            status_uid, uid_data = mail.fetch(num_str, '(UID)')
            if status_uid == 'OK' and uid_data[0]:
                uid_match = re.search(r'UID\s+(\d+)', uid_data[0].decode())
                if uid_match:
                    email_uid = uid_match.group(1)
                    if not is_email_processed(db_conn, email_uid):  # Check DB
                        new_email_uids_to_fetch.append(email_uid)
                    else:
                        print(f"EU: Email UID {email_uid} already processed. Skipping fetch.")
            time.sleep(0.05)

        if not new_email_uids_to_fetch: print("EU: No new emails to fetch after UID check."); return []
        print(f"EU: Found {len(new_email_uids_to_fetch)} new email UIDs to fetch full content.")

        for email_uid in new_email_uids_to_fetch:
            print(f"EU: Fetching RFC822 for UID: {email_uid}")
            status_fetch, msg_data_uid = mail.uid('fetch', email_uid, '(RFC822)')
            if status_fetch != 'OK' or not msg_data_uid or not msg_data_uid[0]:
                print(f"EU: Failed to fetch email for UID {email_uid}. Status: {status_fetch}")
                mark_email_as_processed(db_conn, email_uid, "N/A", "N/A", "FETCH_ERROR_UID")
                continue
            if not isinstance(msg_data_uid[0], tuple) or len(msg_data_uid[0]) < 2:
                print(f"EU: Unexpected fetch data format for UID {email_uid}")
                mark_email_as_processed(db_conn, email_uid, "N/A", "N/A", "FETCH_DATA_ERROR_UID")
                continue

            raw_email = msg_data_uid[0][1]
            msg_object = email.message_from_bytes(raw_email)
            email_subject = decode_mail_header_robust(msg_object["Subject"])
            email_sender = decode_mail_header_robust(msg_object.get("From"))

            emails_to_yield.append({
                "uid": email_uid,
                "msg": msg_object,
                "subject": email_subject,
                "sender": email_sender,
                "mail_connection": mail  # Pass the connection for marking as read later
            })
            time.sleep(0.1)  # Small delay after fetching each full email

        return emails_to_yield  # Returns list of dicts containing msg objects and metadata

    except imaplib.IMAP4.abort as e_abort:
        print(f"EU: IMAP aborted: {e_abort}"); return []
    except Exception as e:
        print(f"EU: Email fetching error: {e}"); import traceback; traceback.print_exc(); return []
    finally:
        # Logout is tricky here if we return mail_connection.
        # The orchestrator function should handle logout.
        # If emails_to_yield is empty or error, then logout.
        if not emails_to_yield and mail:
            try:
                mail.close(); mail.logout(); print("EU: IMAP Logged out (no emails to process or error).")
            except:
                pass
        elif mail and emails_to_yield:
            print(f"EU: IMAP connection kept open for {len(emails_to_yield)} emails. Caller must logout.")