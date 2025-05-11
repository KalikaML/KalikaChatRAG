import streamlit as st
import pywhatkit  # Make sure it's installed
import pyautogui  # Make sure it's installed and platform dependencies met
from datetime import timedelta, datetime
import time
import imaplib
import email
from email.header import decode_header
import PyPDF2  # Make sure it's installed
import re
import io
import threading
import phonenumbers  # Make sure it's installed
import schedule  # Make sure it's installed
import random
import webbrowser
import psycopg2  # Make sure psycopg2-binary is installed
import os  # Not strictly used for secrets if using st.secrets
from streamlit import secrets  # Use this for Streamlit secrets
import pandas as pd  # Make sure it's installed
import matplotlib.pyplot as plt  # Make sure it's installed

# --- Configuration (Conceptually config.py) ---
EMAIL = st.secrets.get("gmail_uname", "your_email@example.com")
PASSWORD = st.secrets.get("gmail_pwd", "your_password")
IMAP_SERVER = st.secrets.get("IMAP_SERVER", "imap.gmail.com")

SELLER_TEAM_RECIPIENTS_STR = st.secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS", "")

DB_HOST = st.secrets.get("DB_HOST", "localhost")
DB_NAME = st.secrets.get("DB_NAME", "po_orders")  # Renamed for clarity
DB_USER = st.secrets.get("DB_USER", "db_user")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "db_password123")
DB_PORT = st.secrets.get("DB_PORT", 5432)

EMAIL_SUBJECTS_TO_MONITOR = st.secrets.get("EMAIL_SUBJECTS_LIST", [
    "PO released // Consumable items", "PO copy", "import po",
    "RFQ-Polybag", "PFA PO", "Purchase Order FOR", "Purchase Order_"
])


# Ensure EMAIL_SUBJECTS_LIST in secrets.toml is a list of strings
# Example secrets.toml:
# EMAIL_SUBJECTS_LIST = [
#   "PO released // Consumable items",
#   "PO copy",
#   # ... other subjects
# ]

# --- End Configuration ---

# --- Database Management (Conceptually db_manager.py) ---
def create_db_tables(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    order_type TEXT,
                    product_name TEXT,
                    category TEXT,
                    price REAL,
                    quantity INTEGER,
                    unit TEXT,
                    specifications TEXT,
                    order_date TIMESTAMP,
                    delivery_date DATE,
                    customer_name TEXT,
                    customer_phone TEXT,
                    email TEXT,
                    address TEXT,
                    payment_method TEXT,
                    payment_status TEXT,
                    order_status TEXT,
                    message_sent BOOLEAN DEFAULT FALSE,
                    source_email_subject TEXT,
                    source_email_sender TEXT,
                    source_email_uid TEXT UNIQUE,
                    raw_text_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_type ON orders(order_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_source_email_uid ON orders(source_email_uid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_message_sent ON orders(message_sent);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_emails (
                    email_uid TEXT PRIMARY KEY,
                    subject TEXT,
                    sender TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    related_order_id INTEGER REFERENCES orders(id) ON DELETE SET NULL
                );
            """)
        conn.commit()
        print("Database tables checked/created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating/checking database tables: {e}")
        if conn: conn.rollback()


def connect_to_db_main():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        print("Database connection successful.")
        create_db_tables(conn)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during database connection: {e}")
        return None


def is_email_processed(conn, email_uid):
    if not conn or not email_uid: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM processed_emails WHERE email_uid = %s", (email_uid,))
            return cursor.fetchone() is not None
    except psycopg2.Error as e:
        print(f"Error checking if email UID {email_uid} is processed: {e}")
        return False


def mark_email_as_processed(conn, email_uid, subject, sender, status, related_order_id=None):
    if not conn or not email_uid: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO processed_emails (email_uid, subject, sender, status, related_order_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (email_uid) DO UPDATE SET
                    subject = EXCLUDED.subject, sender = EXCLUDED.sender,
                    processed_at = CURRENT_TIMESTAMP, status = EXCLUDED.status,
                    related_order_id = EXCLUDED.related_order_id;
            """, (email_uid, subject, sender, status, related_order_id))
        conn.commit()
        print(f"Email UID {email_uid} marked as processed with status: {status}")
        return True
    except psycopg2.Error as e:
        print(f"Error marking email UID {email_uid} as processed: {e}")
        if conn: conn.rollback()
        return False


def store_parsed_order_or_rfq(conn, details):
    if conn is None: print("DB connection not available for storing."); return None
    try:
        if not details.get("order_type") or (not details.get("product_name") and not details.get("specifications")):
            print(f"Skipping store: Missing order_type or product details. Data: {details}")
            return None
        # Ensure source_email_uid is present if this record comes from an email
        if not details.get("source_email_uid") and "TEXT" in details.get("order_type", "") or "PDF" in details.get(
                "order_type", ""):
            print(f"Critical: source_email_uid missing for email-derived order/rfq. Data: {details}. Skipping store.")
            return None

        with conn.cursor() as cursor:
            columns = [
                "order_type", "product_name", "category", "price", "quantity", "unit",
                "specifications", "order_date", "delivery_date", "customer_name",
                "customer_phone", "email", "address", "payment_method",
                "payment_status", "order_status", "message_sent",
                "source_email_subject", "source_email_sender", "source_email_uid",
                "raw_text_content"
            ]
            values = [
                details.get("order_type"), details.get("product_name"), details.get("category"),
                details.get("price"), details.get("quantity"), details.get("unit"),
                details.get("specifications"), details.get("order_date"), details.get("delivery_date"),
                details.get("customer_name"), details.get("customer_phone"), details.get("email"),
                details.get("address"), details.get("payment_method"), details.get("payment_status"),
                details.get("order_status", "PENDING"),
                details.get("message_sent", False),
                details.get("source_email_subject"), details.get("source_email_sender"),
                details.get("source_email_uid"), details.get("raw_text_content")
            ]
            sql = f"INSERT INTO orders ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) RETURNING id;"
            cursor.execute(sql, tuple(values))
            new_order_id = cursor.fetchone()[0]
        conn.commit()
        print(
            f"{details.get('order_type')} stored with ID: {new_order_id}. Product: {details.get('product_name') or details.get('specifications')}")
        if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.pending_msg_count = None
        return new_order_id
    except psycopg2.Error as e:
        if conn: conn.rollback()
        if hasattr(e, 'pgcode') and e.pgcode == '23505':  # Unique violation (likely source_email_uid)
            print(
                f"Constraint Violated: Email UID {details.get('source_email_uid')} likely already processed into an order. {e}")
        else:
            print(f"DB error storing order/rfq: {e} (Code: {getattr(e, 'pgcode', 'N/A')})")
        return None
    except Exception as e:
        if conn: conn.rollback()
        print(f"Unexpected error storing order/rfq: {e}")
        return None


# --- End Database Management ---

# --- Helper Functions (General) ---
def extract_text_from_pdf(pdf_file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file_bytes))
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        if 'st' in globals(): st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def format_phone_number(phone_str):
    if not phone_str or not isinstance(phone_str, str): return None
    try:
        p_num = phonenumbers.parse(phone_str, "IN" if not phone_str.startswith('+') else None)
        return phonenumbers.format_number(p_num, phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(
            p_num) else None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None


def get_seller_team_recipients(recipients_str):
    recipients = set()
    if recipients_str and isinstance(recipients_str, str):
        for phone in recipients_str.split(","):
            fmt_seller = format_phone_number(phone.strip())
            if fmt_seller:
                recipients.add(fmt_seller)
            else:
                print(f"Warning: Invalid seller phone in config: {phone.strip()}")
    if not recipients and 'st' in globals(): st.warning("No valid seller WhatsApp recipients configured in secrets.")
    return list(recipients)


# --- End Helper Functions ---


# --- Order Parsing (Conceptually order_parser.py) ---
def parse_order_details_from_pdf_text(text, source_email_uid, email_subject, email_sender):
    print(f"Parsing PDF text for email UID {source_email_uid}...")
    parsed_data = {"order_type": "PO_PDF", "source_email_uid": source_email_uid,
                   "source_email_subject": email_subject, "source_email_sender": email_sender,
                   "raw_text_content": text[:5000]}

    patterns = {  # Simplified, use your more detailed patterns
        "product_name": r"Product(?: Name)?:?\s*(.+?)(?:\nPrice:|\nQuantity:|$)",
        "price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "quantity": r"Quantity:?\s*(\d+)",
        "customer_name": r"Customer(?: Name)?:?\s*(.+?)(?:\nPhone:|\nAddress:|$)",
        "order_date_str": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == "price":
                parsed_data[key] = float(value.replace(",", "")) if value else None
            elif key == "quantity":
                parsed_data[key] = int(value) if value else None
            elif key == "order_date_str":
                try:
                    parsed_data["order_date"] = datetime.strptime(value.split()[0], "%Y-%m-%d")
                except:
                    parsed_data["order_date"] = datetime.now()  # Fallback
            else:
                parsed_data[key] = value

    parsed_data["product_name"] = parsed_data.get("product_name", "N/A from PDF")
    parsed_data["customer_name"] = parsed_data.get("customer_name", email_sender.split('<')[0].strip())
    parsed_data["email"] = email_sender.split('<')[-1].strip('>').strip() if '<' in email_sender else email_sender
    parsed_data["order_date"] = parsed_data.get("order_date", datetime.now())
    parsed_data["order_status"] = "PROCESSING" if parsed_data.get(
        "product_name") != "N/A from PDF" else "NEEDS_REVIEW_PDF"

    if parsed_data.get("product_name") == "N/A from PDF" and not parsed_data.get("specifications"):
        print(f"Could not parse essential PO details from PDF for UID {source_email_uid}.")
        return None
    return parsed_data


def parse_details_from_email_body(email_body, source_email_uid, email_subject, email_sender):
    print(f"Parsing email body text for UID {source_email_uid}...")
    parsed_items_list = []
    order_type = "RFQ_TEXT"
    customer_name = email_sender.split('<')[0].strip()
    contact_email = email_sender.split('<')[-1].strip('>').strip() if '<' in email_sender else email_sender

    body_lower = email_body.lower();
    subject_lower = email_subject.lower()
    if any(kw in subject_lower for kw in ["purchase order", "po copy", "pfa po"]) or \
            any(kw in body_lower for kw in ["order for", "confirming order", "your order"]):
        order_type = "PO_TEXT";
        order_status = "PENDING_CONFIRMATION_TEXT"
    elif any(kw in subject_lower for kw in ["rfq", "quote", "quotation"]) or \
            any(kw in body_lower for kw in ["offer for", "pls share offer", "request for quotation", "send quote"]):
        order_type = "RFQ_TEXT";
        order_status = "PENDING_QUOTE"
    else:
        order_type = "INQUIRY_TEXT"; order_status = "NEEDS_REVIEW_TEXT"

    # Refined Polybag Pattern (Example)
    polybag_pattern = re.compile(
        r"""
        ^\s*(?P<quantity_val>\d+)?\s* # Optional quantity at start
        (?P<product_type>LDPE\s*(?:COVER|BAG|SHEET)|POLYBAG(?:S)?|BAGS?)[\s,.:-]* # Product type
        (?:(?:SIZE|DIMENSIONS?)\s*[:\s]*)?                                  # Optional "SIZE" or "DIMENSION"
        (?:(?P<width>[\d\.\"\'\s]+(?:CM|MM|INCH|\"|X|x))[\s,X*x]+)?         # Width
        (?:(?P<length>[\d\.\"\'\s]+(?:CM|MM|INCH|\"|X|x))[\s,X*x]+)?        # Length
        (?:(?P<height_gusset>[\d\.\"\'\s]+(?:CM|MM|INCH|\"))?[\s,X*x]*)?     # Optional Height/Gusset
        (?:[\s,]*\b(?P<thickness>[\d\.]+)\s*(?:MICRON|MIC|GAUGE|MIL)\b)?     # Thickness
        (?:[\s,.]*(?P<features>[A-Z\s\d\/,-]+(?:FLAP|PRINT|SEAL|COLOR|COLOUR|PLAIN|NATURAL|TRANSPARENT)[A-Z\s\d\/,-]*))? # Features
        (?:[\s,-]*QUANTITY\s*:\s*(?P<quantity_end>\d+))?                    # Optional quantity at end
        """, re.VERBOSE | re.IGNORECASE
    )

    found_items_flag = False
    lines = [line.strip() for line in email_body.splitlines() if line.strip()]  # Get non-empty lines

    for i, line in enumerate(lines):
        if len(line) < 10 and not any(kw in line.lower() for kw in ["ldpe", "bag", "cover"]): continue  # Basic filter

        match = polybag_pattern.search(line)
        # If no match, try joining with next line if it looks like a continuation
        if not match and i + 1 < len(lines) and len(lines[i + 1]) < 50 and not polybag_pattern.search(lines[i + 1]):
            combined_line = line + " " + lines[i + 1]
            match = polybag_pattern.search(combined_line)
            if match: print(f"  Matched on combined line: {combined_line}")

        if match:
            found_items_flag = True
            item_data = match.groupdict()
            spec_parts = []
            if item_data.get('product_type'): spec_parts.append(item_data['product_type'].strip())
            dimensions = []
            if item_data.get('width'): dimensions.append(item_data['width'].strip())
            if item_data.get('length'): dimensions.append(item_data['length'].strip())
            if item_data.get('height_gusset'): dimensions.append(item_data['height_gusset'].strip())
            if dimensions: spec_parts.append("x".join(filter(None, dimensions)))
            if item_data.get('thickness'): spec_parts.append(
                f"{item_data['thickness'].strip()} MICRON")  # Assuming micron if only number
            if item_data.get('features'): spec_parts.append(item_data['features'].strip().upper())

            final_spec = ", ".join(filter(None, spec_parts))
            quantity = item_data.get('quantity_val') or item_data.get('quantity_end')

            parsed_items_list.append({
                "order_type": order_type, "product_name": item_data.get('product_type', "Polythene Item").strip(),
                "specifications": final_spec if final_spec else line,  # Fallback to raw line
                "quantity": int(quantity) if quantity else None, "unit": "PCS" if quantity else None,  # Assuming PCS
                "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
                "order_status": order_status, "source_email_uid": source_email_uid,
                "source_email_subject": email_subject, "source_email_sender": email_sender,
                "raw_text_content": email_body[:5000]
            })
        # If no polybag match, and line looks like a generic item, add it (more heuristic)
        elif not match and order_type != "INQUIRY_TEXT" and len(line) > 15:  # Heuristic for other items
            # This part needs very careful regex for other types of products if any
            if any(kw in line.lower() for kw in ["item:", "product:", "detail:"]) or (
                    len(line.split()) > 3 and any(char.isdigit() for char in line)):
                print(f"  Found generic item line: {line}")
                found_items_flag = True
                parsed_items_list.append({
                    "order_type": order_type, "product_name": f"Item from email text",
                    "specifications": line, "quantity": None, "unit": None,
                    "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
                    "order_status": order_status, "source_email_uid": source_email_uid,
                    "source_email_subject": email_subject, "source_email_sender": email_sender,
                    "raw_text_content": email_body[:5000]
                })

    if not found_items_flag and order_type == "INQUIRY_TEXT" and len(email_body) > 30:  # Log generic inquiry
        parsed_items_list.append({
            "order_type": order_type, "product_name": "General Inquiry",
            "specifications": email_subject, "raw_text_content": email_body[:5000],
            "customer_name": customer_name, "email": contact_email, "order_date": datetime.now(),
            "order_status": order_status, "source_email_uid": source_email_uid,
            "source_email_subject": email_subject, "source_email_sender": email_sender
        })
    elif not found_items_flag:
        print(f"No specific items or parsable generic inquiry found in body for UID {source_email_uid}")
        return []

    return parsed_items_list


def get_email_body_text(msg):
    text_parts = [];
    html_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type();
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                try:
                    text_parts.append(
                        part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', "replace"))
                except:
                    pass  # Ignore decoding errors for a part
            elif ctype == 'text/html' and 'attachment' not in cdispo:
                try:
                    html_parts.append(
                        part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', "replace"))
                except:
                    pass
    else:  # Not multipart
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
    if html_parts:  # Basic HTML to text conversion
        html_content = "\n".join(html_parts).strip()
        text_content = re.sub('<style[^<]+?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)  # remove style
        text_content = re.sub('<script[^<]+?</script>', '', text_content,
                              flags=re.IGNORECASE | re.DOTALL)  # remove script
        text_content = re.sub('<[^<]+?>', ' ', text_content)  # remove all other tags, replace with space
        text_content = re.sub(r'\s+', ' ', text_content)  # normalize whitespace
        return text_content.strip()
    return ""


# --- End Order Parsing ---

# --- Email Processing (Corrected and using UID logic) ---
def process_incoming_emails(db_conn, subjects_to_monitor, only_recent_days=None, mark_as_read=True):
    processed_order_count = 0;
    processed_rfq_count = 0;
    mail = None
    print(f"[{datetime.now()}] Starting comprehensive email processing...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER);
        mail.login(EMAIL, PASSWORD);
        mail.select("inbox")
        if not subjects_to_monitor: print("No subjects to monitor."); return 0, 0

        escaped_subjects = [s.replace("\"", "\\\"").replace("(", "\\(").replace(")", "\\)") for s in
                            subjects_to_monitor]  # Escape more chars
        base_subject_query = ""
        if len(escaped_subjects) == 1:
            base_subject_query = f'SUBJECT "{escaped_subjects[0]}"'
        elif len(escaped_subjects) > 1:
            base_subject_query = f'(SUBJECT "{escaped_subjects[-1]}")'
            for i in range(len(escaped_subjects) - 2, -1, -1):
                base_subject_query = f'(OR (SUBJECT "{escaped_subjects[i]}") {base_subject_query})'
        else:
            print("No valid subjects after escaping."); return 0, 0

        search_parts = ['(UNSEEN'];
        search_parts.append(base_subject_query)
        if only_recent_days and isinstance(only_recent_days, int) and only_recent_days > 0:
            date_since = (datetime.now() - timedelta(days=only_recent_days)).strftime("%d-%b-%Y")
            search_parts.append(f'SINCE "{date_since}"')
        search_parts.append(')')
        full_search_criteria = ' '.join(search_parts)

        print(f"IMAP Search Criteria: {full_search_criteria}")
        status, msg_numbers_data = mail.search(None, full_search_criteria)
        if status != "OK": print(f"IMAP search failed: {status}"); return 0, 0
        if not msg_numbers_data[0]: print(f"No UNSEEN emails matching criteria."); return 0, 0

        unique_email_uids_to_process = set()
        for num_str in msg_numbers_data[0].split():
            if not num_str.strip(): continue  # Skip empty strings if any
            status_uid, uid_data = mail.fetch(num_str, '(UID)')
            if status_uid == 'OK' and uid_data[0]:
                uid_match = re.search(r'UID\s+(\d+)', uid_data[0].decode())
                if uid_match:
                    email_uid = uid_match.group(1)
                    if not is_email_processed(db_conn, email_uid):
                        unique_email_uids_to_process.add(email_uid)
                    else:
                        print(f"Email UID {email_uid} already processed. Skipping.")
            time.sleep(0.05)  # Shorter delay

        if not unique_email_uids_to_process: print("No new emails to process after UID check."); return 0, 0
        print(f"Found {len(unique_email_uids_to_process)} new email(s) by UID to process.")

        for email_uid in list(unique_email_uids_to_process):
            print(f"Processing new email UID: {email_uid}")
            status_fetch, msg_data_uid = mail.uid('fetch', email_uid, '(RFC822)')
            if status_fetch != 'OK' or not msg_data_uid or not msg_data_uid[0]:  # Check msg_data_uid[0]
                print(f"Failed to fetch email for UID {email_uid}. Status: {status_fetch}");
                mark_email_as_processed(db_conn, email_uid, "N/A", "N/A", "FETCH_ERROR_UID");
                continue

            # Ensure msg_data_uid[0] is a tuple and has the email bytes
            if not isinstance(msg_data_uid[0], tuple) or len(msg_data_uid[0]) < 2:
                print(f"Unexpected fetch data format for UID {email_uid}: {msg_data_uid[0]}");
                mark_email_as_processed(db_conn, email_uid, "N/A", "N/A", "FETCH_DATA_ERROR_UID");
                continue

            raw_email = msg_data_uid[0][1]
            msg = email.message_from_bytes(raw_email)

            def decode_mail_header(header_value):  # Local helper for safety
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
                    print(f"Header decode error: {e_dec}"); return str(header_value)  # Fallback

            email_subject = decode_mail_header(msg["Subject"])
            email_sender = decode_mail_header(msg.get("From"))
            print(f"  Subject: {email_subject}, Sender: {email_sender}")

            has_pdf_attachment = False;
            parsed_order_details_from_pdf = None
            processing_status = "NO_RELEVANT_CONTENT";
            related_order_id_for_processed_email = None

            for part in msg.walk():
                if part.get_content_type() == "application/pdf" and part.get('Content-Disposition'):
                    filename = decode_mail_header(part.get_filename()) or f'attachment_{email_uid}.pdf'
                    pdf_data = part.get_payload(decode=True)
                    if pdf_data:
                        print(f"  Found PDF: {filename}");
                        has_pdf_attachment = True
                        pdf_text = extract_text_from_pdf(pdf_data)
                        if pdf_text:
                            parsed_order_details_from_pdf = parse_order_details_from_pdf_text(pdf_text, email_uid,
                                                                                              email_subject,
                                                                                              email_sender)
                            if parsed_order_details_from_pdf:
                                processing_status = "PROCESSED_PDF_NEEDS_DB_STORE"
                            else:
                                processing_status = "PDF_NO_DETAILS"
                        else:
                            processing_status = "PDF_EMPTY_TEXT"
                        break
            if not has_pdf_attachment:
                print("  No PDF attachment found. Checking email body text...")
                email_body = get_email_body_text(msg)
                if email_body:
                    list_of_parsed_items_from_text = parse_details_from_email_body(email_body, email_uid, email_subject,
                                                                                   email_sender)
                    if list_of_parsed_items_from_text:
                        temp_order_count = 0;
                        temp_rfq_count = 0
                        for item_detail in list_of_parsed_items_from_text:
                            new_id = store_parsed_order_or_rfq(db_conn, item_detail)
                            if new_id:
                                if not related_order_id_for_processed_email: related_order_id_for_processed_email = new_id
                                if item_detail["order_type"].startswith("PO"):
                                    temp_order_count += 1
                                elif item_detail["order_type"].startswith("RFQ") or item_detail[
                                    "order_type"].startswith("INQUIRY"):
                                    temp_rfq_count += 1
                        if temp_order_count > 0 or temp_rfq_count > 0:
                            processing_status = f"TEXT_DB_STORED ({temp_order_count} PO, {temp_rfq_count} RFQ)"  # More concise
                            processed_order_count += temp_order_count;
                            processed_rfq_count += temp_rfq_count
                        else:
                            processing_status = "TEXT_ITEMS_PARSED_BUT_NOT_STORED"
                    else:
                        processing_status = "TEXT_NO_RELEVANT_ITEMS_FOUND"
                else:
                    processing_status = "EMPTY_EMAIL_BODY"

            if parsed_order_details_from_pdf and processing_status == "PROCESSED_PDF_NEEDS_DB_STORE":
                new_id = store_parsed_order_or_rfq(db_conn, parsed_order_details_from_pdf)
                if new_id:
                    related_order_id_for_processed_email = new_id
                    if parsed_order_details_from_pdf["order_type"].startswith("PO"):
                        processed_order_count += 1
                    elif parsed_order_details_from_pdf["order_type"].startswith("RFQ"):
                        processed_rfq_count += 1
                    processing_status = f"PDF_DB_STORED ({parsed_order_details_from_pdf['order_type']})"
                else:
                    processing_status = "PDF_DB_STORE_FAILED"

            mark_email_as_processed(db_conn, email_uid, email_subject, email_sender, processing_status,
                                    related_order_id=related_order_id_for_processed_email)
            if mark_as_read:
                try:
                    mail.uid('store', email_uid, '+FLAGS', '\\Seen'); print(f"  Marked UID {email_uid} as SEEN.")
                except Exception as e_seen:
                    print(f"  Failed to mark UID {email_uid} as SEEN: {e_seen}")
            time.sleep(0.1)
        return processed_order_count, processed_rfq_count
    except imaplib.IMAP4.abort as e_abort:
        print(f"IMAP aborted: {e_abort}"); return processed_order_count, processed_rfq_count
    except Exception as e:
        print(f"Email processing error: {e}"); import \
            traceback; traceback.print_exc(); return processed_order_count, processed_rfq_count
    finally:
        if mail:
            try:
                mail.close(); mail.logout(); print("IMAP Logged out.")
            except:
                pass


# --- End Email Processing ---

# --- WhatsApp Notifier (Conceptually whatsapp_notifier.py) ---
# Includes: send_whatsapp_message_web, send_whatsapp_message_pywhatkit, send_whatsapp_notification (choice)
# format_whatsapp_message_for_order, send_notifications_from_db
# (These functions are largely from previous response, ensure they are adapted for new order_type and DB fields)
def send_whatsapp_message_web(message, recipient_numbers_list):  # Keep your preferred implementation
    status_container = st.empty()
    if not isinstance(recipient_numbers_list, (list, set)) or not recipient_numbers_list:
        status_container.error("Invalid/No recipients for web WhatsApp.");
        return False
    webbrowser.open("https://web.whatsapp.com")
    status_container.info("Opened WhatsApp Web. Scan QR if needed. Waiting 25s...")
    time.sleep(25)
    sent_successfully_all = True
    for recipient in recipient_numbers_list:
        formatted_recipient = format_phone_number(recipient)  # Assumes this helper is defined
        if not formatted_recipient: st.warning(
            f"Skipping invalid WA phone: {recipient}"); sent_successfully_all = False; continue
        try:
            import urllib.parse;
            encoded_message = urllib.parse.quote(message)
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_recipient}&text={encoded_message}"
            webbrowser.open(whatsapp_url)
            status_container.info(f"Web: Opened chat for {formatted_recipient}. Waiting 15s...")
            time.sleep(15);
            pyautogui.press("enter")  # Requires pyautogui and screen focus
            status_container.success(f"Web: 'Enter' pressed for {formatted_recipient}.")
            if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.whatsapp_sent_counter += 1
            time.sleep(random.uniform(4, 7))
        except Exception as e:
            status_container.error(f"Web WA Error for {formatted_recipient}: {e}")
            if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.whatsapp_errors.append(
                f"Web WA Error for {formatted_recipient}: {str(e)}")
            sent_successfully_all = False
    return sent_successfully_all


send_whatsapp_notification = send_whatsapp_message_web  # Choose default


def format_whatsapp_message_for_item(item_data_row, db_columns_list):
    data = dict(zip(db_columns_list, item_data_row))
    item_id, item_type = data['id'], data['order_type']
    product_info = data.get('product_name') or data.get('specifications') or "N/A"
    customer = data.get('customer_name') or data.get('source_email_sender', '').split('<')[0].strip() or "Unknown"
    item_date = data.get('order_date') or data.get('created_at')
    date_str = item_date.strftime('%d-%b-%Y %H:%M') if isinstance(item_date, datetime) else str(item_date or "N/A")

    title_emoji = "📦"
    if "RFQ" in item_type:
        title_emoji = "🔔 RFQ"
    elif "PO" in item_type:
        title_emoji = "📦 PO"
    elif "INQUIRY" in item_type:
        title_emoji = "💡 Inquiry"

    title = f"*{title_emoji} Update (DB-{item_id})* ({item_type})"
    lines = [title, f"👤 From: {customer} ({data.get('email', 'N/A')})", f"🗓️ Date: {date_str}"]
    if "PO" in item_type:
        lines.append(f"📋 Item: {product_info[:150]}")
        lines.append(f"🔢 Qty: {data.get('quantity', 'N/A')} {data.get('unit', '')}")
        if data.get('price') is not None: lines.append(f"💰 Price: ₹{data['price']:.2f}")
        lines.append(f"🚚 Delivery: {data.get('delivery_date', 'N/A')}")
        lines.append(f"📊 Status: {data.get('order_status', 'PROCESSING')}")
    elif "RFQ" in item_type or "INQUIRY" in item_type:
        lines.append(f"📋 Details: {data.get('specifications', product_info)[:200]}")
        if data.get('quantity'): lines.append(f"🔢 Qty: {data.get('quantity')} {data.get('unit', '')}")
        lines.append(f"📊 Status: {data.get('order_status', 'PENDING_QUOTE')}")
    lines.append(f"📧 Subject (Ref): {data.get('source_email_subject', 'N/A')[:70]}")
    return "\n".join(lines)


def send_notifications_from_db(db_conn):  # Pass db_conn
    if not db_conn: st.error("DB connection lost for notifications."); return
    notifications_sent = 0
    try:
        with db_conn.cursor() as cur:
            # Use actual column names from your 'orders' table
            cols = ["id", "order_type", "product_name", "category", "price", "quantity", "unit",
                    "specifications", "order_date", "delivery_date", "customer_name",
                    "customer_phone", "email", "address", "payment_method",
                    "payment_status", "order_status", "message_sent",
                    "source_email_subject", "source_email_sender", "source_email_uid",
                    "raw_text_content", "created_at"]
            cur.execute(
                f"SELECT {', '.join(cols)} FROM orders WHERE message_sent = FALSE ORDER BY created_at ASC LIMIT 5")  # Limit to 5 per run
            pending_items = cur.fetchall()
            if not pending_items: st.info("No new items in DB needing WhatsApp notification currently."); return

            seller_recipients = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
            if not seller_recipients: return  # get_seller_team_recipients will show a warning

            st.info(f"Found {len(pending_items)} item(s) to notify via WhatsApp.")
            for item_row in pending_items:
                item_id_for_notif = item_row[cols.index('id')]
                message = format_whatsapp_message_for_item(item_row, cols)
                st.write(f"Preparing WA for Item ID: {item_id_for_notif} ({item_row[cols.index('order_type')]})")
                if send_whatsapp_notification(message, seller_recipients):
                    cur.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (item_id_for_notif,))
                    db_conn.commit();
                    notifications_sent += 1
                    st.success(f"Notification for item DB-{item_id_for_notif} sent & marked.")
                    if 'st' in globals() and hasattr(st,
                                                     'session_state'): st.session_state.pending_msg_count = None  # Invalidate cache
                    time.sleep(random.uniform(10, 15))  # Longer delay after successful send
                else:
                    st.error(f"Failed WA for item DB-{item_id_for_notif}. Will retry later.")
            if notifications_sent > 0: st.success(f"Sent {notifications_sent} notifications.")
    except Exception as e:
        st.error(f"Error sending notifications from DB: {e}"); import traceback; traceback.print_exc()


# --- End WhatsApp Notifier ---


# --- Scheduler Tasks (Conceptually scheduler_tasks.py) ---
def scheduled_email_check_and_process():
    print(f"[{datetime.now()}] Scheduler: Starting automatic email check...")
    db_conn = connect_to_db_main()
    if not db_conn: print(f"[{datetime.now()}] Scheduler: DB connection failed. Skipping email check."); return
    try:
        orders_found, rfqs_found = process_incoming_emails(
            db_conn=db_conn, subjects_to_monitor=EMAIL_SUBJECTS_TO_MONITOR,
            only_recent_days=st.secrets.get("SCHEDULER_RECENT_DAYS", 3),  # Check recent 3 days for scheduler
            mark_as_read=True
        )
        print(f"[{datetime.now()}] Scheduler: Email check complete. New POs: {orders_found}, New RFQs: {rfqs_found}")
        if orders_found > 0 or rfqs_found > 0:
            print(f"[{datetime.now()}] Scheduler: New items processed, attempting to send notifications.")
            send_notifications_from_db(db_conn)
    except Exception as e:
        print(f"[{datetime.now()}] Scheduler: Error during scheduled check: {e}"); import \
            traceback; traceback.print_exc()
    finally:
        if db_conn: db_conn.close(); print(f"[{datetime.now()}] Scheduler: DB connection closed.")
    if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.last_check_time = datetime.now()


def run_background_scheduler():
    print("Background scheduler thread started. Waiting for jobs...")
    while True: schedule.run_pending(); time.sleep(1)


# --- End Scheduler Tasks ---

# --- Streamlit UI (main_app.py) ---
# Session State Initialization
default_ss_keys = {
    "whatsapp_sent_counter": 0, "whatsapp_errors": [],
    "last_check_time": datetime.now() - timedelta(hours=1),  # Set last check to an hour ago initially
    "auto_check_enabled": st.secrets.get("SCHEDULER_AUTO_ENABLE", False),
    "check_interval_minutes": st.secrets.get("SCHEDULER_INTERVAL_MINUTES", 60),
    "scheduler_started": False, "last_scheduled_interval": None,
    "db_init_success": False, "scheduler_thread": None,
    "pending_msg_count": None, "pending_msg_count_last_updated": None,
}
for key, default_val in default_ss_keys.items():
    if key not in st.session_state: st.session_state[key] = default_val

st.set_page_config(page_title="PO/RFQ Email Processor", layout="wide")
st.title("📧 PO & RFQ Email Processor & Notifier")

if not st.session_state.db_init_success:
    print("UI: Attempting initial DB connection for UI setup...")
    with st.spinner("Connecting to Database..."):
        db_conn_ui_init = connect_to_db_main()
        if db_conn_ui_init:
            st.session_state.db_init_success = True
            st.sidebar.success("Database Connected.")
            db_conn_ui_init.close()
        else:
            st.sidebar.error("DB Connection FAILED!")

with st.sidebar:
    st.header("⚙️ Controls & Status")
    if not st.session_state.db_init_success: st.warning("Database not connected. Functionality limited.")

    st.subheader("📧 Email Processing")
    st.info(f"Monitors {len(EMAIL_SUBJECTS_TO_MONITOR)} subject patterns.")
    if st.button("Manually Process Emails Now", key="manual_email_check_btn",
                 disabled=not st.session_state.db_init_success):
        with st.spinner("Manual Email Check... This may take time."):
            db_manual_conn = connect_to_db_main()
            if db_manual_conn:
                try:
                    o, r = process_incoming_emails(db_manual_conn, EMAIL_SUBJECTS_TO_MONITOR,
                                                   only_recent_days=st.secrets.get("MANUAL_CHECK_RECENT_DAYS", 7),
                                                   mark_as_read=True)
                    st.success(f"Manual check: {o} POs, {r} RFQs processed.")
                    if o > 0 or r > 0: send_notifications_from_db(db_manual_conn)
                finally:
                    db_manual_conn.close()
            else:
                st.error("Manual check failed: No DB connection.")
        st.rerun()

    st.subheader("⚙️ Automatic Scheduler")
    auto_check_ui = st.checkbox("Enable Automatic Email Processing", value=st.session_state.auto_check_enabled,
                                key="auto_check_cb")
    if auto_check_ui != st.session_state.auto_check_enabled: st.session_state.auto_check_enabled = auto_check_ui; st.rerun()
    interval_ui = st.slider("Scheduler Interval (min)", 5, 180, st.session_state.check_interval_minutes, 5,
                            key="interval_sl",
                            disabled=not st.session_state.auto_check_enabled)
    if interval_ui != st.session_state.check_interval_minutes: st.session_state.check_interval_minutes = interval_ui; st.rerun()
    st.caption(f"Last auto-check: {st.session_state.last_check_time.strftime('%I:%M %p, %d-%b')}")
    if st.session_state.auto_check_enabled and st.session_state.scheduler_started:
        next_run_time = schedule.next_run()
        if next_run_time: st.caption(f"Next auto-check: {next_run_time.strftime('%I:%M %p, %d-%b')}")

    st.subheader("📱 WhatsApp Notifications")


    def get_displayed_pending_count_sb():
        now = datetime.now()
        if (st.session_state.pending_msg_count is None or
                st.session_state.pending_msg_count_last_updated is None or
                (now - st.session_state.pending_msg_count_last_updated) > timedelta(
                    minutes=st.secrets.get("PENDING_COUNT_CACHE_MIN", 1))):
            print("Sidebar: Refreshing pending notification count from DB.")
            conn = connect_to_db_main()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM orders WHERE message_sent = FALSE")
                    st.session_state.pending_msg_count = cur.fetchone()[0]
                except Exception as e_cnt:
                    st.session_state.pending_msg_count = f"Err ({e_cnt})"
                finally:
                    conn.close()
            else:
                st.session_state.pending_msg_count = "DB Err"
            st.session_state.pending_msg_count_last_updated = now
        return st.session_state.pending_msg_count


    pending_val_sb = get_displayed_pending_count_sb()
    st.metric("Items Needing Notification", pending_val_sb if isinstance(pending_val_sb, int) else str(pending_val_sb))
    if st.button("Send Pending Notifications Now", key="send_pending_wa_btn",
                 disabled=not (isinstance(pending_val_sb, int) and pending_val_sb > 0)):
        with st.spinner(f"Sending {pending_val_sb} notifications..."):
            conn_notif = connect_to_db_main()
            if conn_notif:
                try:
                    send_notifications_from_db(conn_notif)
                finally:
                    conn_notif.close()
            else:
                st.error("Notification sending failed: No DB.")
        st.rerun()
    st.caption(f"Recipients: {SELLER_TEAM_RECIPIENTS_STR or 'None'}")
    st.caption(f"Total WA Sent (Session): {st.session_state.whatsapp_sent_counter}")
    if st.session_state.whatsapp_errors:
        with st.expander("WA Errors (Session)"): st.error("\n".join(st.session_state.whatsapp_errors))

# Main Tabs
tab_dash, tab_items, tab_manual, tab_qwa, tab_logs = st.tabs(
    ["📊 Dashboard", "📋 Processed Items", "✍️ Manual Entry", "📞 Quick WA", "📜 Logs"])

with tab_dash:  # Dashboard Tab
    st.header("📊 Overview Dashboard")
    if st.session_state.db_init_success:
        conn = connect_to_db_main()
        if conn:
            try:  # Your dashboard queries from previous version, adapted
                with conn.cursor() as cur:
                    st.subheader("📈 Key Metrics (Last 30 Days)")
                    c1, c2, c3, c4 = st.columns(4)
                    thirty_days_ago = datetime.now() - timedelta(days=30)
                    cur.execute("SELECT COUNT(*) FROM orders WHERE order_type LIKE 'PO%%' AND created_at >= %s",
                                (thirty_days_ago,));
                    c1.metric("POs (30d)", cur.fetchone()[0] or 0)
                    cur.execute("SELECT COUNT(*) FROM orders WHERE order_type LIKE 'RFQ%%' AND created_at >= %s",
                                (thirty_days_ago,));
                    c2.metric("RFQs (30d)", cur.fetchone()[0] or 0)
                    cur.execute(
                        "SELECT SUM(price * quantity) FROM orders WHERE order_type LIKE 'PO%%' AND payment_status = 'Paid' AND created_at >= %s",
                        (thirty_days_ago,));
                    rev = cur.fetchone()[0];
                    c3.metric("Paid PO Rev (30d)", f"₹{rev or 0:.2f}")
                    cur.execute("SELECT COUNT(*) FROM processed_emails WHERE processed_at >= %s", (thirty_days_ago,));
                    c4.metric("Emails Logged (30d)", cur.fetchone()[0] or 0)
                    st.divider()
                    st.subheader("📊 Item Type Distribution (All Time)")
                    cur.execute("SELECT order_type, COUNT(*) as count FROM orders GROUP BY order_type")
                    type_data = cur.fetchall()
                    if type_data:
                        df_type = pd.DataFrame(type_data, columns=['Item Type', 'Count']).set_index(
                            'Item Type'); st.bar_chart(df_type)
                    else:
                        st.write("No item type data.")
            except Exception as e_d:
                st.error(f"Dashboard error: {e_d}")
            finally:
                conn.close()
        else:
            st.warning("Dashboard: DB connection failed.")
    else:
        st.warning("Dashboard: DB not initialized.")

with tab_items:  # Processed Items Tab
    st.header("📋 Processed Orders, RFQs & Inquiries")
    if st.session_state.db_init_success:
        conn = connect_to_db_main()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""SELECT id, created_at, order_type, product_name, specifications, quantity, unit,
                                   customer_name, email, order_status, message_sent, source_email_uid
                                   FROM orders ORDER BY created_at DESC LIMIT 100""")  # Show last 100
                    items_rows = cur.fetchall()
                    if items_rows:
                        cols_display = ['ID', 'Logged At', 'Type', 'Product', 'Specs', 'Qty', 'Unit', 'Customer',
                                        'Contact', 'Status', 'Notified', 'Email UID']
                        df_disp_items = pd.DataFrame(items_rows, columns=cols_display)
                        st.dataframe(df_disp_items, use_container_width=True, hide_index=True)
                    else:
                        st.write("No processed items found.")
            except Exception as e_li:
                st.error(f"Error loading items: {e_li}")
            finally:
                conn.close()
        else:
            st.warning("Items List: DB connection failed.")
    else:
        st.warning("Items List: DB not initialized.")

with tab_manual:  # Manual Entry Tab
    st.header("✍️ Manual Data Entry for PO / RFQ")
    st.write("Use this form to manually log an order, RFQ, or inquiry.")
    # Build a form similar to previous examples, mapping fields to the `orders` table.
    # On submit, call `store_parsed_order_or_rfq(db_conn, form_data_dict)`
    # Example field:
    # manual_order_type = st.selectbox("Entry Type", ["PO_MANUAL", "RFQ_MANUAL", "INQUIRY_MANUAL"])
    # manual_product = st.text_input("Product Name/Description")
    # ... etc. ...
    st.info(
        "Manual entry form to be implemented here, similar to previous examples but adapted for the new 'orders' table structure (including 'order_type', 'specifications', 'unit', etc.).")

with tab_qwa:  # Quick WhatsApp Tab
    st.header("📞 Quick WhatsApp Message")
    # Re-use the Quick WhatsApp UI from previous full script
    st.info("Quick WhatsApp functionality to be implemented here, as per previous examples.")
    # Ensure it uses get_seller_team_recipients and send_whatsapp_notification

with tab_logs:  # Log Tab
    st.header("📜 Email Processing Logs (Recent)")
    if st.session_state.db_init_success:
        conn = connect_to_db_main()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""SELECT processed_at, email_uid, subject, sender, status, related_order_id
                                   FROM processed_emails ORDER BY processed_at DESC LIMIT 100""")
                    logs = cur.fetchall()
                    if logs:
                        df_logs = pd.DataFrame(logs, columns=['Timestamp', 'Email UID', 'Subject', 'Sender',
                                                              'Processing Status', 'Related Order ID'])
                        st.dataframe(df_logs, use_container_width=True, hide_index=True)
                    else:
                        st.write("No email processing logs found.")
            except Exception as e_log:
                st.error(f"Error loading logs: {e_log}")
            finally:
                conn.close()
        else:
            st.warning("Logs: DB connection failed.")
    else:
        st.warning("Logs: DB not initialized.")

# Background Scheduler Setup
if st.session_state.auto_check_enabled:
    current_interval = st.session_state.check_interval_minutes
    interval_has_changed = st.session_state.last_scheduled_interval != current_interval
    if not st.session_state.scheduler_started or interval_has_changed:
        schedule.clear()
        job = schedule.every(current_interval).minutes.do(scheduled_email_check_and_process)
        st.session_state.last_scheduled_interval = current_interval
        print(f"Scheduler: Job (re)defined: Every {current_interval} mins. Job: {job}")
        if not st.session_state.scheduler_started:
            if st.session_state.scheduler_thread is None or not st.session_state.scheduler_thread.is_alive():
                thread = threading.Thread(target=run_background_scheduler, daemon=True, name="BgScheduler")
                thread.start();
                st.session_state.scheduler_thread = thread;
                st.session_state.scheduler_started = True
                print("Scheduler: Bg thread started.");
                st.toast(f"Auto-processing ON: every {current_interval}m.", icon="⏰")
            else:
                print("Scheduler: Bg thread already running. Job updated."); st.toast(
                    f"Auto-processing updated: every {current_interval}m.", icon="🔄")
        else:
            print(f"Scheduler: Job interval updated."); st.toast(f"Auto-processing interval: {current_interval}m.",
                                                                 icon="🔄")
elif not st.session_state.auto_check_enabled and st.session_state.scheduler_started:
    schedule.clear();
    st.session_state.scheduler_started = False;
    st.session_state.last_scheduled_interval = None
    print("Scheduler: Auto-processing disabled. Jobs cleared.");
    st.toast("Auto-processing OFF.", icon="🛑")