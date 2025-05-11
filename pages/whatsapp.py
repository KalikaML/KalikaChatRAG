import streamlit as st
import pywhatkit
import pyautogui
from datetime import timedelta, datetime
import time
import imaplib
import email
from email.header import decode_header
import PyPDF2
import re
import io
import threading
import phonenumbers
import schedule
import random
import webbrowser
import psycopg2
import os
from streamlit import secrets  # Ensure this is streamlit.secrets, not os.secrets
import pandas as pd
import matplotlib.pyplot as plt  # Import for dashboard pie chart

# Secrets for email
EMAIL = st.secrets["gmail_uname"]
PASSWORD = st.secrets["gmail_pwd"]
IMAP_SERVER = "imap.gmail.com"
SELLER_TEAM_RECIPIENTS_STR = st.secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS")

# Database credentials
DB_HOST = st.secrets.get("DB_HOST", "localhost")
DB_NAME = st.secrets.get("DB_NAME", "po_orders")
DB_USER = st.secrets.get("DB_USER", "po_user")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "postdb123")
DB_PORT = st.secrets.get("DB_PORT", 5432)


def create_orders_table(conn):
    """Creates the 'orders' table if it doesn't exist."""
    try:
        with conn.cursor() as cursor:
            # For large tables, ensure appropriate indexes on frequently queried columns
            # e.g., order_date, message_sent, customer_name, product_name, order_status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    product_name TEXT,
                    category TEXT,
                    price REAL,
                    quantity INTEGER,
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
                    source_email_sender TEXT
                );
                -- Example: CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date DESC);
                -- Example: CREATE INDEX IF NOT EXISTS idx_orders_message_sent ON orders(message_sent);
            """)
        conn.commit()
        print("Orders table checked/created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating orders table: {e}")
        st.error(f"Error creating orders table: {e}")
        if conn: conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred during table creation: {e}")
        st.error(f"An unexpected error occurred during table creation: {e}")
        if conn: conn.rollback()


def connect_to_db():
    """
    Connects to the PostgreSQL database.
    Table creation is called here, ensuring it's checked on new connections.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("Database connection successful.")
        create_orders_table(conn)  # Ensures table exists
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        # st.error(f"Database connection error: {e}") # Avoid too many errors on backend if DB is temporarily down
        return None
    except Exception as e:
        print(f"An unexpected error occurred during database connection: {e}")
        # st.error(f"An unexpected error occurred during database connection: {e}")
        return None


def store_order(conn, order_details):
    if conn is None:
        st.error("Database connection is not available. Cannot store order.")
        print("Store order error: Database connection is None.")
        return False
    try:
        with conn.cursor() as cursor:
            columns = [
                "product_name", "category", "price", "quantity", "order_date", "delivery_date",
                "customer_name", "customer_phone", "email", "address", "payment_method",
                "payment_status", "order_status", "message_sent",
                "source_email_subject", "source_email_sender"
            ]
            values = [
                order_details.get("Product Name"), order_details.get("Category", ""),
                order_details.get("Price"), order_details.get("Quantity"),
                order_details.get("Order Date"), order_details.get("Delivery Date"),
                order_details.get("Customer Name"), order_details.get("Raw Customer Phone"),
                order_details.get("Email", ""), order_details.get("Address"),
                order_details.get("Payment Method"), order_details.get("Payment Status"),
                order_details.get("Order Status"), False,  # message_sent
                order_details.get("Source Email Subject", ""),
                order_details.get("Source Email Sender", "")
            ]
            if not all([order_details.get("Product Name"), order_details.get("Price") is not None,
                        order_details.get("Quantity") is not None,
                        order_details.get("Order Date"), order_details.get("Delivery Date"),
                        order_details.get("Customer Name"),
                        order_details.get("Address")]):
                st.error("Required fields missing in order_details. Cannot store.")
                print(f"Validation failed for order_details: {order_details}")
                return False
            sql = f"INSERT INTO orders ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
            cursor.execute(sql, tuple(values))
        conn.commit()
        st.session_state.pending_msg_count = None  # Invalidate cache
        print(f"Order for {order_details.get('Product Name')} stored successfully.")
        return True
    except psycopg2.Error as e:
        if conn: conn.rollback()
        print(f"Database error storing order: {e}")
        st.error(f"Database error storing order: {e}")
        return False
    except Exception as e:
        if conn: conn.rollback()
        print(f"Store order error: {e}")
        st.error(f"Failed to store order: {e}")
        return False


# --- Session State Initialization ---
# Initialize only if keys don't exist to preserve state across reruns
default_session_state = {
    "whatsapp_sent_counter": 0, "whatsapp_errors": [], "manual_order_sent": False,
    "last_check_time": datetime.now(), "auto_check_enabled": False,  # Auto check OFF by default
    "check_interval_minutes": 30, "sending_in_progress": False,
    "email_search_query": "Purchase Order", "scheduler_started": False,
    "last_scheduled_query": None, "last_scheduled_interval": None,
    "db_init_success": False, "scheduler_thread": None,
    "pending_msg_count": None, "pending_msg_count_last_updated": None,  # For caching
}
for key, default in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_bytes):
    # ... (original code)
    try:
        pdf_file_like_object = io.BytesIO(pdf_file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file_like_object)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def format_phone_number(phone_str):
    # ... (original code)
    if not phone_str or not isinstance(phone_str, str): return None
    try:
        p = phonenumbers.parse(phone_str, "IN" if not phone_str.startswith('+') else None)
        return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(
            p) else None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None


def parse_order_details(text):
    # ... (original detailed parsing logic - ensure it's robust)
    patterns = {
        "Order ID": r"Order ID:?\s*([A-Z0-9-]+)",
        "Product Name": r"Product(?: Name)?:?\s*(.+?)(?:\nCategory:|\nPrice:|\nQuantity:|$)",
        "Category": r"Category:?\s*(.+?)(?:\nPrice:|\nQuantity:|\nOrder Date:|$)",
        "Price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "Quantity": r"Quantity:?\s*(\d+)",
        "Order Date": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
        "Delivery Date": r"(?:Expected )?Delivery(?: Date)?:?\s*(\d{4}-\d{2}-\d{2})",
        "Customer Name": r"Customer(?: Name)?:?\s*(.+?)(?:\nPhone:|\nEmail:|\nAddress:|$)",
        "Phone": r"Phone:?\s*(\+?\d[\d\s-]{8,15}\d)",
        "Email": r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        "Address": r"Address:?\s*(.+?)",  # Will be refined
        "Payment Method": r"Payment(?: Method)?:?\s*(COD|Cash on Delivery|Credit Card|UPI|Bank Transfer|Other)",
        "Payment Status": r"Payment Status:?\s*(Paid|Unpaid|Pending)",
        "Order Status": r"(?:Order )?Status:?\s*(Pending|Processing|Confirmed|Shipped|Delivered|Cancelled)",
    }
    order_details = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if key == "Price":
                order_details[key] = float(value.replace(",", "")) if value else None
            elif key == "Quantity":
                order_details[key] = int(value) if value else None
            elif key == "Order Date" and value:
                try:
                    order_details[key] = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        order_details[key] = datetime.strptime(value, "%Y-%m-%d %H:%M")
                    except ValueError:
                        order_details[key] = "Not found"
            elif key == "Delivery Date" and value:
                try:
                    order_details[key] = datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    order_details[key] = "Not found"
            else:
                order_details[key] = value
        else:
            order_details[key] = "Not found" if key not in ["Category", "Email", "Order ID"] else ""
    address_pattern_refined = r"Address:?\s*(.*?)(?:Payment Method:|Payment Status:|Order Status:|Notes:|Tracking ID:|Order ID:|Customer Name:|Phone:|Email:|---|$)"
    match_addr = re.search(address_pattern_refined, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match_addr:
        address_candidate = match_addr.group(1).strip().replace('\n', ' ')
        stop_keywords = ["Payment Method:", "Payment Status:", "Order Status:", "Notes:", "Tracking ID:"]
        for keyword in stop_keywords:
            if keyword.lower() in address_candidate.lower():
                address_candidate = address_candidate.lower().split(keyword.lower())[0].strip()
        order_details["Address"] = address_candidate
    elif order_details.get("Address") == "Not found" or len(order_details.get("Address", "")) < 10:
        order_details["Address"] = "Not found"
    order_details["Raw Customer Phone"] = order_details.get("Phone", "Not found")
    return order_details


# --- WhatsApp Sending Functions ---
def send_whatsapp_message_pywhatkit(message, recipient_numbers_list):
    # ... (original pywhatkit sending logic - can be slow and open browser tabs)
    status_container = st.empty()
    if not isinstance(recipient_numbers_list, (list, set)):  # ... (rest of your validation)
        status_container.error("Invalid recipient numbers format for pywhatkit.");
        return False
    if not recipient_numbers_list: status_container.error("No recipients for pywhatkit."); return False
    sent_successfully = True
    for recipient in recipient_numbers_list:
        try:
            formatted_recipient = format_phone_number(recipient)
            if not formatted_recipient: st.warning(
                f"Skipping invalid phone for pywhatkit: {recipient}"); sent_successfully = False; continue
            current_time = datetime.now();
            send_hour = current_time.hour;
            send_minute = (current_time.minute + 2) % 60  # Send in 2 mins
            if send_minute < current_time.minute: send_hour = (send_hour + 1) % 24
            status_container.info(
                f"PyWhatKit: Queuing to {formatted_recipient} at {send_hour:02d}:{send_minute:02d}...")
            pywhatkit.sendwhatmsg(formatted_recipient, message, send_hour, send_minute, wait_time=20, tab_close=True,
                                  close_time=5)
            status_container.success(f"PyWhatKit: Message queued for {formatted_recipient}!")
            st.session_state.whatsapp_sent_counter += 1;
            time.sleep(5)
        except Exception as e:
            status_container.error(f"PyWhatKit Error for {recipient}: {e}")
            st.session_state.whatsapp_errors.append(f"Error for {recipient}: {str(e)}");
            sent_successfully = False
    return sent_successfully


def send_whatsapp_message_web(message, recipient_numbers_list):
    # ... (original web sending logic - also opens browser, requires manual interaction if QR scan needed)
    status_container = st.empty()
    if not isinstance(recipient_numbers_list, (list, set)):  # ... (rest of your validation)
        status_container.error("Invalid recipient format for web WhatsApp.");
        return False
    if not recipient_numbers_list: status_container.error("No recipients for web WhatsApp."); return False

    webbrowser.open("https://web.whatsapp.com")
    status_container.info("Opened WhatsApp Web. Scan QR if needed. Waiting 25s...")
    time.sleep(25)  # Increased wait time
    sent_successfully = True
    for recipient in recipient_numbers_list:
        try:
            formatted_recipient = format_phone_number(recipient)
            if not formatted_recipient: st.warning(
                f"Skipping invalid phone for web: {recipient}"); sent_successfully = False; continue
            import urllib.parse;
            encoded_message = urllib.parse.quote(message)
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_recipient}&text={encoded_message}"
            webbrowser.open(whatsapp_url)
            status_container.info(f"Web: Opened chat for {formatted_recipient}. Waiting 15s...")
            time.sleep(15);
            pyautogui.press("enter")
            status_container.success(f"Web: 'Enter' pressed for {formatted_recipient}.")
            st.session_state.whatsapp_sent_counter += 1;
            time.sleep(random.uniform(4, 7))
        except Exception as e:
            status_container.error(f"Web WhatsApp Error for {formatted_recipient}: {e}")
            st.session_state.whatsapp_errors.append(f"Web Error for {formatted_recipient}: {str(e)}");
            sent_successfully = False
    return sent_successfully


# Choose default: Web method is often more reliable if user can manage browser.
send_whatsapp_notification = send_whatsapp_message_web


def fetch_email_pdfs(subject_query, only_recent_days=None, mark_as_read_after_extraction=True):
    """
    Fetches PDFs from UNSEEN emails matching the subject query.

    Args:
        subject_query (str): The subject to search for.
        only_recent_days (int, optional): If provided, only fetch emails from the last N days. Defaults to None.
        mark_as_read_after_extraction (bool): If True, marks emails as SEEN after successfully extracting PDF(s).
                                              This helps avoid reprocessing when searching for UNSEEN emails next time.
                                              Defaults to True.

    Returns:
        list: A list of dictionaries, each containing PDF data and metadata.
    """
    pdf_files_with_info = []
    mail = None  # Initialize mail object for the finally block

    # For detailed performance analysis, uncomment and use these timing lines
    # overall_start_time = time.time()
    # print(f"[{datetime.now()}] Starting email fetch. Subject: '{subject_query}', Recent: {only_recent_days} days, Mark read: {mark_as_read_after_extraction}")

    try:
        # mail_conn_start_time = time.time()
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        # print(f"IMAP Connection and Login: {time.time() - mail_conn_start_time:.2f}s")

        # Build search criteria
        search_parts = ['(UNSEEN']  # Start with UNSEEN criteria
        search_parts.append(f'SUBJECT "{subject_query}"')

        if only_recent_days and isinstance(only_recent_days, int) and only_recent_days > 0:
            date_since = (datetime.now() - timedelta(days=only_recent_days)).strftime("%d-%b-%Y")
            search_parts.append(f'SINCE "{date_since}"')
        search_parts.append(')')
        search_criteria = ' '.join(search_parts)

        # print(f"Searching with criteria: {search_criteria}")
        # search_start_time = time.time()
        status, messages = mail.search(None, search_criteria)
        # print(f"IMAP Search: {time.time() - search_start_time:.2f}s, Status: {status}")

        if status != "OK":
            # st.error(f"IMAP search failed with status: {status}") # Use st.error if in Streamlit app
            print(f"IMAP search failed with status: {status}")
            return [] # mail.logout() will be called in finally

        if not messages[0]:  # No email IDs found
            print(f"No UNSEEN emails found matching criteria: {search_criteria}")
            return [] # mail.logout() will be called in finally

        mail_ids = messages[0].split()
        # print(f"Found {len(mail_ids)} email(s) to process. Processing newest first (if server sorts that way).")

        # Process emails (consider processing in reverse if mail_ids are sorted oldest to newest by default)
        for mail_id in reversed(mail_ids):
            # email_process_start_time = time.time()
            # print(f"Fetching email ID: {mail_id.decode()}")
            fetch_status, msg_data = mail.fetch(mail_id, '(RFC822)')
            if fetch_status != "OK":
                print(f"Failed to fetch email ID {mail_id.decode()}. Status: {fetch_status}")
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Robust header decoding function
            def decode_mail_header(header_value):
                if header_value is None:
                    return ""
                decoded_parts = decode_header(header_value)
                header_str = ""
                for part, charset in decoded_parts:
                    if isinstance(part, bytes):
                        header_str += part.decode(charset or "utf-8", errors="ignore")
                    else:
                        header_str += part
                return header_str

            subject = decode_mail_header(msg["Subject"])
            sender = decode_mail_header(msg.get("From"))

            pdf_extracted_this_email = False
            for part in msg.walk():
                if part.get_content_type() == "application/pdf" and part.get('Content-Disposition'):
                    filename = decode_mail_header(part.get_filename()) or f'untitled_{mail_id.decode()}.pdf'
                    pdf_data = part.get_payload(decode=True)

                    if pdf_data:
                        pdf_files_with_info.append({
                            "data": pdf_data,
                            "filename": filename,
                            "sender": sender,
                            "subject": subject,
                            "email_id": mail_id.decode() # Useful for logging or later reference
                        })
                        pdf_extracted_this_email = True
                        # print(f"  Extracted PDF: {filename} from email ID {mail_id.decode()}")

            if pdf_extracted_this_email and mark_as_read_after_extraction:
                try:
                    # print(f"  Attempting to mark email ID {mail_id.decode()} as SEEN.")
                    store_status, _ = mail.store(mail_id, '+FLAGS', '\\Seen')
                    if store_status == "OK":
                        pass
                        # print(f"  Successfully marked email ID {mail_id.decode()} as SEEN.")
                    else:
                        print(f"  Failed to mark email ID {mail_id.decode()} as SEEN. Status: {store_status}")
                except Exception as e_store:
                    print(f"  Error marking email ID {mail_id.decode()} as SEEN: {e_store}")
            # print(f"  Email ID {mail_id.decode()} processing time: {time.time() - email_process_start_time:.2f}s")
        return pdf_files_with_info

    except imaplib.IMAP4.abort as e_abort: # Specific IMAP abort error
        # This can happen due to server issues or prolonged inactivity.
        error_msg = f"IMAP connection aborted: {e_abort}. Try again later."
        print(error_msg)
        if 'st' in globals(): st.error(error_msg) # Check if st is available
        mail = None # Connection is already closed/broken
        return []
    except Exception as e:
        error_msg = f"An error occurred while fetching emails: {str(e)}"
        print(error_msg)
        if 'st' in globals(): st.error(error_msg) # Check if st is available
        return []
    finally:
        if mail:
            try:
                mail.close()  # Close the selected mailbox
                mail.logout()
                # print("Logged out from IMAP server.")
            except Exception as e_logout:
                print(f"Error during IMAP logout/close: {e_logout}")
        # print(f"Total email fetching function time: {time.time() - overall_start_time:.2f}s")


# --- Utility and Processing Functions ---
def get_seller_team_recipients(recipients_str):  # ... (original logic)
    recipients = set()
    if recipients_str and isinstance(recipients_str, str):
        for phone in recipients_str.split(","):
            fmt_seller = format_phone_number(phone.strip())
            if fmt_seller:
                recipients.add(fmt_seller)
            else:
                st.warning(f"Invalid seller phone in config: {phone.strip()}")
    return list(recipients)


def get_pending_message_count_from_db():  # Renamed for clarity
    """Actually hits the DB to get the count."""
    conn = connect_to_db()
    if not conn: return 0  # Return 0 if DB connection fails
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM orders WHERE message_sent = FALSE")
            return cursor.fetchone()[0]
    except psycopg2.Error as e:
        st.error(f"DB error getting message count: {e}")
        return 0  # Return 0 on error
    finally:
        if conn: conn.close()


def send_whatsapp_from_db():  # ... (original logic)
    status_container = st.empty()
    conn = connect_to_db()
    if not conn: status_container.error("DB connection failed for sending pending messages."); return
    orders_sent_successfully = 0
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, product_name, category, price, quantity, order_date, delivery_date, customer_name, customer_phone, email, address, payment_method, payment_status, order_status FROM orders WHERE message_sent = FALSE ORDER BY order_date ASC")
            pending_orders = cursor.fetchall()
            if not pending_orders: status_container.info("No pending WhatsApp messages from DB."); return
            seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
            if not seller_numbers: status_container.error("No valid seller WhatsApp numbers configured."); return
            status_container.info(f"Found {len(pending_orders)} pending order(s) for WhatsApp.")
            for order_data in pending_orders:
                # ... (message formatting as before)
                order_id, product_name, category, price, quantity, order_date, delivery_date, customer_name, customer_phone, cust_email, address, payment_method, payment_status, order_status = order_data
                message_lines = [f"📦 *New Order Notification (from DB)* 📦", f"Order ID: DB-{order_id}",
                                 f"🛍️ Product: {product_name or 'N/A'}",
                                 f"💰 Price: ₹{price:.2f}" if price is not None else "Price: N/A",
                                 f"🔢 Quantity: {quantity or 'N/A'}",
                                 f"🗓️ Order Date: {order_date.strftime('%Y-%m-%d %H:%M') if order_date else 'N/A'}",
                                 f"👤 Customer: {customer_name or 'N/A'}",
                                 f"🏠 Address: {address or 'N/A'}"]  # Shortened example
                formatted_message_text = "\n".join(message_lines)

                if send_whatsapp_notification(formatted_message_text, seller_numbers):
                    cursor.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (order_id,))
                    conn.commit();
                    orders_sent_successfully += 1
                    status_container.success(f"Notification for order DB-{order_id} sent & marked.")
                    st.session_state.pending_msg_count = None  # Invalidate cache
                    time.sleep(random.uniform(5, 10))
                else:
                    status_container.error(f"Failed WhatsApp for order DB-{order_id}. Will retry.")
            if orders_sent_successfully > 0: st.success(f"Sent notifications for {orders_sent_successfully} order(s).")
    except psycopg2.Error as e:
        status_container.error(f"DB error processing pending messages: {e}");
    except Exception as e:
        status_container.error(f"Unexpected error sending from DB: {e}")
    finally:
        if conn: conn.close()


def process_and_store_email_orders(subject_query):  # Called by button or scheduler
    # This function can be time-consuming due to email fetching and PDF parsing.
    # It should NOT be called directly on script load, only by user action or scheduler.
    status_container = st.empty()
    status_container.info(f"Checking emails for subject: '{subject_query}'...")
    pdf_files_info = fetch_email_pdfs(subject_query)  # This is the slow part
    if not pdf_files_info: status_container.info("No new PO emails with parsable PDFs found."); return
    processed_count, failed_store_count = 0, 0
    conn = connect_to_db()
    if not conn: status_container.error("DB connection failed. Cannot process email orders."); return
    with st.spinner(f"Processing {len(pdf_files_info)} email(s)..."):
        for pdf_info in pdf_files_info:
            try:
                st.write(f"Processing PDF: {pdf_info['filename']} from {pdf_info['sender']}")
                text = extract_text_from_pdf(pdf_info["data"])
                if text:
                    order_details = parse_order_details(text)
                    order_details["Source Email Subject"] = pdf_info["subject"]
                    order_details["Source Email Sender"] = pdf_info["sender"]
                    if order_details.get("Product Name") == "Not found" or order_details.get("Price") is None:
                        st.warning(f"Essential details not parsed from {pdf_info['filename']}. Skipping.");
                        failed_store_count += 1;
                        continue
                    if store_order(conn, order_details):
                        processed_count += 1; st.success(f"Stored order from '{pdf_info['filename']}'")
                    else:
                        st.error(f"Failed to store order from '{pdf_info['filename']}'."); failed_store_count += 1
                else:
                    st.warning(f"No text extracted from PDF: {pdf_info['filename']}"); failed_store_count += 1
            except Exception as e:
                st.error(f"Error processing PDF '{pdf_info['filename']}': {e}"); failed_store_count += 1
    if conn: conn.close()
    if processed_count > 0: status_container.success(f"Successfully stored {processed_count} order(s) from emails.")
    if failed_store_count > 0: status_container.warning(f"Failed to process/store {failed_store_count} PDF(s).")
    st.session_state.pending_msg_count = None  # Invalidate cache after processing


def check_and_process_emails_automatically(subject_query_for_scheduler):  # For scheduler thread
    print(f"[{datetime.now()}] Auto email check: '{subject_query_for_scheduler}'")
    # st.toast is tricky from non-main threads. Use print for thread logging.
    process_and_store_email_orders(subject_query_for_scheduler)
    st.session_state.last_check_time = datetime.now()
    print(f"[{datetime.now()}] Auto: Attempting to send pending WhatsApps from DB.")
    send_whatsapp_from_db()
    # Avoid st.rerun() in background threads; it can lead to unexpected behavior.
    # UI updates should happen naturally on next user interaction or via polling if needed.


def run_scheduled_tasks():
    while True:
        schedule.run_pending()
        time.sleep(1)  # Check schedule every second


# --- Streamlit UI ---
st.set_page_config(page_title="PO Order Manager", layout="wide")
st.title("🛒 PO Order Management Dashboard")

# --- Initial DB Connection Check (runs once per session ideally) ---
if not st.session_state.db_init_success:
    # This block runs on first load / if db_init_success is False.
    # connect_to_db() can be slow if DB is remote or unresponsive.
    print("Attempting initial DB connection and table check...")
    initial_conn = connect_to_db()
    if initial_conn:
        initial_conn.close()
        st.sidebar.success("Database connected & 'orders' table verified.")
        st.session_state.db_init_success = True
    else:
        st.sidebar.error("Initial Database connection FAILED! App may not function correctly.")
        # No st.stop() here, allow app to load but show error.

# --- Sidebar ---
with st.sidebar:
    st.header("📊 Status & Config")
    st.metric("WhatsApp Sent (Session)", st.session_state.whatsapp_sent_counter)
    if st.session_state.whatsapp_errors:
        with st.expander("WhatsApp Sending Errors"):
            for err in st.session_state.whatsapp_errors: st.error(err)

    st.subheader("⚙️ Auto-Email Check")
    auto_check_enabled_ui = st.checkbox("Enable Auto-Email Check & Notify", value=st.session_state.auto_check_enabled,
                                        key="auto_check_toggle")
    if auto_check_enabled_ui != st.session_state.auto_check_enabled:
        st.session_state.auto_check_enabled = auto_check_enabled_ui
        st.rerun()  # Rerun to update scheduler logic

    interval_minutes_ui = st.slider("Check Interval (minutes)", 5, 120, st.session_state.check_interval_minutes, 5,
                                    key="interval_slider")
    if interval_minutes_ui != st.session_state.check_interval_minutes:
        st.session_state.check_interval_minutes = interval_minutes_ui
        st.rerun()  # Rerun to update scheduler

    st.text(f"Last Auto-Check: {st.session_state.last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.auto_check_enabled and st.session_state.scheduler_started:
        next_run = schedule.next_run()
        if next_run:
            st.text(f"Next Auto-Check: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.text("Next Auto-Check: Pending schedule or just ran.")

    st.subheader("📧 Manual Email PO Processing")
    # This button directly triggers the potentially slow email processing.
    if st.button("Manually Check Emails & Store Orders Now", key="manual_email_check_button_sidebar"):
        with st.spinner("Manually checking emails and processing PDFs... This may take time."):
            process_and_store_email_orders(st.session_state.email_search_query)
            send_whatsapp_from_db()  # Also try to send any new or pending messages
        st.rerun()  # Update UI, including pending counts

    st.subheader("📱 Database WhatsApp Notifications")


    # --- Cached Pending Message Count Display ---
    def get_displayed_pending_message_count():
        now = datetime.now()
        # Re-fetch from DB if cache is old (e.g., > 1 min) or invalidated
        if (st.session_state.pending_msg_count is None or
                st.session_state.pending_msg_count_last_updated is None or
                (now - st.session_state.pending_msg_count_last_updated) > timedelta(minutes=1)):
            print("Fetching pending message count from DB for display...")
            st.session_state.pending_msg_count = get_pending_message_count_from_db()
            st.session_state.pending_msg_count_last_updated = now
        return st.session_state.pending_msg_count


    pending_count_display = get_displayed_pending_message_count()
    st.metric("Pending Orders in DB (WhatsApp)",
              pending_count_display if pending_count_display is not None else "Loading...")
    # --- End Cached Pending Message Count ---

    if st.button("Send All Pending Order Notifications from DB", key="send_pending_db_whatsapp"):
        if pending_count_display is not None and pending_count_display > 0:
            with st.spinner(f"Sending {pending_count_display} pending notifications..."):
                send_whatsapp_from_db()
            st.rerun()
        else:
            st.info("No pending notifications in DB to send (or count is loading).")

# --- Main Content Tabs ---
# Streamlit generally loads tab content lazily (when tab is clicked).
# If the default first tab has heavy DB queries, initial load will reflect that.
tab_email_po, tab_manual_entry, tab_quick_whatsapp, tab_dashboard = st.tabs([
    "📬 Email PO Processing", "📝 Manual PO Entry", "📞 Quick WhatsApp", "📊 Dashboard & Analysis"
])

with tab_email_po:
    st.subheader("📧 Email Order Search & Recent Orders")
    current_email_query_main = st.session_state.email_search_query
    subject_query_input_main = st.text_input(
        "Email Subject to Search for POs:",
        value=current_email_query_main,
        key="main_email_search_query_input"
    )
    if subject_query_input_main != current_email_query_main:
        st.session_state.email_search_query = subject_query_input_main
        st.rerun()  # Updates query for manual check and scheduler on next cycle
    st.write(f"Current search subject for manual/auto checks: **'{st.session_state.email_search_query}'**")
    st.info("Use 'Manually Check Emails' in sidebar for immediate processing. Auto-checking configured in sidebar.")

    st.subheader("📋 Recent Orders (Last 10 from Database)")
    # This DB query runs when this tab is active.
    conn_tab1 = connect_to_db()
    if conn_tab1:
        try:
            with conn_tab1.cursor() as cur:
                # Query optimized to fetch only necessary columns.
                cur.execute(
                    "SELECT order_date, product_name, customer_name, price, quantity, order_status, message_sent FROM orders ORDER BY order_date DESC LIMIT 10")
                recent_orders = cur.fetchall()
                if recent_orders:
                    df_recent = pd.DataFrame(recent_orders,
                                             columns=["Order Date", "Product", "Customer", "Price", "Qty", "Status",
                                                      "Notified"])
                    st.dataframe(df_recent, use_container_width=True, hide_index=True)
                else:
                    st.write("No orders found in the database yet.")
        except Exception as e:
            st.error(f"Could not load recent orders: {e}")
        finally:
            conn_tab1.close()
    else:
        st.warning("Database not connected. Cannot display recent orders.")

with tab_manual_entry:  # ... (original manual entry form logic)
    st.subheader("✍️ Manual Order Entry & Instant Notification")
    with st.expander("➕ Add New Purchase Order Manually", expanded=True):
        with st.form("manual_order_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:  # Product Details
                m_product_name = st.text_input("Product Name *", key="m_prod")
                m_category = st.text_input("Category", key="m_cat")
                m_price = st.number_input("Price (₹) *", min_value=0.01, step=0.01, format="%.2f", key="m_price")
                m_quantity = st.number_input("Quantity *", min_value=1, step=1, key="m_qty")
            with col2:  # Customer & Order Details
                m_customer_name = st.text_input("Customer Name *", key="m_cname")
                m_customer_phone_raw = st.text_input("Customer Phone (for ref)", key="m_cphone")
                m_email = st.text_input("Customer Email", key="m_cemail")
                m_address = st.text_area("Delivery Address *", key="m_addr")
            st.divider()
            col3, col4 = st.columns(2)  # Dates & Status
            with col3:
                m_order_date = st.date_input("Order Date *", value=datetime.now().date(), key="m_odate")
                m_order_time = st.time_input("Order Time *", value=datetime.now().time(), key="m_otime")
                m_delivery_date = st.date_input("Expected Delivery Date *",
                                                value=(datetime.now() + timedelta(days=7)).date(), key="m_ddate")
            with col4:
                m_payment_method = st.selectbox("Payment Method",
                                                ["COD", "Credit Card", "UPI", "Bank Transfer", "Other"],
                                                key="m_pmethod")
                m_payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Pending"], key="m_pstatus")
                m_order_status = st.selectbox("Order Status",
                                              ["Pending", "Processing", "Confirmed", "Shipped", "Delivered",
                                               "Cancelled"], key="m_ostatus")
            submit_manual_order = st.form_submit_button("Submit Order & Notify Seller Team")
            if submit_manual_order:
                if not all([m_product_name, m_price, m_quantity, m_order_date, m_order_time, m_delivery_date,
                            m_customer_name, m_address]) or m_price <= 0 or m_quantity <= 0:
                    st.error("Please fill ALL required (*) fields with valid values.")
                else:
                    order_datetime = datetime.combine(m_order_date, m_order_time)
                    manual_order_details = {  # ... (details as before)
                        "Product Name": m_product_name, "Category": m_category, "Price": float(m_price),
                        "Quantity": int(m_quantity), "Order Date": order_datetime,
                        "Delivery Date": m_delivery_date, "Customer Name": m_customer_name,
                        "Raw Customer Phone": m_customer_phone_raw, "Email": m_email, "Address": m_address,
                        "Payment Method": m_payment_method, "Payment Status": m_payment_status,
                        "Order Status": m_order_status,
                    }
                    conn_manual = connect_to_db()
                    if conn_manual:
                        if store_order(conn_manual, manual_order_details):
                            st.success("Manual order stored in database!")
                            # ... (WhatsApp notification logic as before) ...
                            message_lines = [f"📦 *New Manual PO Order!* 📦", f"🛍️ Product: {m_product_name}",
                                             f"💰 Price: ₹{m_price:.2f}",
                                             f"👤 Customer: {m_customer_name}"]  # Shortened
                            formatted_message = "\n".join(message_lines)
                            seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
                            if seller_numbers:
                                st.info(f"Sending WhatsApp to: {', '.join(seller_numbers)}")
                                if send_whatsapp_notification(formatted_message, seller_numbers):
                                    st.success("WhatsApp sent to seller team!")
                                    try:  # Mark as sent
                                        with conn_manual.cursor() as cur:
                                            cur.execute(
                                                "UPDATE orders SET message_sent = TRUE WHERE product_name = %s AND customer_name = %s AND order_date = %s AND message_sent = FALSE ORDER BY id DESC LIMIT 1",
                                                (m_product_name, m_customer_name, order_datetime))
                                            conn_manual.commit()
                                    except Exception as e_upd:
                                        st.warning(f"Could not mark manual order as sent: {e_upd}")
                                else:
                                    st.error("Failed to send WhatsApp (stored with message_sent=FALSE).")
                            else:
                                st.warning("No seller numbers. Notification not sent (order stored).")
                        else:
                            st.error("Failed to store manual order in database.")
                        conn_manual.close()
                    else:
                        st.error("Database connection failed. Cannot store manual order.")
                    st.rerun()

with tab_quick_whatsapp:  # ... (original quick WhatsApp logic)
    st.subheader("📱 Quick WhatsApp to Contacts")
    contact_dict = {"Narayan": "+919067847003", "Rani Bhise": "+917070242402", "Abhishek": "+919284625240",
                    "Damini": "+917499353409", "Sandeep": "+919850030215", "Chandrakant": "+919665934999",
                    "Vikas Kumbharkar": "+919284238738"}
    custom_message = st.text_area("Custom Message", "Hello! Quick update: ", height=100, key="quick_msg_text")
    st.write("Select contacts:");
    selected_contacts_numbers = [];
    cols = st.columns(3)
    for i, (name, phone) in enumerate(list(contact_dict.items())):
        if cols[i % 3].checkbox(f"{name} ({phone})", key=f"cb_contact_{name.replace(' ', '_')}"):
            fmt_num = format_phone_number(phone)
            if fmt_num:
                selected_contacts_numbers.append(fmt_num)
            else:
                st.warning(f"Invalid phone for {name}: {phone}. Skipping.")
    if st.button("Send WhatsApp to Selected Contacts", key="quick_send_button"):
        if not custom_message.strip():
            st.warning("Please enter a message.")
        elif selected_contacts_numbers:
            st.info(f"Sending '{custom_message}' to {len(selected_contacts_numbers)} contact(s)...")
            if send_whatsapp_notification(custom_message, selected_contacts_numbers):
                st.success("WhatsApp message(s) queued/sent!")
            else:
                st.error("Issue sending one or more WhatsApp messages.")
        else:
            st.warning("No contacts selected or message empty.")

with tab_dashboard:  # ... (original dashboard logic, ensure queries are efficient for large data)
    st.header("📊 Order Analysis Dashboard")
    # Dashboard queries run when this tab is active. For very large datasets,
    # consider pre-aggregating data or adding filters/pagination.
    conn_dash = connect_to_db()
    if conn_dash:
        try:
            with conn_dash.cursor() as cur:
                st.subheader("📈 Key Metrics");
                c1, c2, c3 = st.columns(3)
                cur.execute("SELECT COUNT(*) FROM orders");
                total_orders = cur.fetchone()[0]
                c1.metric("Total Orders", total_orders or 0)
                cur.execute("SELECT SUM(price * quantity) FROM orders WHERE payment_status = 'Paid'");
                total_revenue = cur.fetchone()[0]
                c2.metric("Total Revenue (Paid)", f"₹{total_revenue or 0:.2f}")
                cur.execute("SELECT AVG(price * quantity) FROM orders WHERE quantity > 0 AND price > 0");
                avg_order_value = cur.fetchone()[0]
                c3.metric("Avg. Order Value", f"₹{avg_order_value or 0:.2f}")
                st.divider()

                st.subheader("📊 Order Status Distribution")
                cur.execute("SELECT order_status, COUNT(*) as count FROM orders GROUP BY order_status")
                status_data = cur.fetchall()
                if status_data:
                    df_status = pd.DataFrame(status_data, columns=['Order Status', 'Count']).set_index(
                        'Order Status'); st.bar_chart(df_status)
                else:
                    st.write("No order status data.")
                st.divider()

                st.subheader("📦 Sales by Category")
                cur.execute(
                    "SELECT category, SUM(price * quantity) as total_sales FROM orders WHERE category IS NOT NULL AND category != '' AND quantity > 0 AND price > 0 GROUP BY category ORDER BY total_sales DESC")
                cat_sales_data = cur.fetchall()
                if cat_sales_data:
                    df_cat_sales = pd.DataFrame(cat_sales_data, columns=['Category', 'Total Sales']).set_index(
                        'Category')
                    if not df_cat_sales.empty:
                        fig, ax = plt.subplots();
                        df_cat_sales.plot(kind='pie', y='Total Sales', ax=ax, autopct='%1.1f%%', legend=False);
                        ax.set_ylabel('');
                        st.pyplot(fig)
                    else:
                        st.write("No category sales data to plot.")
                else:
                    st.write("No category sales data.")
                st.divider()

                st.subheader("📋 Detailed Recent Orders (Last 15)")
                cur.execute(
                    "SELECT id, order_date, product_name, customer_name, (price * quantity) as total_value, order_status, payment_status, message_sent FROM orders ORDER BY order_date DESC LIMIT 15")
                detailed_orders = cur.fetchall()
                if detailed_orders:
                    df_detailed = pd.DataFrame(detailed_orders,
                                               columns=['ID', 'Order Date', 'Product', 'Customer', 'Total Value',
                                                        'Status', 'Payment', 'Notified'])
                    st.dataframe(df_detailed, use_container_width=True, hide_index=True)
                else:
                    st.write("No orders to display.")
        except psycopg2.Error as e:
            st.error(f"Dashboard DB error: {e}")
        except Exception as e_dash:
            st.error(f"Error generating dashboard: {e_dash}")
        finally:
            conn_dash.close()
    else:
        st.warning("Database not connected. Cannot display dashboard.")

# --- Background Task (Scheduler) ---
# This section sets up the scheduler if auto_check_enabled.
# The actual email processing (slow part) happens in the thread, not blocking the main app.
if st.session_state.auto_check_enabled:
    current_subject_query = st.session_state.email_search_query
    current_interval = st.session_state.check_interval_minutes
    query_changed = st.session_state.last_scheduled_query != current_subject_query
    interval_changed = st.session_state.last_scheduled_interval != current_interval

    if not st.session_state.scheduler_started or query_changed or interval_changed:
        schedule.clear()
        job = schedule.every(current_interval).minutes.do(
            check_and_process_emails_automatically,
            subject_query_for_scheduler=current_subject_query
        )
        st.session_state.last_scheduled_query = current_subject_query
        st.session_state.last_scheduled_interval = current_interval
        print(f"Scheduler: Job (re)defined for '{current_subject_query}' every {current_interval} mins. Job: {job}")

        if not st.session_state.scheduler_started:
            if st.session_state.scheduler_thread is None or not st.session_state.scheduler_thread.is_alive():
                thread = threading.Thread(target=run_scheduled_tasks, daemon=True, name="SchedulerThread")
                thread.start()
                st.session_state.scheduler_thread = thread
                st.session_state.scheduler_started = True
                print("Scheduler: Thread started.")
                st.toast(f"Auto-email check scheduled.", icon="⏰")  # Brief toast
            else:
                print("Scheduler: Thread already running. Job updated.")
        else:
            print(f"Scheduler: Job updated."); st.toast(f"Auto-email check updated.", icon="🔄")

elif not st.session_state.auto_check_enabled and st.session_state.scheduler_started:
    schedule.clear()
    st.session_state.scheduler_started = False
    st.session_state.last_scheduled_query = None;
    st.session_state.last_scheduled_interval = None
    print("Scheduler: Auto-check disabled. Jobs cleared.")
    st.toast("Auto-email check disabled.", icon="🛑")

print(
    f"Script rerun at {datetime.now()}. Auto-check: {st.session_state.auto_check_enabled}, Scheduler started: {st.session_state.scheduler_started}")