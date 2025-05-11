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
from streamlit import secrets

# Secrets for email
EMAIL = st.secrets["gmail_uname"]
PASSWORD = st.secrets["gmail_pwd"]
IMAP_SERVER = "imap.gmail.com"
SELLER_TEAM_RECIPIENTS_STR = st.secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS")

# Database credentials (consider using secrets for these as well)
DB_HOST = st.secrets.get("DB_HOST", "localhost")
DB_NAME = st.secrets.get("DB_NAME", "po_orders")
DB_USER = st.secrets.get("DB_USER", "po_user")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "postdb123") # Example, use secrets
DB_PORT = st.secrets.get("DB_PORT", 5432)


def create_orders_table(conn):
    """Creates the 'orders' table if it doesn't exist."""
    try:
        with conn.cursor() as cursor:
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
                    message_sent BOOLEAN DEFAULT FALSE
                )
            """)
        conn.commit()
        print("Orders table checked/created successfully.")
    except psycopg2.Error as e:
        print(f"Error creating orders table: {e}")
        st.error(f"Error creating orders table: {e}") # Show error in Streamlit UI
        conn.rollback() # Rollback in case of error during table creation
    except Exception as e:
        print(f"An unexpected error occurred during table creation: {e}")
        st.error(f"An unexpected error occurred during table creation: {e}")
        if conn: # Ensure conn is not None before trying to rollback
            conn.rollback()

def connect_to_db():
    """Connects to the PostgreSQL database and ensures the orders table exists."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("Database connection successful.")
        # Ensure the table is created after connection
        create_orders_table(conn)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        st.error(f"Database connection error: {e}") # Show error in Streamlit UI
        return None
    except Exception as e:
        print(f"An unexpected error occurred during database connection: {e}")
        st.error(f"An unexpected error occurred during database connection: {e}")
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
                "payment_status", "order_status", "message_sent"
            ]
            # Ensure all keys exist, providing defaults if necessary
            values = [
                order_details.get("Product Name"),
                order_details.get("Category", ""), # Default if not found
                order_details.get("Price"),
                order_details.get("Quantity"),
                order_details.get("Order Date"),
                order_details.get("Delivery Date"),
                order_details.get("Customer Name"),
                order_details.get("Raw Customer Phone"), # This should be present from parsing
                order_details.get("Email", ""), # Default if not found
                order_details.get("Address"),
                order_details.get("Payment Method"),
                order_details.get("Payment Status"),
                order_details.get("Order Status"),
                False # message_sent defaults to False
            ]

            # Basic validation for required fields before attempting insert
            if not all([order_details.get("Product Name"), order_details.get("Price") is not None, order_details.get("Quantity") is not None,
                        order_details.get("Order Date"), order_details.get("Delivery Date"), order_details.get("Customer Name"),
                        order_details.get("Address")]):
                st.error("One or more required fields are missing in order_details. Cannot store order.")
                print(f"Validation failed for order_details: {order_details}")
                return False

            sql = f"""
                INSERT INTO orders ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
            """
            print("SQL Query:", sql)
            print("Values for DB:", tuple(values)) # Ensure values are logged correctly
            cursor.execute(sql, tuple(values))
        conn.commit()
        print(f"Order for {order_details.get('Product Name')} stored successfully.")
        return True
    except psycopg2.Error as e: # More specific psycopg2 error
        conn.rollback()
        print(f"Database error storing order: {e}", type(e))
        st.error(f"Database error storing order: {e}")
        return False
    except Exception as e:
        if conn: # Check if conn is not None
            conn.rollback()
        print(f"Store order error: {e}", type(e))
        st.error(f"Failed to store order: {e}")
        return False

# --- Session State Initialization ---
for key, default in {
    "whatsapp_sent_counter": 0,
    "whatsapp_errors": [],
    "manual_order_sent": False,
    "last_check_time": datetime.now(),
    "auto_check_enabled": True,
    "check_interval_minutes": 30,
    "sending_in_progress": False,
    "email_search_query": "Purchase Order",
    "scheduler_started": False, # Added to ensure scheduler starts only once
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_bytes):
    try:
        pdf_file_like_object = io.BytesIO(pdf_file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file_like_object)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def format_phone_number(phone_str):
    if not phone_str or not isinstance(phone_str, str):
        return None
    try:
        # Attempt to parse with a default region if it's a local number without country code
        if not phone_str.startswith('+'):
            # Try parsing with IN, common in India. Adjust if needed.
            phone_number = phonenumbers.parse(phone_str, "IN")
        else:
            phone_number = phonenumbers.parse(phone_str, None)

        if phonenumbers.is_valid_number(phone_number):
            formatted = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)
            return formatted
        return None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None

def parse_order_details(text):
    patterns = {
        "Order ID": r"Order ID:?\s*([A-Z0-9-]+)",
        "Product Name": r"Product(?: Name)?:?\s*(.+?)(?:\nCategory:|\nPrice:|\nQuantity:|$)", # Made non-greedy
        "Category": r"Category:?\s*(.+?)(?:\nPrice:|\nQuantity:|\nOrder Date:|$)", # Made non-greedy
        "Price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "Quantity": r"Quantity:?\s*(\d+)",
        "Order Date": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
        "Delivery Date": r"(?:Expected )?Delivery(?: Date)?:?\s*(\d{4}-\d{2}-\d{2})",
        "Customer Name": r"Customer(?: Name)?:?\s*(.+?)(?:\nPhone:|\nEmail:|\nAddress:|$)", # Made non-greedy
        "Phone": r"Phone:?\s*(\+?\d[\d\s-]{8,15}\d)",
        "Email": r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        "Address": r"Address:?\s*(.+?)", # Will be refined below
        "Payment Method": r"Payment(?: Method)?:?\s*(COD|Cash on Delivery|Credit Card|UPI|Bank Transfer)",
        "Payment Status": r"Payment Status:?\s*(Paid|Unpaid|Pending)",
        "Order Status": r"(?:Order )?Status:?\s*(Pending|Processing|Shipped|Delivered|Cancelled)",
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
                        order_details[key] = "Not found" # Or handle error appropriately
            elif key == "Delivery Date" and value:
                try:
                    order_details[key] = datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    order_details[key] = "Not found" # Or handle error appropriately
            else:
                order_details[key] = value
        else:
            order_details[key] = "Not found" if key not in ["Category", "Email"] else "" # Some fields can be empty

    # Refined Address Parsing
    # Try to capture address until a known subsequent field or end of relevant block
    address_pattern_refined = r"Address:?\s*(.*?)(?:Payment Method:|Payment Status:|Order Status:|Notes:|Tracking ID:|Order ID:|Customer Name:|Phone:|Email:|---|$)"
    match_addr = re.search(address_pattern_refined, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match_addr:
        address_candidate = match_addr.group(1).strip().replace('\n', ' ')
        # Avoid capturing other field names if they are part of the address block due to poor formatting
        stop_keywords = ["Payment Method:", "Payment Status:", "Order Status:", "Notes:", "Tracking ID:"]
        for keyword in stop_keywords:
            if keyword.lower() in address_candidate.lower():
                address_candidate = address_candidate.lower().split(keyword.lower())[0].strip()
        order_details["Address"] = address_candidate
    elif order_details["Address"] == "Not found" or len(order_details["Address"]) < 10 : # Fallback or if initial was bad
        order_details["Address"] = "Not found"


    order_details["Raw Customer Phone"] = order_details.get("Phone", "Not found")
    # Format phone after extraction
    # formatted_phone = format_phone_number(order_details.get("Phone"))
    # order_details["Formatted Customer Phone"] = formatted_phone if formatted_phone else order_details.get("Phone", "Not found")

    # Validate and ensure essential fields are present
    essential_fields = ["Product Name", "Price", "Quantity", "Order Date", "Delivery Date", "Customer Name", "Address"]
    for field in essential_fields:
        if order_details.get(field) == "Not found" or order_details.get(field) is None:
            print(f"Warning: Essential field '{field}' not found or invalid in parsed text.")
            # Potentially mark this order as needing review or handle error
    return order_details

def send_whatsapp_message_pywhatkit(message, recipient_numbers_list):
    status_container = st.empty()
    if not isinstance(recipient_numbers_list, (list, set)):
        status_container.error("Invalid recipient numbers format for pywhatkit. Must be a list or set.")
        st.session_state.whatsapp_errors.append("Invalid recipient format.")
        return False
    if not recipient_numbers_list:
        status_container.error("No valid recipient numbers provided for pywhatkit.")
        st.session_state.whatsapp_errors.append("No recipients.")
        return False

    sent_successfully = True
    for recipient in recipient_numbers_list:
        try:
            # PyWhatKit needs numbers in E.164 format (e.g., +91XXXXXXXXXX)
            formatted_recipient = format_phone_number(recipient)
            if not formatted_recipient:
                st.warning(f"Skipping invalid phone number for pywhatkit: {recipient}")
                st.session_state.whatsapp_errors.append(f"Invalid number: {recipient}")
                sent_successfully = False
                continue

            current_time = datetime.now()
            send_hour = current_time.hour
            send_minute = (current_time.minute + 1) % 60 # Send in the next minute
            if send_minute == 0 and current_time.minute == 59 : # If next minute is 0, increment hour
                 send_hour = (send_hour + 1) % 24

            status_container.info(f"Attempting to send to {formatted_recipient} via PyWhatKit at {send_hour:02d}:{send_minute:02d}...")
            pywhatkit.sendwhatmsg(formatted_recipient, message, send_hour, send_minute, wait_time=15, tab_close=True, close_time=5)
            status_container.success(f"Message queued for {formatted_recipient} via PyWhatKit!")
            st.session_state.whatsapp_sent_counter +=1
            time.sleep(5) # Give some time between messages
        except Exception as e:
            status_container.error(f"Error sending message to {recipient} via PyWhatKit: {e}")
            st.session_state.whatsapp_errors.append(f"Error for {recipient}: {str(e)}")
            sent_successfully = False
    return sent_successfully


def send_whatsapp_message_web(message, recipient_numbers_list):
    status_container = st.empty()
    if not isinstance(recipient_numbers_list, (list, set)):
        status_container.error("Invalid recipient numbers format. Must be a list or set.")
        return False
    if not recipient_numbers_list:
        status_container.error("No valid recipient numbers provided.")
        return False

    webbrowser.open("https://web.whatsapp.com")
    status_container.info("Opened WhatsApp Web. Please scan QR code if needed. Waiting 20 seconds...")
    time.sleep(20) # Increased wait time for QR scan

    sent_successfully = True
    for recipient in recipient_numbers_list:
        try:
            formatted_recipient = format_phone_number(recipient)
            if not formatted_recipient:
                st.warning(f"Skipping invalid phone number for web: {recipient}")
                sent_successfully = False
                continue

            # WhatsApp URL encoding for message
            import urllib.parse
            encoded_message = urllib.parse.quote(message)
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_recipient}&text={encoded_message}"

            webbrowser.open(whatsapp_url)
            status_container.info(f"Opened chat for {formatted_recipient}. Waiting 15 seconds for page load...")
            time.sleep(15)  # Wait for the chat to open and page to load

            # Try to send Enter. This might need focus on the input box.
            # For reliability, manual 'Enter' press by user might still be best here.
            pyautogui.press("enter")
            status_container.success(f"Message sent to {formatted_recipient} (Enter key pressed).")
            st.session_state.whatsapp_sent_counter += 1
            time.sleep(random.uniform(3, 6)) # Random delay
        except Exception as e:
            status_container.error(f"Error sending message to {formatted_recipient} via web: {e}")
            st.session_state.whatsapp_errors.append(f"Error for {formatted_recipient} (web): {str(e)}")
            sent_successfully = False
    return sent_successfully

# Choose which WhatsApp method to use:
# send_whatsapp_notification = send_whatsapp_message_pywhatkit
send_whatsapp_notification = send_whatsapp_message_web # Or choose _web


def fetch_email_pdfs(subject_query):
    pdf_files_with_info = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        # Fetch UNSEEN emails first, then SEEN if no UNSEEN found with the subject
        search_criteria_unseen = f'(UNSEEN SUBJECT "{subject_query}")'
        search_criteria_seen = f'(SEEN SUBJECT "{subject_query}")' # Fallback

        status, messages = mail.search(None, search_criteria_unseen)
        if status != "OK" or not messages[0]: # No unseen emails
            print(f"No UNSEEN emails found with subject '{subject_query}'. Trying SEEN emails.")
            status, messages = mail.search(None, search_criteria_seen) # Try seen
            if status != "OK" or not messages[0]:
                mail.logout()
                print(f"No SEEN emails found with subject '{subject_query}' either.")
                return []

        mail_ids = messages[0].split()
        print(f"Found {len(mail_ids)} email(s) with subject query '{subject_query}'.")

        for mail_id in reversed(mail_ids): # Process newest first
            fetch_status, msg_data = mail.fetch(mail_id, '(RFC822)')
            if fetch_status != "OK":
                continue
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject_tuple = decode_header(msg["Subject"])[0]
            subject = subject_tuple[0].decode(subject_tuple[1] or "utf-8") if isinstance(subject_tuple[0], bytes) else subject_tuple[0]

            sender_tuple = decode_header(msg.get("From"))[0]
            sender = sender_tuple[0].decode(sender_tuple[1] or "utf-8") if isinstance(sender_tuple[0], bytes) else sender_tuple[0]

            for part in msg.walk():
                if part.get_content_type() == "application/pdf" and part.get('Content-Disposition'):
                    filename = part.get_filename()
                    if filename: # Ensure filename exists
                        decoded_filename_tuple = decode_header(filename)[0]
                        decoded_filename = decoded_filename_tuple[0].decode(decoded_filename_tuple[1] or 'utf-8') if isinstance(decoded_filename_tuple[0], bytes) else decoded_filename_tuple[0]

                        pdf_data = part.get_payload(decode=True)
                        if pdf_data:
                            pdf_files_with_info.append({
                                "data": pdf_data,
                                "filename": decoded_filename,
                                "sender": sender,
                                "subject": subject,
                                "email_id": mail_id.decode()
                            })
                            # Mark email as read (SEEN) after successful processing of PDF
                            # mail.store(mail_id, '+FLAGS', '\\Seen') # Uncomment if you want to mark as read
            # mail.store(mail_id, '+FLAGS', '\\Seen') # Mark as read after processing all parts, or only if PDF found
        mail.logout()
        return pdf_files_with_info
    except imaplib.IMAP4.error as e: # Specific IMAP error
        st.error(f"IMAP Error fetching emails: {str(e)}")
        print(f"IMAP Error fetching emails: {str(e)}")
        return []
    except Exception as e:
        st.error(f"General error fetching emails: {str(e)}")
        print(f"General error fetching emails: {str(e)}")
        return []


def get_seller_team_recipients(seller_team_recipients_str):
    recipients = set()
    if seller_team_recipients_str and isinstance(seller_team_recipients_str, str):
        for phone in seller_team_recipients_str.split(","):
            formatted_seller = format_phone_number(phone.strip())
            if formatted_seller:
                recipients.add(formatted_seller)
            else:
                st.warning(f"Invalid seller phone number in config: {phone.strip()}")
    return list(recipients) # Return as list for pywhatkit or web iteration

def get_pending_message_count():
    conn = connect_to_db()
    if not conn:
        # st.error("Failed to connect to database to get message count.") # Already handled in connect_to_db
        return 0
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM orders WHERE message_sent = FALSE")
            count = cursor.fetchone()[0]
            return count
    except psycopg2.Error as e:
        st.error(f"Database error getting message count: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def send_whatsapp_from_db():
    status_container = st.empty()
    conn = connect_to_db()
    if not conn:
        status_container.error("Failed to connect to the database to send pending messages.")
        return

    orders_sent_successfully = 0
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, product_name, category, price, quantity, order_date, delivery_date,
                customer_name, customer_phone, email, address, payment_method, payment_status, order_status
                FROM orders
                WHERE message_sent = FALSE
                ORDER BY order_date ASC
                """
            ) # Process older orders first
            pending_orders = cursor.fetchall()

            if not pending_orders:
                status_container.info("No pending WhatsApp messages from database to send.")
                return

            seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
            if not seller_numbers:
                status_container.error("No valid seller WhatsApp numbers configured in secrets.")
                return

            status_container.info(f"Found {len(pending_orders)} pending order(s) to notify via WhatsApp.")

            for order_data in pending_orders:
                order_id, product_name, category, price, quantity, order_date, delivery_date, \
                customer_name, customer_phone, cust_email, address, payment_method, \
                payment_status, order_status = order_data

                message_lines = [
                    "📦 *New Order Notification (from DB)* 📦",
                    f"Order ID: DB-{order_id}", # Distinguish from direct email orders if needed
                    f"🛍️ Product: {product_name or 'N/A'}",
                    f"🏷️ Category: {category or 'N/A'}",
                    f"💰 Price: ₹{price:.2f}" if price is not None else "Price: N/A",
                    f"🔢 Quantity: {quantity or 'N/A'}",
                    f"🗓️ Order Date: {order_date.strftime('%Y-%m-%d %H:%M') if order_date else 'N/A'}",
                    f"🚚 Exp. Delivery: {delivery_date.strftime('%Y-%m-%d') if delivery_date else 'N/A'}",
                    f"👤 Customer: {customer_name or 'N/A'}",
                    f"📞 Cust. Phone (Ref): {customer_phone or 'N/A'}",
                    f"📧 Cust. Email: {cust_email or 'N/A'}",
                    f"🏠 Address: {address or 'N/A'}",
                    f"💳 Payment: {payment_method or 'N/A'} ({payment_status or 'N/A'})",
                    f"📊 Status: {order_status or 'N/A'}",
                ]
                formatted_message_text = "\n".join(message_lines)

                st.write(f"Preparing to send: {formatted_message_text}") # For debugging in Streamlit

                if send_whatsapp_notification(formatted_message_text, seller_numbers):
                    cursor.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (order_id,))
                    conn.commit() # Commit after each successful send and DB update
                    status_container.success(f"Notification for order DB-{order_id} sent and marked as processed.")
                    orders_sent_successfully +=1
                    time.sleep(random.uniform(5, 10)) # Pause between messages
                else:
                    status_container.error(f"Failed to send WhatsApp for order DB-{order_id}. It will be retried later.")
                    # No rollback needed here for the main transaction, just the update for this order failed.

            if orders_sent_successfully > 0:
                st.success(f"Successfully sent notifications for {orders_sent_successfully} order(s).")
            if orders_sent_successfully < len(pending_orders):
                 st.warning(f"Could not send notifications for {len(pending_orders) - orders_sent_successfully} order(s). Check errors.")


    except psycopg2.Error as e:
        status_container.error(f"Database error during pending message processing: {e}")
        if conn:
            conn.rollback() # Rollback the overall transaction if a DB error occurs
    except Exception as e:
        status_container.error(f"An unexpected error occurred sending messages from DB: {e}")
    finally:
        if conn:
            conn.close()


def process_and_store_email_orders(subject_query):
    status_container = st.empty()
    status_container.info(f"Checking for new PO emails with subject: '{subject_query}'...")
    pdf_files_info = fetch_email_pdfs(subject_query)

    if not pdf_files_info:
        status_container.info("No new PO emails with parsable PDFs found.")
        return

    processed_count = 0
    failed_store_count = 0
    conn = connect_to_db() # Establish connection once for all PDFs in this batch
    if not conn:
        status_container.error("Failed to connect to the database. Cannot process email orders.")
        return

    with st.spinner(f"Processing {len(pdf_files_info)} email(s)..."):
        for pdf_info in pdf_files_info:
            try:
                st.write(f"Processing PDF: {pdf_info['filename']} from {pdf_info['sender']}")
                text = extract_text_from_pdf(pdf_info["data"])
                if text:
                    order_details = parse_order_details(text)
                    # Add email specific info if needed, e.g., original sender, email_id
                    order_details["Source Email Subject"] = pdf_info["subject"]
                    order_details["Source Email Sender"] = pdf_info["sender"]

                    # Basic check if essential details were parsed
                    if order_details.get("Product Name") == "Not found" or order_details.get("Price") is None:
                        st.warning(f"Could not parse essential details from {pdf_info['filename']}. Skipping store.")
                        print(f"Skipping store for {pdf_info['filename']} due to missing essential details. Parsed: {order_details}")
                        failed_store_count +=1
                        continue

                    if store_order(conn, order_details):
                        processed_count += 1
                        st.success(f"Stored order from '{pdf_info['filename']}' (Subject: {pdf_info['subject']})")
                    else:
                        st.error(f"Failed to store order from '{pdf_info['filename']}'. See logs for details.")
                        failed_store_count +=1
                else:
                    st.warning(f"No text extracted from PDF: {pdf_info['filename']}")
                    failed_store_count +=1
            except Exception as e:
                st.error(f"Error processing PDF '{pdf_info['filename']}': {str(e)}")
                failed_store_count +=1

    if conn:
        conn.close()

    if processed_count > 0:
        status_container.success(f"Successfully processed and stored {processed_count} order(s) from emails.")
    if failed_store_count > 0:
        status_container.warning(f"Failed to process/store {failed_store_count} order(s)/PDF(s) from emails. Check details above/logs.")
    if processed_count == 0 and failed_store_count == 0 and pdf_files_info: # PDFs found but none led to stored order
        status_container.info("Found email(s) with PDFs, but no new orders were stored (e.g. parsing issues or already processed).")


def check_and_process_emails_automatically(subject_query_for_scheduler):
    print(f"[{datetime.now()}] Automatic email check triggered for subject: '{subject_query_for_scheduler}'")
    st.toast(f"Auto-checking emails for '{subject_query_for_scheduler}'...", icon="📧")
    process_and_store_email_orders(subject_query_for_scheduler)
    st.session_state["last_check_time"] = datetime.now()
    # After processing emails, try to send any newly added (and previously pending) messages from DB
    print(f"[{datetime.now()}] Attempting to send pending WhatsApp messages from DB after email check.")
    send_whatsapp_from_db()
    st.rerun() # Rerun to update UI elements like pending count


def run_scheduled_tasks():
    # Ensure schedule is configured with the correct parameters from session state
    # This function will be run in a thread, so it needs to access session_state carefully if Streamlit context matters
    # For 'do' calls, pass arguments directly rather than relying on Streamlit's current session_state values if they can change.
    # However, for schedule setup, it's usually done once.
    while True:
        schedule.run_pending()
        time.sleep(1)


# --- Streamlit UI ---
st.set_page_config(page_title="PO Order Manager", layout="wide")
st.title("🛒 PO Order Management Dashboard")

# --- Initialize DB Connection & Table ---
# This ensures the DB is ready when the app starts.
# However, functions like connect_to_db() will still be called by operations.
# This initial call is primarily for the create_orders_table check.
initial_conn = connect_to_db()
if initial_conn:
    initial_conn.close()
    if not st.session_state.get("db_init_success", False): # Prevent multiple messages if successful
        st.sidebar.success("Database connected & 'orders' table verified.")
        st.session_state.db_init_success = True
else:
    st.sidebar.error("Initial Database connection failed! App may not function.")
    st.session_state.db_init_success = False


# --- Sidebar ---
with st.sidebar:
    st.header("Status & Configuration")
    st.metric("WhatsApp Sent (Session)", st.session_state.whatsapp_sent_counter)
    if st.session_state.whatsapp_errors:
        with st.expander("WhatsApp Sending Errors (Session)"):
            for err in st.session_state.whatsapp_errors:
                st.error(err)

    st.subheader("📧 Email PO Processing")
    current_email_query = st.session_state.get("email_search_query", "Purchase Order")
    subject_query_input = st.text_input(
        "Email Subject to Search",
        value=current_email_query,
        key="sidebar_email_search_query_input" # Use a different key if main tab has one
    )
    if subject_query_input != current_email_query:
        st.session_state["email_search_query"] = subject_query_input # Update session state
        st.rerun() # Rerun to reflect change and potentially re-schedule

    if st.button("Manually Check Emails & Store Orders", key="manual_email_check_button"):
        with st.spinner("Checking emails..."):
            process_and_store_email_orders(st.session_state["email_search_query"])
            send_whatsapp_from_db() # Also try to send any new or pending messages
        st.rerun()

    st.subheader("⚙️ Auto-Check Settings")
    auto_check_enabled = st.checkbox("Enable Auto-Email Check & Notify", value=st.session_state["auto_check_enabled"], key="auto_check_toggle")
    if auto_check_enabled != st.session_state["auto_check_enabled"]:
        st.session_state["auto_check_enabled"] = auto_check_enabled
        if auto_check_enabled and not st.session_state.scheduler_started:
            st.warning("Auto-check enabled. Scheduler will be (re)started. Ensure interval is set.")
            # Need to handle scheduler re-initialization if interval changes or it's re-enabled
        elif not auto_check_enabled:
            st.info("Auto-check disabled. Scheduler will be cleared.")
            schedule.clear() # Stop all scheduled tasks
            st.session_state.scheduler_started = False # Allow restart if re-enabled
        st.rerun()

    interval_minutes = st.slider("Check Interval (minutes)", 5, 120, st.session_state["check_interval_minutes"], 5, key="interval_slider")
    if interval_minutes != st.session_state["check_interval_minutes"]:
        st.session_state["check_interval_minutes"] = interval_minutes
        if st.session_state.auto_check_enabled: # If auto-check is on, reschedule
            schedule.clear()
            st.session_state.scheduler_started = False # Force re-init of scheduler
            st.warning(f"Interval changed to {interval_minutes} min. Auto-check will use this new interval.")
        st.rerun()

    last_check_str = st.session_state["last_check_time"].strftime("%Y-%m-%d %H:%M:%S")
    st.text(f"Last Auto-Email Check: {last_check_str}")
    if st.session_state.auto_check_enabled and st.session_state.scheduler_started:
        next_run = schedule.next_run()
        if next_run:
            st.text(f"Next Auto-Check: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.text("Next Auto-Check: Not scheduled (or just ran).")


    st.subheader("📱 Database WhatsApp Notifications")
    pending_count = get_pending_message_count() # Get fresh count
    st.metric("Pending Orders in DB for WhatsApp", pending_count)
    if st.button("Send All Pending Order Notifications from DB", key="send_pending_db_whatsapp"):
        if pending_count > 0:
            with st.spinner(f"Sending {pending_count} pending notifications..."):
                send_whatsapp_from_db()
            st.rerun() # Update the pending count and UI
        else:
            st.info("No pending notifications in the database to send.")


# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📬 Email PO Summary & Actions", "📝 Manual PO Entry", "📞 Quick WhatsApp"])

with tab1:
    st.subheader("📧 Email Order Extraction & Processing")
    st.write(f"Currently searching emails with subject containing: **'{st.session_state['email_search_query']}'** (change in sidebar).")
    st.info("Use the 'Manually Check Emails' button in the sidebar to fetch and process orders from emails. Automatic checking can also be enabled.")
    # Display recently processed orders or logs here if desired

    st.subheader("📋 Recent Orders (from Database)")
    conn_tab1 = connect_to_db()
    if conn_tab1:
        try:
            with conn_tab1.cursor() as cur:
                cur.execute("SELECT order_date, product_name, customer_name, price, quantity, order_status, message_sent FROM orders ORDER BY order_date DESC LIMIT 10")
                recent_orders = cur.fetchall()
                if recent_orders:
                    st.dataframe(recent_orders, use_container_width=True)
                else:
                    st.write("No orders found in the database yet.")
        except Exception as e:
            st.error(f"Could not load recent orders: {e}")
        finally:
            conn_tab1.close()
    else:
        st.warning("Database not connected. Cannot display recent orders.")


with tab2:
    st.subheader("✍️ Manual Order Entry & Instant Notification")
    with st.expander("➕ Add New Purchase Order Manually", expanded=True):
        with st.form("manual_order_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                m_product_name = st.text_input("Product Name *", key="m_prod")
                m_category = st.text_input("Category", key="m_cat")
                m_price = st.number_input("Price (₹) *", min_value=0.01, step=0.01, format="%.2f", key="m_price")
                m_quantity = st.number_input("Quantity *", min_value=1, step=1, key="m_qty")
                m_order_date = st.date_input("Order Date *", value=datetime.now().date(), key="m_odate")
                m_order_time = st.time_input("Order Time *", value=datetime.now().time(), key="m_otime")

            with col2:
                m_delivery_date = st.date_input("Expected Delivery Date *", value=(datetime.now() + timedelta(days=7)).date(), key="m_ddate")
                m_customer_name = st.text_input("Customer Name *", key="m_cname")
                m_customer_phone_raw = st.text_input("Customer Phone (for notification & ref)", key="m_cphone")
                m_email = st.text_input("Customer Email", key="m_cemail")
                m_address = st.text_area("Delivery Address *", key="m_addr")
                m_payment_method = st.selectbox("Payment Method", ["COD", "Credit Card", "UPI", "Bank Transfer", "Other"], key="m_pmethod")
                m_payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Pending"], key="m_pstatus")
                m_order_status = st.selectbox("Order Status", ["Pending", "Processing", "Confirmed", "Shipped", "Delivered", "Cancelled"], key="m_ostatus")

            submit_manual_order = st.form_submit_button("Submit Order & Notify Seller Team")

            if submit_manual_order:
                required_fields = [
                    m_product_name, m_price, m_quantity, m_order_date,
                    m_order_time, m_delivery_date, m_customer_name, m_address
                ]
                if not all(required_fields) or m_price <= 0 or m_quantity <=0: # Basic validation
                    st.error("Please fill in all required fields (*) with valid values.")
                else:
                    order_datetime = datetime.combine(m_order_date, m_order_time)
                    manual_order_details = {
                        "Product Name": m_product_name, "Category": m_category, "Price": float(m_price),
                        "Quantity": int(m_quantity), "Order Date": order_datetime, # Store as datetime object
                        "Delivery Date": m_delivery_date, # Store as date object
                        "Customer Name": m_customer_name, "Raw Customer Phone": m_customer_phone_raw,
                        "Email": m_email, "Address": m_address, "Payment Method": m_payment_method,
                        "Payment Status": m_payment_status, "Order Status": m_order_status,
                        # "message_sent" will be False by default in DB, or handled by store_order
                    }

                    conn_manual = connect_to_db()
                    if conn_manual:
                        if store_order(conn_manual, manual_order_details):
                            st.success("Manual order details stored successfully in database!")
                            st.session_state.manual_order_sent = True # Flag for potential UI update

                            # Prepare message for WhatsApp notification
                            message_lines = [
                                "📦 *New Manual PO Order Received!* 📦",
                                f"🛍️ Product: {m_product_name}",
                                f"🏷️ Category: {m_category if m_category else 'N/A'}",
                                f"💰 Price: ₹{m_price:.2f}",
                                f"🔢 Quantity: {m_quantity}",
                                f"🗓️ Order Date: {order_datetime.strftime('%Y-%m-%d %H:%M')}",
                                f"🚚 Exp. Delivery: {m_delivery_date.strftime('%Y-%m-%d')}",
                                f"👤 Customer: {m_customer_name}",
                                f"📞 Cust. Phone (Ref): {m_customer_phone_raw if m_customer_phone_raw else 'N/A'}",
                                f"📧 Cust. Email: {m_email if m_email else 'N/A'}",
                                f"🏠 Address: {m_address}",
                                f"💳 Payment: {m_payment_method} ({m_payment_status})",
                                f"📊 Status: {m_order_status}",
                            ]
                            formatted_message = "\n".join(message_lines)
                            seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)

                            if seller_numbers:
                                st.info(f"Sending WhatsApp notification for manual order to: {', '.join(seller_numbers)}")
                                if send_whatsapp_notification(formatted_message, seller_numbers):
                                    st.success("WhatsApp notification sent successfully to seller team!")
                                    # Update the message_sent status in DB for this order
                                    try:
                                        with conn_manual.cursor() as cur:
                                            # Find the ID of the just inserted order (assuming no immediate other inserts)
                                            # This is a bit fragile; ideally store_order would return the ID.
                                            cur.execute("UPDATE orders SET message_sent = TRUE WHERE product_name = %s AND customer_name = %s AND order_date = %s AND message_sent = FALSE ORDER BY id DESC LIMIT 1",
                                                        (m_product_name, m_customer_name, order_datetime))
                                            conn_manual.commit()
                                            if cur.rowcount > 0:
                                                print(f"Marked manual order for {m_product_name} as message_sent=TRUE.")
                                            else:
                                                print(f"Could not find or update manual order for {m_product_name} to set message_sent=TRUE immediately.")
                                    except Exception as e_update:
                                        st.warning(f"Could not mark manual order as sent in DB: {e_update}")
                                else:
                                    st.error("Failed to send WhatsApp notification for the manual order. It's stored in DB with message_sent=FALSE.")
                            else:
                                st.warning("No seller WhatsApp numbers configured. Notification not sent for manual order (but it is stored).")
                        else:
                            st.error("Failed to store manual order details in the database.")
                        conn_manual.close()
                    else:
                        st.error("Database connection failed. Cannot store manual order.")
                    st.rerun() # Rerun to update pending counts etc.

with tab3:
    st.subheader("📱 Quick WhatsApp to Contacts")
    # --- Contact Dictionary ---
    contact_dict = {
        "Narayan": "+919067847003", # Standardize format slightly for easier parsing later if needed
        "Rani Bhise": "+917070242402",
        "Abhishek": "+919284625240",
        "Damini": "+917499353409",
        "Sandeep": "+919850030215",
        "Chandrakant": "+919665934999",
        "Vikas Kumbharkar": "+919284238738",
        # Add more contacts here
    }

    custom_message = st.text_area("Custom Message to Send", "Hello! Just a quick update: ", height=100, key="quick_msg_text")
    st.write("Select contacts to send the message to:")

    selected_contacts_numbers = []
    cols = st.columns(3) # Adjust number of columns for checkbox layout
    contact_items = list(contact_dict.items())

    for i, (contact_name, phone_number) in enumerate(contact_items):
        column_index = i % 3
        if cols[column_index].checkbox(f"{contact_name} ({phone_number})", key=f"checkbox_contact_{contact_name.replace(' ', '_')}"):
            formatted_num = format_phone_number(phone_number) # Validate/format number
            if formatted_num:
                selected_contacts_numbers.append(formatted_num)
            else:
                st.warning(f"Invalid phone number format for {contact_name}: {phone_number}. Skipping.")


    if st.button("Send WhatsApp to Selected Contacts", key="quick_send_button"):
        if not custom_message.strip():
            st.warning("Please enter a message to send.")
        elif selected_contacts_numbers:
            st.info(f"Sending message: '{custom_message}' to {len(selected_contacts_numbers)} selected contact(s)...")
            if send_whatsapp_notification(custom_message, selected_contacts_numbers):
                st.success("WhatsApp message(s) queued/sent to selected contacts!")
            else:
                st.error("There was an issue sending one or more WhatsApp messages. Check logs/above errors.")
        else:
            st.warning("No contacts selected or message is empty.")


# --- Background Task (Email Checking & Notification Scheduler) ---
# This part sets up the schedule and starts the thread.
# It should only run once.
if st.session_state.auto_check_enabled and not st.session_state.scheduler_started:
    current_subject_query = st.session_state["email_search_query"]
    current_interval = st.session_state["check_interval_minutes"]

    schedule.clear() # Clear any existing schedules before setting a new one
    job = schedule.every(current_interval).minutes.do(
        check_and_process_emails_automatically,
        subject_query_for_scheduler=current_subject_query # Pass current query
    )
    print(f"Scheduled email check for '{current_subject_query}' every {current_interval} minutes. Job: {job}")

    # Only start the thread if it hasn't been started before or if schedule was re-initialized
    # The 'scheduler_started' flag helps manage this.
    # Ensure daemon=True so thread exits when main app exits.
    thread = threading.Thread(target=run_scheduled_tasks, daemon=True, name="SchedulerThread")
    thread.start()
    st.session_state.scheduler_started = True
    print("Scheduler thread started.")
    st.toast(f"Auto-email check scheduled every {current_interval} mins for '{current_subject_query}'.", icon="⏰")

elif not st.session_state.auto_check_enabled and st.session_state.scheduler_started:
    schedule.clear()
    st.session_state.scheduler_started = False # Reset flag
    print("Auto-check disabled. Scheduler cleared and thread will eventually stop if it was just for this.")
    st.toast("Auto-email check disabled and scheduler stopped.", icon="🛑")

# --- Final check for UI refresh if needed (less critical now with targeted st.rerun) ---
# This is a general catch-all; specific reruns are better.
# if st.session_state.get("manual_order_sent"):
#     st.session_state.manual_order_sent = False # Reset flag
#     # st.rerun() # Consider if this is needed or if specific actions already rerun