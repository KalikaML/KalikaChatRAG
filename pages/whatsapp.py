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
from local_postgresql_db import connect_to_db, store_order
from streamlit import secrets

# Secrets for email
EMAIL = st.secrets["gmail_uname"]
PASSWORD = st.secrets["gmail_pwd"]
IMAP_SERVER = "imap.gmail.com"
SELLER_TEAM_RECIPIENTS_STR = st.secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS")

# --- Session State Initialization ---
for key, default in {
    "whatsapp_sent_counter": 0,
    "whatsapp_errors": [],
    "manual_order_sent": False,
    "last_check_time": datetime.now(),
    "auto_check_enabled": True,
    "check_interval_minutes": 30,
    "sending_in_progress": False,
    "email_search_query": "Purchase Order",  # Default search subject
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def format_phone_number(phone_str):
    try:
        phone_number = phonenumbers.parse(phone_str, "IN")
        if phonenumbers.is_valid_number(phone_number):
            formatted = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.E164)
            return formatted
        return None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None

def parse_order_details(text):
    patterns = {
        "Order ID": r"Order ID:?\s*([A-Z0-9-]+)",
        "Product Name": r"Product(?: Name)?:?\s*(.+)",
        "Category": r"Category:?\s*(.+)",
        "Price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "Quantity": r"Quantity:?\s*(\d+)",
        "Order Date": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
        "Delivery Date": r"(?:Expected )?Delivery(?: Date)?:?\s*(\d{4}-\d{2}-\d{2})",
        "Customer Name": r"Customer(?: Name)?:?\s*(.+)",
        "Phone": r"Phone:?\s*(\+?\d[\d\s-]{8,15}\d)",
        "Email": r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        "Address": r"Address:?\s*(.+?)",
        "Payment Method": r"Payment(?: Method)?:?\s*(COD|Cash on Delivery|Credit Card|UPI|Bank Transfer)",
        "Payment Status": r"Payment Status:?\s*(Paid|Unpaid|Pending)",
        "Order Status": r"(?:Order )?Status:?\s*(Pending|Processing|Shipped|Delivered|Cancelled)",
    }
    order_details = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        order_details[key] = (
            match.group(1).strip().replace(",", "") if (match and key == "Price")
            else match.group(1).strip() if match else "Not found"
        )
    if order_details["Address"] == "Not found" or len(order_details["Address"]) < 10:
        match_addr = re.search(
            r"Address:?\s*(.*?)(?:Payment Method:|Payment Status:|Order Status:|Notes:|Tracking ID:|---|$)",
            text, re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if match_addr:
            order_details["Address"] = match_addr.group(1).strip()
    order_details["Raw Customer Phone"] = order_details.get("Phone", "Not found")
    return order_details

def send_whatsapp_message(message, recipient_numbers):
    status_container = st.empty()
    if not isinstance(recipient_numbers, list):
        status_container.error("Invalid recipient numbers format.  Must be a list.")
        return
    if not recipient_numbers:
        status_container.error("No valid recipient numbers provided.")
        return
    webbrowser.open("https://web.whatsapp.com")
    time.sleep(15)
    for recipient in recipient_numbers:
        try:
            whatsapp_url = f"https://web.whatsapp.com/send?phone={recipient}&text={message}"
            webbrowser.open(whatsapp_url)
            time.sleep(10)
            pyautogui.press("enter")
            time.sleep(random.uniform(2, 4))
            status_container.success(f"Message sent to {recipient} successfully!")
        except Exception as e:
            status_container.error(f"Error sending message to {recipient}: {e}")

def fetch_email_pdfs(subject_query):
    pdf_files_with_info = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        # Dynamic subject search
        search_criteria = f'(UNSEEN SUBJECT "{subject_query}")'
        status, messages = mail.search(None, search_criteria)
        if status != "OK" or not messages[0]:
            mail.logout()
            return []
        mail_ids = messages[0].split()
        for mail_id in mail_ids:
            fetch_status, msg_data = mail.fetch(mail_id, '(RFC822)')
            if fetch_status != "OK":
                continue
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject = decode_header(msg["Subject"])[0][0].decode(decode_header(msg["Subject"])[0][1] or "utf-8") if isinstance(decode_header(msg["Subject"])[0][0], bytes) else decode_header(msg["Subject"])[0][0]
            sender = decode_header(msg.get("From"))[0][0].decode(decode_header(msg.get("From"))[0][1] or "utf-8") if isinstance(decode_header(msg.get("From"))[0][0], bytes) else decode_header(msg.get("From"))[0][0]
            for part in msg.walk():
                if part.get_content_type() == "application/pdf" and part.get('Content-Disposition'):
                    filename = part.get_filename() or 'untitled.pdf'
                    pdf_data = part.get_payload(decode=True)
                    if pdf_data:
                        pdf_files_with_info.append({
                            "data": pdf_data,
                            "filename": filename,
                            "sender": sender,
                            "subject": subject,
                            "email_id": mail_id.decode()
                        })
        mail.logout()
        return pdf_files_with_info
    except Exception as e:
        st.error(f"Error fetching emails: {str(e)}")
        return []

def get_seller_team_recipients(seller_team_recipients_str):
    recipients = set()
    if seller_team_recipients_str:
        for phone in seller_team_recipients_str.split(","):
            formatted_seller = format_phone_number(phone.strip())
            if formatted_seller:
                recipients.add(formatted_seller)
    return recipients

def get_pending_message_count():
    conn = connect_to_db()
    if not conn:
        st.error("Failed to connect to database to get message count.")
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
        status_container.error("Failed to connect to the database.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, product_name, category, price, quantity, order_date, delivery_date,
                customer_name, customer_phone, email, address, payment_method, payment_status, order_status
                FROM orders
                WHERE message_sent = FALSE
                """
            )
            pending_orders = cursor.fetchall()
            if pending_orders:
                seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
                if not seller_numbers:
                    status_container.error("No valid seller numbers configured.")
                    return
                webbrowser.open("https://web.whatsapp.com")
                time.sleep(5)
                for order in pending_orders:
                    message_lines = [
                        "New Manual PO Order:",
                        f"Product: {order[1]}",
                        f"Category: {order[2]}",
                        f"Price: ₹{order[3]}",
                        f"Quantity: {order[4]}",
                        f"Order Date: {order[5]}",
                        f"Expected Delivery: {order[6]}",
                        f"Customer: {order[7]}",
                        f"Cust. Phone (for ref): {order[8]}",
                        f"Cust. Email: {order[9]}",
                        f"Address: {order[10]}",
                        f"Payment: {order[11]} ({order[12]})",
                        f"Status: {order[13]}",
                    ]
                    formatted_message = "%0A".join(message_lines)
                    send_whatsapp_message(formatted_message, seller_numbers)
                    cursor.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (order[0],))
                conn.commit()
            else:
                status_container.info("No pending messages to send.")
    except psycopg2.Error as e:
        status_container.error(f"Database error: {e}")
        if conn:
            conn.rollback()

def process_and_store_email_orders(subject_query):
    status_container = st.empty()
    status_container.info("Checking for new PO emails...")
    pdf_files_info = fetch_email_pdfs(subject_query)
    if pdf_files_info:
        processed_count = 0
        conn = connect_to_db()
        if not conn:
            status_container.error("Failed to connect to the database.")
            return
        with st.spinner("Processing emails..."):
            for pdf_info in pdf_files_info:
                try:
                    pdf_file = io.BytesIO(pdf_info["data"])
                    text = extract_text_from_pdf(pdf_file)
                    if text:
                        order_details = parse_order_details(text)
                        if store_order(conn, order_details):
                            processed_count += 1
                            st.success(f"Stored order from {pdf_info['filename']}")
                        else:
                            st.error(f"Failed to store order from {pdf_info['filename']}")
                    else:
                        st.warning(f"No text extracted from {pdf_info['filename']}")
                except Exception as e:
                    st.error(f"Error processing {pdf_info['filename']}: {str(e)}")
        if conn:
            conn.close()
        status_container.success(f"Processed {processed_count} email(s) and stored orders.")
    else:
        status_container.info("No new PO emails found.")

def check_and_process_emails_automatically(subject_query):
    process_and_store_email_orders(subject_query)
    st.session_state["last_check_time"] = datetime.now()

def run_scheduled_tasks():
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Streamlit UI ---
st.set_page_config(page_title="PO Order Manager", layout="wide")
st.title("🛒 PO Order Management Dashboard")

# --- Sidebar ---
st.sidebar.header("Status & Configuration")
st.sidebar.metric("WhatsApp Sent (Session)", st.session_state.whatsapp_sent_counter)
st.sidebar.subheader("Auto-Check Settings")
st.sidebar.subheader("Database Actions")
if st.sidebar.button("Send Pending Messages"):
    send_whatsapp_from_db()
pending_count = get_pending_message_count()
st.sidebar.metric("Pending WhatsApp Messages", pending_count)
auto_check = st.sidebar.checkbox("Enable Auto-Check", value=st.session_state["auto_check_enabled"])
interval = st.sidebar.slider("Check Interval (min)", 5, 120, st.session_state["check_interval_minutes"], 5)
st.session_state["auto_check_enabled"] = auto_check
st.session_state["check_interval_minutes"] = interval
last_check_str = st.session_state["last_check_time"].strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.text(f"Last Email Check: {last_check_str}")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📬 Email PO Summary", "📝 Manual PO + Alerts"])

with tab1:  # Email PO Summary Tab
    st.subheader("📧 Email Order Extraction")
    # Dynamic subject search input
    subject_query = st.text_input(
        "Enter subject to search in emails",
        value=st.session_state.get("email_search_query", "Purchase Order"),
        key="email_search_query"
    )
    if st.button("Check Emails Now"):
        check_and_process_emails_automatically(subject_query)

with tab2:
    st.subheader("Manual Order Entry")
    with st.expander("➕ Add Manual PO"):
        with st.form("manual_order", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                m_product_name = st.text_input("Product Name *")
                m_category = st.text_input("Category")
                m_price = st.number_input("Price *", min_value=0.0, step=0.01)
                m_quantity = st.number_input("Quantity *", min_value=1, step=1)
                m_order_date = st.date_input("Order Date *", value=datetime.now())
                m_order_time = st.time_input("Order Time *", value=datetime.now().time())
                m_order_status = st.selectbox("Order Status", ["Pending", "Processing", "Shipped", "Delivered"])
            with col2:
                m_delivery_date = st.date_input("Delivery Date *", value=datetime.now() + timedelta(days=7))
                m_customer_name = st.text_input("Customer Name *")
                m_customer_phone_raw = st.text_input("Customer Phone")
                m_email = st.text_input("Customer Email")
                m_address = st.text_area("Delivery Address *")
                m_payment_method = st.selectbox("Payment Method", ["COD", "Credit Card", "UPI", "Bank Transfer"])
                m_payment_status = st.selectbox("Payment Status", ["Paid", "Unpaid", "Pending"])

            if st.form_submit_button("Submit & Notify"):
                required_fields = [
                    m_product_name, m_price, m_quantity, m_order_date,
                    m_order_time, m_delivery_date, m_customer_name, m_address
                ]
                if not all(required_fields):
                    st.error("Please fill in all required fields (*)")
                else:
                    order_datetime = datetime.combine(m_order_date, m_order_time)
                    order_details = {
                        "Product Name": m_product_name,
                        "Category": m_category,
                        "Price": m_price,
                        "Quantity": m_quantity,
                        "Order Date": order_datetime.strftime("%Y-%m-%d %H:%M"),
                        "Delivery Date": m_delivery_date.strftime("%Y-%m-%d"),
                        "Customer Name": m_customer_name,
                        "Raw Customer Phone": m_customer_phone_raw,
                        "Email": m_email,
                        "Address": m_address,
                        "Payment Method": m_payment_method,
                        "Payment Status": m_payment_status,
                        "Order Status": m_order_status
                    }
                    if store_order(connect_to_db(), order_details):
                        st.success("Order details stored successfully!")
                        st.session_state.manual_order_sent = True
                        # --- WhatsApp Notification ---
                        message_lines = [
                            "New Manual PO Order:",
                            f"Product: {m_product_name}",
                            f"Category: {m_category}",
                            f"Price: ₹{m_price}",
                            f"Quantity: {m_quantity}",
                            f"Order Date: {order_datetime.strftime('%Y-%m-%d %H:%M')}",
                            f"Expected Delivery: {m_delivery_date.strftime('%Y-%m-%d')}",
                            f"Customer: {m_customer_name}",
                            f"Cust. Phone (for ref): {m_customer_phone_raw}",
                            f"Cust. Email: {m_email}",
                            f"Address: {m_address}",
                            f"Payment: {m_payment_method} ({m_payment_status})",
                            f"Status: {m_order_status}",
                        ]
                        formatted_message = "%0A".join(message_lines)
                        seller_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
                        send_whatsapp_message(formatted_message, seller_numbers)  # Send to seller team
                    else:
                        st.error("Failed to store order details.")

 # --- Contact Dictionary ---
    contact_dict = {
        "Narayan": "+91 90678 47003",
        "Rani Bhise": "+91 70702 42402",
        "Abhishek": "+91 92846 25240",
        "Damini": "+91 74993 53409",
        "Sandeep": "+91 98500 30215",
        "Chandrakant":"+91 96659 34999",
        "Vikas Kumbharkar ":"+91 92842 38738"
    }

    # --- Streamlit Checkboxes for Contacts ---
    st.subheader("Quick WhatsApp Contact")
    custom_message = st.text_area("Custom Message", "Enter your message here...")
    selected_contacts = []

    for contact_name, phone_number in contact_dict.items():
        if st.checkbox(contact_name, key=f"checkbox_{contact_name}"):
            selected_contacts.append(phone_number)

    # Send WhatsApp Message Button
    if selected_contacts:
        if st.button("Send WhatsApp Message to Selected Contacts"):
            final_message = f"{custom_message}"
            send_whatsapp_message(final_message, selected_contacts)
    else:
        st.warning("Please select at least one contact before sending a message.")

# --- Background Task (Email Checking) ---
if st.session_state["auto_check_enabled"]:
    next_check_time = st.session_state["last_check_time"] + timedelta(minutes=st.session_state["check_interval_minutes"])
    if datetime.now() >= next_check_time:
        check_and_process_emails_automatically()

# Start the scheduler in a separate thread
if not hasattr(st, 'scheduler_started'):
    schedule.every(st.session_state["check_interval_minutes"]).minutes.do(check_and_process_emails_automatically)
    threading.Thread(target=run_scheduled_tasks, daemon=True).start()
    #st.scheduler_started = True
    st.session_state["scheduler_started"] = True

