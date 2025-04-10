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
from PO_s3store import process_po_emails
import phonenumbers
import schedule
import random
import webbrowser

# Secrets for email
EMAIL = st.secrets["gmail_uname"]
PASSWORD = st.secrets["gmail_pwd"]
IMAP_SERVER = "imap.gmail.com"



SELLER_TEAM_RECIPIENTS_STR = st.secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS")

# --- Session state ---
if "whatsapp_sent_counter" not in st.session_state:
    st.session_state["whatsapp_sent_counter"] = 0
if "whatsapp_errors" not in st.session_state:
    st.session_state["whatsapp_errors"] = []
if "manual_order_sent" not in st.session_state:
    st.session_state["manual_order_sent"] = False
if "last_check_time" not in st.session_state:
    st.session_state["last_check_time"] = datetime.now()
if "auto_check_enabled" not in st.session_state:
    st.session_state["auto_check_enabled"] = True
if "check_interval_minutes" not in st.session_state:
    st.session_state["check_interval_minutes"] = 30
if "sending_in_progress" not in st.session_state:
    st.session_state["sending_in_progress"] = False

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
        phone_number = phonenumbers.parse(phone_str)
        if phonenumbers.is_valid_number(phone_number):
            return phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None

def parse_order_details(text):
    patterns = {
        "Order ID": r"Order ID:?\s*([A-Z0-9-]+)",
        "Product Name": r"Product(?: Name)?:?\s*(.+)",
        "Category": r"Category:?\s*(.+)",
        "Price": r"Price:?\s*[₹$]?\s*(\d[\d,]\.?\d)",
        "Quantity": r"Quantity:?\s*(\d+)",
        "Order Date": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
        "Delivery Date": r"(?:Expected )?Delivery(?: Date)?:?\s*(\d{4}-\d{2}-\d{2})",
        "Customer Name": r"Customer(?: Name)?:?\s*(.+)",
        "Phone": r"Phone:?\s*(\+?\d[\d\s-]{8,15}\d)",
        "Email": r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        "Address": r"Address:?\s*(.+?)",
        "Payment Method": r"Payment(?: Method)?:?\s*(COD|Cash on Delivery|Credit Card|UPI|Bank Transfer)",
        "Payment Status": r"Payment Status:?\s*(Paid|Unpaid|Pending)",
        "Order Status": r"(?:Order )?Status:?\s*(Pending|Processing|Shipped|Delivered|Cancelled)"
    }
    order_details = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        order_details[key] = match.group(1).strip().replace(",", "") if (match and key == "Price") else match.group(1).strip() if match else "Not found"

    if order_details["Address"] == "Not found" or len(order_details["Address"]) < 10:
        match_addr = re.search(r"Address:?\s*(.*?)(?:Payment Method:|Payment Status:|Order Status:|Notes:|Tracking ID:|---|$)", text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match_addr:
            order_details["Address"] = match_addr.group(1).strip()

    order_details["Raw Customer Phone"] = order_details.get("Phone", "Not found")
    return order_details

def fetch_email_pdfs():
    pdf_files_with_info = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN SUBJECT "PO Dump")')
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
        for phone in seller_team_recipients_str.split(','):
            formatted_seller = format_phone_number(phone.strip())
            if formatted_seller:
                recipients.add(formatted_seller)
    return recipients

def send_whatsapp_bulk(recipients, message):
    if st.session_state["sending_in_progress"]:
        return False

    st.session_state["sending_in_progress"] = True
    status_container = st.empty()
    try:
        recipient_list = list(recipients)
        total = len(recipient_list)
        success_count = 0

        webbrowser.open("https://web.whatsapp.com")
        time.sleep(15)  # Wait for WhatsApp to load

        for idx, phone_number in enumerate(recipient_list):
            status_container.info(f"Sending message {idx+1}/{total}...")
            clean_phone = phone_number.replace('+', '').replace(' ', '')
            webbrowser.open(f"https://web.whatsapp.com/send?phone={clean_phone}")
            time.sleep(10)
            pyautogui.write(message)
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(3)
            success_count += 1
            st.session_state["whatsapp_sent_counter"] += 1
            time.sleep(random.uniform(2, 4))

        status_container.success(f"Sent {success_count}/{total} messages")
        time.sleep(2)
        pyautogui.hotkey('ctrl', 'w')
        return success_count > 0
    except Exception as e:
        st.error(f"Bulk sending error: {str(e)}")
        return False
    finally:
        st.session_state["sending_in_progress"] = False

def check_and_process_emails_automatically():
    if st.session_state["sending_in_progress"]:
        return

    status_container = st.empty()
    status_container.info("Checking for new PO emails...")
    pdf_files_info = fetch_email_pdfs()

    if pdf_files_info:
        seller_team_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
        if not seller_team_numbers:
            status_container.error("No valid Seller Team numbers configured.")
            return

        processed_count = 0
        with st.spinner("Processing emails..."):
            for pdf_info in pdf_files_info:
                try:
                    pdf_file = io.BytesIO(pdf_info["data"])
                    text = extract_text_from_pdf(pdf_file)
                    if text:
                        order_details = parse_order_details(text)
                        message = f"""New PO from Email:
 Product: {order_details.get('Product Name', 'Not found')}
 Price: ₹{order_details.get('Price', 'Not found')}
 Quantity: {order_details.get('Quantity', 'Not found')}
 Order Date: {order_details.get('Order Date', 'Not found')}
 Delivery Date: {order_details.get('Delivery Date', 'Not found')}
 Customer: {order_details.get('Customer Name', 'Not found')}
 Cust. Phone: {order_details.get('Raw Customer Phone', 'Not found')}
 Address: {order_details.get('Address', 'Not found')}
 Payment: {order_details.get('Payment Method', 'Not found')} ({order_details.get('Payment Status', 'Not found')})
 Status: {order_details.get('Order Status', 'Not found')}
 """
                        if send_whatsapp_bulk(seller_team_numbers, message):
                            processed_count += 1
                except Exception as e:
                    st.error(f"Error processing {pdf_info['filename']}: {str(e)}")
            status_container.success(f"Processed {processed_count} email(s)")
    else:
        status_container.info("No new PO emails found")
    st.session_state["last_check_time"] = datetime.now()

def run_scheduled_tasks():
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Streamlit App Layout ---
st.set_page_config(page_title="PO Order Manager", layout="wide")
st.title("🛒 PO Order Management Dashboard")

# --- Sidebar ---
st.sidebar.header("Status & Configuration")
st.sidebar.metric("WhatsApp Sent (Session)", st.session_state.whatsapp_sent_counter)
st.sidebar.subheader("Auto-Check Settings")
auto_check = st.sidebar.checkbox("Enable Auto-Check", value=st.session_state["auto_check_enabled"])
interval = st.sidebar.slider("Check Interval (min)", 5, 120, st.session_state["check_interval_minutes"], 5)
st.session_state["auto_check_enabled"] = auto_check
st.session_state["check_interval_minutes"] = interval

if st.session_state["last_check_time"]:
    st.sidebar.info(f"Last Check: {st.session_state['last_check_time'].strftime('%H:%M:%S')}")
if st.session_state["sending_in_progress"]:
    st.sidebar.warning("Sending in Progress...")
if st.session_state["whatsapp_errors"]:
    st.sidebar.error("Errors Occurred")
    with st.sidebar.expander("View Errors"):
        for error in st.session_state["whatsapp_errors"][-10:]:
            st.write(f"- {error}")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📊 Email PO Summary", "📝 Manual PO + Alerts"])

with tab1:
    st.subheader("📬 Incoming PO Emails")
    extracted_count_container = st.empty()
    with st.spinner("Checking emails..."):
        try:
            extracted_count = process_po_emails()
            extracted_count_container.metric("New PO PDFs Processed", extracted_count if isinstance(extracted_count, int) else 0)
        except Exception as e:
            st.error(f"Error: {e}")
            extracted_count_container.metric("New PO PDFs Processed", 0)

    if st.button("Check Emails Now"):
        check_and_process_emails_automatically()

with tab2:
    st.subheader("Manual Order & Alerts")
    with st.expander("➕ Add Manual PO", expanded=False):
        with st.form("manual_order", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                m_product_name = st.text_input("Product Name *")
                m_category = st.text_input("Category")
                m_price = st.number_input("Price *", min_value=0.0, step=0.01, format="%.2f")
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
                required_fields = [m_product_name, m_price, m_quantity, m_order_date, m_order_time, m_delivery_date, m_customer_name, m_address]
                if not all(required_fields):
                    st.error("Fill all required fields (*)")
                else:
                    order_datetime = datetime.combine(m_order_date, m_order_time)
                    seller_team_numbers = get_seller_team_recipients(SELLER_TEAM_RECIPIENTS_STR)
                    if not seller_team_numbers:
                        st.error("No valid Seller Team numbers")
                    else:
                        message = f"""New Manual PO:
 Product: {m_product_name}
 Category: {m_category or 'N/A'}
 Price: ₹{m_price:.2f}
 Quantity: {m_quantity}
 Order Date: {order_datetime.strftime('%Y-%m-%d %H:%M')}
 Delivery: {m_delivery_date.strftime('%Y-%m-%d')}
 Customer: {m_customer_name}
 Phone: {m_customer_phone_raw or 'N/A'}
 Email: {m_email or 'N/A'}
 Address: {m_address}
 Payment: {m_payment_method} ({m_payment_status})
 Status: {m_order_status}
"""
                        send_whatsapp_bulk(seller_team_numbers, message)
                        st.success("Order submitted and notifications sent")

# --- Scheduler ---
if st.session_state["auto_check_enabled"]:
    schedule.clear()
    schedule.every(st.session_state["check_interval_minutes"]).minutes.do(check_and_process_emails_automatically)
    check_and_process_emails_automatically()
    threading.Thread(target=run_scheduled_tasks, daemon=True).start()

# --- End of Streamlit App ---