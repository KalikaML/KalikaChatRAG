import os
import io
import re
import time
import imaplib
import email
from email.header import decode_header
import PyPDF2
import phonenumbers
import psycopg2
from datetime import datetime, timedelta, date
import logging
import psycopg2.extras
import webbrowser  # For WhatsApp Web - not ideal for API
import pyautogui  # For WhatsApp Web - not ideal for API
import pywhatkit  # For WhatsApp Web - not ideal for API
import random
import uvicorn  # For running the API server
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, status
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Union, Dict, Any
from dotenv import load_dotenv  # To load .env file for secrets
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.security import OAuth2PasswordBearer
import jwt
import bcrypt

# --- Configuration Setup (Replaces st.secrets) ---
load_dotenv()  # Load variables from .env file if it exists

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")
# Email Secrets
EMAIL_USER = os.getenv("GMAIL_UNAME")
EMAIL_PASSWORD = os.getenv("GMAIL_PWD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
SELLER_TEAM_RECIPIENTS_STR = os.getenv("ADDITIONAL_WHATSAPP_RECIPIENTS", "")  # Comma-separated string

# Database Credentials
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "po_orders")
DB_USER = os.getenv("DB_USER", "po_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postdb123")
DB_PORT = int(os.getenv("DB_PORT", 5432))


# --- Pydantic Models (for Request/Response Validation) ---

class OrderBase(BaseModel):
    product_name: str = Field(..., example="Laptop XYZ")
    category: Optional[str] = Field(None, example="Electronics")
    price: float = Field(..., gt=0, example=75000.00)
    quantity: int = Field(..., gt=0, example=1)
    order_date: datetime = Field(default_factory=datetime.now)
    delivery_date: date = Field(..., example="2025-05-20")
    customer_name: str = Field(..., example="John Doe")
    customer_phone: Optional[str] = Field(None, example="+919876543210")  # Raw phone, will be formatted
    email: Optional[EmailStr] = Field(None, example="john.doe@example.com")
    address: str = Field(..., example="123 Main St, Anytown")
    payment_method: Optional[str] = Field("COD", example="Credit Card")
    payment_status: Optional[str] = Field("Pending", example="Paid")
    order_status: Optional[str] = Field("Pending", example="Confirmed")
    source_email_subject: Optional[str] = None
    source_email_sender: Optional[str] = None


class OrderCreate(OrderBase):
    pass


class Order(OrderBase):
    id: int
    message_sent: bool = False

    class Config:
        from_attributes = True


class WhatsAppMessageRequest(BaseModel):
    recipient_numbers: List[str] = Field(..., example=["+919000000000", "+919111111111"])
    message: str = Field(..., example="Your order has been shipped!")


class QuickWhatsAppRequest(BaseModel):
    contact_names: List[str]  # Names from a predefined contact_dict
    message: str


class EmailProcessRequest(BaseModel):
    subject_query: str = Field("Purchase Order", example="New Order Confirmation")
    only_recent_days: Optional[int] = Field(None, example=7)
    mark_as_read_after_extraction: bool = True


class Metrics(BaseModel):
    total_orders: int
    total_revenue_paid: float
    avg_order_value: float


class StatusDistribution(BaseModel):
    status: str
    count: int


class SalesByCategory(BaseModel):
    category: str
    total_sales: float


class UserSignup(BaseModel):
    email: EmailStr
    password: str


class UserSignin(BaseModel):
    email: EmailStr
    password: str


# --- Database Connection and Operations ---

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            cursor_factory=psycopg2.extras.DictCursor
        )
        logger.info("Database connection successful.")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection error: {e}")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM login WHERE email = %s", (email,))
            user = cur.fetchone()
            if user is None:
                raise credentials_exception
            return {"email": user["email"]}
    finally:
        conn.close()


def create_orders_table_if_not_exists():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    product_name TEXT NOT NULL,
                    category TEXT,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_date TIMESTAMP NOT NULL,
                    delivery_date DATE NOT NULL,
                    customer_name TEXT NOT NULL,
                    customer_phone TEXT,
                    email TEXT,
                    address TEXT NOT NULL,
                    payment_method TEXT,
                    payment_status TEXT,
                    order_status TEXT,
                    message_sent BOOLEAN DEFAULT FALSE,
                    source_email_subject TEXT,
                    source_email_sender TEXT
                );
            """)
        conn.commit()
        logger.info("Orders table checked/created successfully.")
    except psycopg2.Error as e:
        logger.error(f"Error creating orders table: {e}")
        if conn: conn.rollback()
    except Exception as e:
        logger.error(f"An unexpected error occurred during table creation: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()


# Call it once at startup (FastAPI specific way)
# @app.on_event("startup")
# async def startup_event():
#     create_orders_table_if_not_exists()
# This function can be called manually or by the API on its first run,
# or ensured by deployment scripts. For simplicity, we'll assume it's created.

def store_order_in_db(order_details: OrderCreate, conn: Optional[psycopg2.extensions.connection] = None) -> Optional[
    int]:
    """Stores an order in the database and returns the new order ID."""
    close_conn_here = False
    if conn is None:
        conn = get_db_connection()
        close_conn_here = True

    try:
        with conn.cursor() as cursor:
            # Format phone number before storing
            raw_phone = order_details.customer_phone
            formatted_phone = format_phone_number(raw_phone) if raw_phone else None

            columns = [
                "product_name", "category", "price", "quantity", "order_date", "delivery_date",
                "customer_name", "customer_phone", "email", "address", "payment_method",
                "payment_status", "order_status", "message_sent",
                "source_email_subject", "source_email_sender"
            ]
            values = [
                order_details.product_name, order_details.category,
                order_details.price, order_details.quantity,
                order_details.order_date, order_details.delivery_date,
                order_details.customer_name, formatted_phone,  # Store formatted phone
                order_details.email, order_details.address,
                order_details.payment_method, order_details.payment_status,
                order_details.order_status, False,  # message_sent default
                order_details.source_email_subject,
                order_details.source_email_sender
            ]

            sql = f"INSERT INTO orders ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) RETURNING id"
            cursor.execute(sql, tuple(values))
            order_id = cursor.fetchone()
            if order_id:
                order_id = order_id[0]
        conn.commit()
        logger.info(f"Order for {order_details.product_name} stored successfully with ID: {order_id}.")
        return order_id
    except psycopg2.Error as e:
        if conn: conn.rollback()
        logger.error(f"Database error storing order: {e}")
        raise HTTPException(status_code=500, detail=f"Database error storing order: {e}")
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"Store order error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store order: {e}")
    finally:
        if conn and close_conn_here:
            conn.close()


# --- Helper Functions (Adapted from original) ---

def extract_text_from_pdf(pdf_file_bytes: bytes) -> str:
    try:
        pdf_file_like_object = io.BytesIO(pdf_file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file_like_object)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def format_phone_number(phone_str: str) -> Optional[str]:
    if not phone_str or not isinstance(phone_str, str): return None
    try:
        # Ensure country code is present, default to IN if not an international number
        if not phone_str.startswith('+') and not phone_str.startswith('00'):
            # Check if it's a valid length for an Indian number without country code
            cleaned_phone = re.sub(r'\D', '', phone_str)
            if 10 <= len(cleaned_phone) <= 12:  # e.g. 9876543210 or 09876543210
                p = phonenumbers.parse(phone_str, "IN")
            else:  # If not clearly Indian, try generic parsing
                p = phonenumbers.parse(phone_str, None)
        else:
            p = phonenumbers.parse(phone_str, None)  # None for auto-detection if + is present

        if phonenumbers.is_valid_number(p):
            return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
        else:
            logger.warning(f"Could not format invalid phone number: {phone_str}")
            return None  # Return None for invalid numbers to avoid sending to wrong numbers
    except phonenumbers.phonenumberutil.NumberParseException:
        logger.warning(f"Phone number parsing error for: {phone_str}")
        return None


def parse_order_details_from_text(text: str) -> Dict[str, Any]:
    # This is a simplified version for brevity. The original regexes should be used.
    patterns = {
        "Product Name": r"Product(?: Name)?:?\s*(.+?)(?:\nCategory:|\nPrice:|\nQuantity:|$)",
        "Category": r"Category:?\s*(.+?)(?:\nPrice:|\nQuantity:|\nOrder Date:|$)",
        "Price": r"Price:?\s*[₹$]?\s*(\d[\d,]*\.?\d*)",
        "Quantity": r"Quantity:?\s*(\d+)",
        "Order Date": r"Order Date:?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
        "Delivery Date": r"(?:Expected )?Delivery(?: Date)?:?\s*(\d{4}-\d{2}-\d{2})",
        "Customer Name": r"Customer(?: Name)?:?\s*(.+?)(?:\nPhone:|\nEmail:|\nAddress:|$)",
        "Phone": r"Phone:?\s*(\+?\d[\d\s-]{8,15}\d)",
        "Email": r"Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        "Address": r"Address:?\s*(.+?)(?:Payment Method:|Payment Status:|Order Status:|Notes:|Tracking ID:|Order ID:|Customer Name:|Phone:|Email:|---|$)",
        "Payment Method": r"Payment(?: Method)?:?\s*(COD|Cash on Delivery|Credit Card|UPI|Bank Transfer|Other)",
        "Payment Status": r"Payment Status:?\s*(Paid|Unpaid|Pending)",
        "Order Status": r"(?:Order )?Status:?\s*(Pending|Processing|Confirmed|Shipped|Delivered|Cancelled)",
    }
    order_details = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | (re.DOTALL if key == "Address" else 0))
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
                        order_details[key] = None  # Or raise error / log
            elif key == "Delivery Date" and value:
                try:
                    order_details[key] = datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    order_details[key] = None
            elif key == "Address":
                address_candidate = value.strip().replace('\n', ' ')
                stop_keywords = ["Payment Method:", "Payment Status:", "Order Status:", "Notes:", "Tracking ID:"]
                for stop_word in stop_keywords:
                    if stop_word.lower() in address_candidate.lower():
                        address_candidate = address_candidate.lower().split(stop_word.lower())[0].strip()
                order_details["Address"] = address_candidate.title()  # Title case for consistency
            else:
                order_details[key] = value
        else:
            order_details[key] = None if key not in ["Category", "Email"] else ""  # Default for optional fields

    # Special handling for phone: store raw for reference, format it elsewhere.
    order_details["Raw Customer Phone"] = order_details.pop("Phone", None)
    return order_details


# --- WhatsApp Sending Functions (NOTE: Server-side limitations) ---

def send_whatsapp_message_pywhatkit_api(message: str, recipient_numbers_list: List[str]) -> Dict:
    results = {"success": [], "failed": []}
    if not all([EMAIL_USER, EMAIL_PASSWORD, IMAP_SERVER]):  # Check if running in an env where these are available
        logger.warning(
            "PyWhatKit WhatsApp sending is likely to fail in a non-interactive environment or without display.")

    for recipient in recipient_numbers_list:
        formatted_recipient = format_phone_number(recipient)
        if not formatted_recipient:
            logger.warning(f"Skipping invalid phone for pywhatkit: {recipient}")
            results["failed"].append({"recipient": recipient, "error": "Invalid phone number format"})
            continue
        try:
            current_time = datetime.now()
            send_hour = current_time.hour
            send_minute = (current_time.minute + 1) % 60  # Send in 1 min
            if send_minute < current_time.minute: send_hour = (send_hour + 1) % 24

            logger.info(f"PyWhatKit: Queuing to {formatted_recipient} at {send_hour:02d}:{send_minute:02d}...")
            # This will open a browser tab, highly problematic for APIs
            pywhatkit.sendwhatmsg(formatted_recipient, message, send_hour, send_minute, wait_time=15, tab_close=True,
                                  close_time=3)
            logger.info(f"PyWhatKit: Message queued for {formatted_recipient}!")
            results["success"].append(formatted_recipient)
            time.sleep(5)  # Stagger messages
        except Exception as e:
            logger.error(f"PyWhatKit Error for {recipient}: {e}")
            results["failed"].append({"recipient": recipient, "error": str(e)})
    return results


def send_whatsapp_message_web_api(message: str, recipient_numbers_list: List[str]) -> Dict:
    results = {"success": [], "failed": []}
    logger.warning(
        "Web WhatsApp sending via webbrowser/pyautogui is not suitable for production APIs and will likely fail on a server.")

    try:
        webbrowser.open("https://web.whatsapp.com")
        logger.info("Opened WhatsApp Web. Manual QR scan might be needed if first time. Waiting 25s...")
        time.sleep(25)
    except Exception as e:
        logger.error(f"Failed to open WhatsApp Web: {e}")
        for recipient in recipient_numbers_list:
            results["failed"].append({"recipient": recipient, "error": f"Failed to open WhatsApp Web: {e}"})
        return results

    for recipient in recipient_numbers_list:
        formatted_recipient = format_phone_number(recipient)
        if not formatted_recipient:
            logger.warning(f"Skipping invalid phone for web: {recipient}")
            results["failed"].append({"recipient": recipient, "error": "Invalid phone number format"})
            continue
        try:
            import urllib.parse
            encoded_message = urllib.parse.quote(message)
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_recipient}&text={encoded_message}"
            webbrowser.open(whatsapp_url)
            logger.info(f"Web: Opened chat for {formatted_recipient}. Waiting 15s...")
            time.sleep(15)
            pyautogui.press("enter")  # This requires GUI
            logger.info(f"Web: 'Enter' pressed for {formatted_recipient}.")
            results["success"].append(formatted_recipient)
            time.sleep(random.uniform(4, 7))
        except Exception as e:
            logger.error(f"Web WhatsApp Error for {formatted_recipient}: {e}")
            results["failed"].append({"recipient": formatted_recipient, "error": str(e)})
    return results


# Default send function (choose one, or make it configurable)
# For an API, neither is ideal. This is just for conversion demonstration.
# A proper API would use WhatsApp Business API.
active_whatsapp_sender = send_whatsapp_message_pywhatkit_api  # or send_whatsapp_message_web_api


# --- Email Fetching and Processing Service ---

def fetch_email_pdfs_service(subject_query: str, only_recent_days: Optional[int] = None, mark_as_read: bool = True) -> \
List[Dict[str, Any]]:
    if not all([EMAIL_USER, EMAIL_PASSWORD, IMAP_SERVER]):
        logger.error("Email credentials not configured. Cannot fetch emails.")
        raise HTTPException(status_code=500, detail="Email service not configured.")

    pdf_files_with_info = []
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select("inbox")

        search_parts = ['(UNSEEN'] if mark_as_read else ['(']  # Option to search all if not marking as read
        search_parts.append(f'SUBJECT "{subject_query}"')
        if only_recent_days and isinstance(only_recent_days, int) and only_recent_days > 0:
            date_since = (datetime.now() - timedelta(days=only_recent_days)).strftime("%d-%b-%Y")
            search_parts.append(f'SINCE "{date_since}"')
        search_parts.append(')')  # Closing parenthesis for UNSEEN or general search
        search_criteria = ' '.join(search_parts)

        logger.info(f"Searching emails with criteria: {search_criteria}")
        status, messages = mail.search(None, search_criteria)

        if status != "OK":
            logger.error(f"IMAP search failed with status: {status}")
            return []
        if not messages[0]:
            logger.info(f"No emails found matching criteria: {search_criteria}")
            return []

        mail_ids = messages[0].split()
        logger.info(f"Found {len(mail_ids)} email(s) to process.")

        for mail_id in reversed(mail_ids):  # Process newest first usually
            fetch_status, msg_data = mail.fetch(mail_id, '(RFC822)')
            if fetch_status != "OK":
                logger.warning(f"Failed to fetch email ID {mail_id.decode()}. Status: {fetch_status}")
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            def decode_mail_header(header_value):
                if header_value is None: return ""
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
                            "data": pdf_data, "filename": filename, "sender": sender,
                            "subject": subject, "email_id": mail_id.decode()
                        })
                        pdf_extracted_this_email = True
                        logger.info(f"Extracted PDF: {filename} from email ID {mail_id.decode()}")

            if pdf_extracted_this_email and mark_as_read:
                try:
                    store_status, _ = mail.store(mail_id, '+FLAGS', '\\Seen')
                    if store_status == "OK":
                        logger.info(f"Successfully marked email ID {mail_id.decode()} as SEEN.")
                    else:
                        logger.warning(f"Failed to mark email ID {mail_id.decode()} as SEEN. Status: {store_status}")
                except Exception as e_store:
                    logger.error(f"Error marking email ID {mail_id.decode()} as SEEN: {e_store}")
        return pdf_files_with_info
    except imaplib.IMAP4.abort as e_abort:
        logger.error(f"IMAP connection aborted: {e_abort}. Try again later.")
        raise HTTPException(status_code=503, detail=f"IMAP connection error: {e_abort}")
    except Exception as e:
        logger.error(f"An error occurred while fetching emails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching emails: {e}")
    finally:
        if mail:
            try:
                mail.close()
                mail.logout()
                logger.info("Logged out from IMAP server.")
            except Exception as e_logout:
                logger.error(f"Error during IMAP logout/close: {e_logout}")


def process_and_store_email_orders_service(subject_keywords: List[str], only_recent_days: Optional[int] = None,
                                           mark_as_read: bool = True):
    logger.info(f"Starting email order processing for subjects: {subject_keywords}")
    pdf_files_info = []
    text_files_info = []
    mail = None
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select("inbox")

        for subject_query in subject_keywords:
            search_parts = ['(UNSEEN'] if mark_as_read else ['(']
            search_parts.append(f'SUBJECT "{subject_query}"')
            if only_recent_days and isinstance(only_recent_days, int) and only_recent_days > 0:
                date_since = (datetime.now() - timedelta(days=only_recent_days)).strftime("%d-%b-%Y")
                search_parts.append(f'SINCE "{date_since}"')
            search_parts.append(')')
            search_criteria = ' '.join(search_parts)

            logger.info(f"Searching emails with criteria: {search_criteria}")
            status, messages = mail.search(None, search_criteria)

            if status != "OK":
                logger.error(f"IMAP search failed for '{subject_query}' with status: {status}")
                continue
            if not messages[0]:
                logger.info(f"No emails found for subject: {subject_query}")
                continue

            mail_ids = messages[0].split()
            logger.info(f"Found {len(mail_ids)} email(s) for subject: {subject_query}")

            for mail_id in reversed(mail_ids):
                fetch_status, msg_data = mail.fetch(mail_id, '(RFC822)')
                if fetch_status != "OK":
                    logger.warning(f"Failed to fetch email ID {mail_id.decode()}. Status: {fetch_status}")
                    continue

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                def decode_mail_header(header_value):
                    if header_value is None: return ""
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
                pdf_extracted = False
                body_text = ""

                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "application/pdf" and part.get('Content-Disposition'):
                        filename = decode_mail_header(part.get_filename()) or f'untitled_{mail_id.decode()}.pdf'
                        pdf_data = part.get_payload(decode=True)
                        if pdf_data:
                            pdf_files_info.append({
                                "data": pdf_data, "filename": filename, "sender": sender,
                                "subject": subject, "email_id": mail_id.decode()
                            })
                            pdf_extracted = True
                            logger.info(f"Extracted PDF: {filename} from email ID {mail_id.decode()}")
                    elif content_type == "text/plain" and not pdf_extracted:
                        try:
                            body_text = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        except Exception as e:
                            logger.warning(f"Error decoding text body for email ID {mail_id.decode()}: {e}")
                            body_text = ""

                if body_text and not pdf_extracted:
                    text_files_info.append({
                        "text": body_text, "filename": f"email_body_{mail_id.decode()}.txt",
                        "sender": sender, "subject": subject, "email_id": mail_id.decode()
                    })
                    logger.info(f"Extracted text body from email ID {mail_id.decode()}")

                if (pdf_extracted or body_text) and mark_as_read:
                    try:
                        store_status, _ = mail.store(mail_id, '+FLAGS', '\\Seen')
                        if store_status == "OK":
                            logger.info(f"Marked email ID {mail_id.decode()} as SEEN.")
                        else:
                            logger.warning(
                                f"Failed to mark email ID {mail_id.decode()} as SEEN. Status: {store_status}")
                    except Exception as e_store:
                        logger.error(f"Error marking email ID {mail_id.decode()} as SEEN: {e_store}")

        processed_count = 0
        failed_store_count = 0
        conn = get_db_connection()
        results = []

        # Process PDFs
        for pdf_info in pdf_files_info:
            logger.info(f"Processing PDF: {pdf_info['filename']} from {pdf_info['sender']}")
            text = extract_text_from_pdf(pdf_info["data"])
            if text:
                parsed_details = parse_order_details_from_text(text)
                try:
                    if not parsed_details.get("Product Name") or parsed_details.get("Price") is None:
                        logger.warning(f"Essential details missing from {pdf_info['filename']}. Skipping.")
                        failed_store_count += 1
                        results.append({"filename": pdf_info['filename'], "status": "skipped",
                                        "reason": "Essential details missing"})
                        continue

                    order_date_from_pdf = parsed_details.get("Order Date")
                    delivery_date_from_pdf = parsed_details.get("Delivery Date")

                    order_create_data = OrderCreate(
                        product_name=parsed_details.get("Product Name", "N/A"),
                        category=parsed_details.get("Category"),
                        price=float(parsed_details.get("Price", 0.0)),
                        quantity=int(parsed_details.get("Quantity", 1)),
                        order_date=order_date_from_pdf if isinstance(order_date_from_pdf, datetime) else datetime.now(),
                        delivery_date=delivery_date_from_pdf if isinstance(delivery_date_from_pdf, date) else (
                                    datetime.now() + timedelta(days=7)).date(),
                        customer_name=parsed_details.get("Customer Name", "N/A"),
                        customer_phone=parsed_details.get("Raw Customer Phone"),
                        email=parsed_details.get("Email"),
                        address=parsed_details.get("Address", "N/A"),
                        payment_method=parsed_details.get("Payment Method", "COD"),
                        payment_status=parsed_details.get("Payment Status", "Pending"),
                        order_status=parsed_details.get("Order Status", "Pending"),
                        source_email_subject=pdf_info["subject"],
                        source_email_sender=pdf_info["sender"]
                    )

                    order_id = store_order_in_db(order_create_data, conn)
                    if order_id:
                        processed_count += 1
                        logger.info(f"Stored order from '{pdf_info['filename']}' with ID {order_id}")
                        results.append({"filename": pdf_info['filename'], "status": "success", "order_id": order_id})
                    else:
                        failed_store_count += 1
                        results.append({"filename": pdf_info['filename'], "status": "failed_to_store_in_db"})
                except Exception as e_val:
                    logger.error(f"Error processing PDF {pdf_info['filename']}: {e_val}")
                    failed_store_count += 1
                    results.append(
                        {"filename": pdf_info['filename'], "status": "failed-processing", "reason": str(e_val)})
            else:
                logger.warning(f"No text extracted from PDF: {pdf_info['filename']}")
                failed_store_count += 1
                results.append({"filename": pdf_info['filename'], "status": "no_text_extracted"})

        # Process email bodies
        for text_info in text_files_info:
            logger.info(f"Processing email body: {text_info['filename']} from {text_info['sender']}")
            text = text_info["text"]
            if text:
                parsed_details = parse_order_details_from_text(text)
                try:
                    if not parsed_details.get("Product Name") or parsed_details.get("Price") is None:
                        logger.warning(f"Essential details missing from {text_info['filename']}. Skipping.")
                        failed_store_count += 1
                        results.append({"filename": text_info['filename'], "status": "skipped",
                                        "reason": "Essential details missing"})
                        continue

                    order_date_from_text = parsed_details.get("Order Date")
                    delivery_date_from_text = parsed_details.get("Delivery Date")

                    order_create_data = OrderCreate(
                        product_name=parsed_details.get("Product Name", "N/A"),
                        category=parsed_details.get("Category"),
                        price=float(parsed_details.get("Price", 0.0)),
                        quantity=int(parsed_details.get("Quantity", 1)),
                        order_date=order_date_from_text if isinstance(order_date_from_text,
                                                                      datetime) else datetime.now(),
                        delivery_date=delivery_date_from_text if isinstance(delivery_date_from_text, date) else (
                                    datetime.now() + timedelta(days=7)).date(),
                        customer_name=parsed_details.get("Customer Name", "N/A"),
                        customer_phone=parsed_details.get("Raw Customer Phone"),
                        email=parsed_details.get("Email"),
                        address=parsed_details.get("Address", "N/A"),
                        payment_method=parsed_details.get("Payment Method", "COD"),
                        payment_status=parsed_details.get("Payment Status", "Pending"),
                        order_status=parsed_details.get("Order Status", "Pending"),
                        source_email_subject=text_info["subject"],
                        source_email_sender=text_info["sender"]
                    )

                    order_id = store_order_in_db(order_create_data, conn)
                    if order_id:
                        processed_count += 1
                        logger.info(f"Stored order from email body '{text_info['filename']}' with ID {order_id}")
                        results.append({"filename": text_info['filename'], "status": "success", "order_id": order_id})
                    else:
                        failed_store_count += 1
                        results.append({"filename": text_info['filename'], "status": "failed_to_store_in_db"})
                except Exception as e_val:
                    logger.error(f"Error processing email body {text_info['filename']}: {e_val}")
                    failed_store_count += 1
                    results.append(
                        {"filename": text_info['filename'], "status": "failed_processing", "reason": str(e_val)})
            else:
                logger.warning(f"No text in email body: {text_info['filename']}")
                failed_store_count += 1
                results.append({"filename": text_info['filename'], "status": "no_text_extracted"})

        conn.close()
        summary_msg = f"Processed {len(pdf_files_info) + len(text_files_info)} item(s). Successfully stored {processed_count} order(s). Failed to process/store {failed_store_count} item(s)."
        logger.info(summary_msg)
        return {"message": summary_msg, "processed_count": processed_count, "failed_count": failed_store_count,
                "details": results}
    except imaplib.IMAP4.abort as e_abort:
        logger.error(f"IMAP connection aborted: {e_abort}")
        raise HTTPException(status_code=503, detail=f"IMAP connection error: {e_abort}")
    except Exception as e:
        logger.error(f"Error in email processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing emails: {e}")
    finally:
        if mail:
            try:
                mail.close()
                mail.logout()
                logger.info("Logged out from IMAP server.")
            except Exception as e_logout:
                logger.error(f"Error during IMAP logout/close: {e_logout}")
        if 'conn' in locals() and conn:
            conn.close()


# --- Utility for Seller Team Recipients ---
def get_seller_team_recipients() -> List[str]:
    recipients = set()
    if SELLER_TEAM_RECIPIENTS_STR:
        for phone in SELLER_TEAM_RECIPIENTS_STR.split(","):
            fmt_seller = format_phone_number(phone.strip())
            if fmt_seller:
                recipients.add(fmt_seller)
            else:
                logger.warning(f"Invalid seller phone in config: {phone.strip()}")
    return list(recipients)


# Scheduler Setup
scheduler = BackgroundScheduler()


def scheduled_email_processing():
    subject_keywords = [
        "PO released // Consumable items",
        "PO copy",
        "import po",
        "RFQ-Polybag",
        "PFA PO",
        "Purchase Order FOR",
        "Purchase Order_"
    ]
    try:
        result = process_and_store_email_orders_service(
            subject_keywords=subject_keywords,
            only_recent_days=7,  # Look at emails from the last 7 days
            mark_as_read=True
        )
        logger.info(f"Scheduled email processing completed: {result['message']}")
    except Exception as e:
        logger.error(f"Scheduled email processing failed: {e}")


# Add job to scheduler (every 2 hours = 7200 seconds)
scheduler.add_job(scheduled_email_processing, 'interval', seconds=7200, id='email_processing_job')
scheduler.start()

# Ensure scheduler shuts down gracefully
import atexit

atexit.register(lambda: scheduler.shutdown())
# --- FastAPI App Instance ---
app = FastAPI(title="PO Order Management API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---

@app.on_event("startup")
async def on_startup():
    create_orders_table_if_not_exists()
    logger.info("API Started. Database table checked/created.")


@app.post("/orders/", response_model=Order, status_code=201, summary="Create a new order manually")
async def create_manual_order(order: OrderCreate, background_tasks: BackgroundTasks,
                              current_user: dict = Depends(get_current_user)):
    """
    Creates a new order in the database.
    Optionally sends a WhatsApp notification to the seller team.
    """
    conn = get_db_connection()
    try:
        order_id = store_order_in_db(order, conn)
        if not order_id:  # Should be handled by HTTPException in store_order_in_db, but as a fallback
            raise HTTPException(status_code=500, detail="Failed to store order, no ID returned.")

        # Fetch the created order to return it
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
            created_order_dict = cursor.fetchone()

        if not created_order_dict:
            raise HTTPException(status_code=404, detail=f"Order with ID {order_id} not found after creation.")

        # Background task for WhatsApp notification
        seller_numbers = get_seller_team_recipients()
        if seller_numbers:
            message_lines = [
                "📦 *New Manual PO Order!* 📦",
                f"Order ID: API-{order_id}",
                f" Product: {order.product_name}",
                f" Category: {order.category or '-'}",
                f" Price: ₹{order.price:.2f}",
                f" Quantity: {order.quantity}",
                f" Order Date: {order.order_date.strftime('%Y-%m-%d %H:%M') if order.order_date else '-'}",
                f" Delivery Date: {order.delivery_date.strftime('%Y-%m-%d') if order.delivery_date else '-'}",
                f" Customer: {order.customer_name}",
                f" Phone: {order.customer_phone or '-'}",
                f" Email: {order.email or '-'}",
                f" Address: {order.address or '-'}",
                f" Payment Method: {order.payment_method or '-'}",
                f" Payment Status: {order.payment_status or '-'}",
                f" Order Status: {order.order_status or '-'}"
            ]
            formatted_message = "\n".join(message_lines)
            background_tasks.add_task(active_whatsapp_sender, formatted_message, seller_numbers)
            logger.info(f"WhatsApp notification for order API-{order_id} queued for seller team.")

            # Update message_sent status in DB (can also be done in the background task for robustness)
            try:
                with conn.cursor() as cursor_update:
                    cursor_update.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (order_id,))
                conn.commit()
                created_order_dict['message_sent'] = True  # Reflect in response
                logger.info(f"Order API-{order_id} marked as message_sent=TRUE in DB.")
            except psycopg2.Error as e_upd:
                conn.rollback()
                logger.error(f"Failed to update message_sent for order API-{order_id}: {e_upd}")
                # The main order creation is still successful.
        else:
            logger.warning("No seller WhatsApp numbers configured. Notification not sent for manual order.")

        return Order.model_validate(created_order_dict)  # Convert dict to Pydantic model

    except HTTPException:  # Re-raise HTTPExceptions from called functions
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating manual order: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if conn: conn.close()


@app.get("/orders/", response_model=List[Order], summary="Get a list of orders")
async def get_orders(
        skip: int = 0,
        limit: int = Query(10, ge=1, le=100),  # Default 10, min 1, max 100
        order_status: Optional[str] = Query(None, example="Pending"),
        customer_name: Optional[str] = Query(None, example="John Doe"),
        order_date_from: Optional[date] = Query(None),
        order_date_to: Optional[date] = Query(None), current_user: dict = Depends(get_current_user)
):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:  # Use DictCursor
            query = "SELECT * FROM orders"
            conditions = []
            params = []

            if order_status:
                conditions.append("order_status = %s")
                params.append(order_status)
            if customer_name:
                conditions.append("customer_name ILIKE %s")  # Case-insensitive search
                params.append(f"%{customer_name}%")
            if order_date_from:
                conditions.append("order_date >= %s")
                params.append(order_date_from)
            if order_date_to:
                conditions.append("order_date <= %s")
                params.append(order_date_to)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY order_date DESC LIMIT %s OFFSET %s"
            params.extend([limit, skip])

            cursor.execute(query, tuple(params))
            orders_db = cursor.fetchall()
            # Convert list of DictRow to list of Pydantic Order models
            return [Order.model_validate(row) for row in orders_db]
    except psycopg2.Error as e:
        logger.error(f"Database error fetching orders: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()


@app.get("/orders/{order_id}", response_model=Order, summary="Get a specific order by ID")
async def get_order_by_id(order_id: int, current_user: dict = Depends(get_current_user)):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM orders WHERE id = %s", (order_id,))
            order_db = cursor.fetchone()
            if not order_db:
                raise HTTPException(status_code=404, detail=f"Order with ID {order_id} not found")
            return Order.model_validate(order_db)
    except psycopg2.Error as e:
        logger.error(f"Database error fetching order by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()


@app.post("/emails/process-orders", summary="Fetch and process orders from emails")
async def trigger_email_order_processing(request: EmailProcessRequest, background_tasks: BackgroundTasks,
                                         current_user: dict = Depends(get_current_user)):
    """
    Triggers a background task to fetch emails based on subject keywords,
    parse PDFs or email bodies for order details, and store them.
    """
    subject_keywords = request.subject_query.split(",") if "," in request.subject_query else [request.subject_query]
    subject_keywords = [keyword.strip() for keyword in subject_keywords]
    logger.info(f"Received request to process emails with subjects: {subject_keywords}")
    background_tasks.add_task(
        process_and_store_email_orders_service,
        subject_keywords,
        request.only_recent_days,
        request.mark_as_read_after_extraction
    )
    return {"message": f"Email order processing task started in the background for subjects: {subject_keywords}"}


@app.post("/whatsapp/send-pending-db", summary="Send WhatsApp notifications for pending orders from DB")
async def send_pending_whatsapp_from_db(background_tasks: BackgroundTasks,
                                        current_user: dict = Depends(get_current_user)):
    """
    Fetches orders from DB where message_sent is FALSE and sends WhatsApp notifications.
    """
    conn = None
    pending_orders_info = []
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute("""
                SELECT id, product_name, customer_name, price, quantity, order_date
                FROM orders WHERE message_sent = FALSE ORDER BY order_date ASC
            """)
            pending_orders = cursor.fetchall()

            if not pending_orders:
                return {"message": "No pending WhatsApp messages from DB."}

            seller_numbers = get_seller_team_recipients()
            if not seller_numbers:
                logger.error("No valid seller WhatsApp numbers configured for pending DB messages.")
                raise HTTPException(status_code=500, detail="Seller WhatsApp numbers not configured.")

            orders_to_notify_count = 0
            for order_data in pending_orders:
                order_id = order_data["id"]
                message_lines = [
                    f"📦 *New Order Notification (from DB)* 📦",
                    f"Order ID: DB-{order_id}",
                    f"🛍️ Product: {order_data['product_name'] or 'N/A'}",
                    f"💰 Price: ₹{order_data['price']:.2f}" if order_data['price'] is not None else "Price: N/A",
                    f"👤 Customer: {order_data['customer_name'] or 'N/A'}",
                ]
                formatted_message = "\n".join(message_lines)

                # Add to background tasks
                background_tasks.add_task(active_whatsapp_sender, formatted_message, seller_numbers)
                # It's better to update message_sent AFTER the WhatsApp task confirms success,
                # but for simplicity here we'll assume it will be sent.
                # A more robust system would use a task queue and update status upon task completion.
                try:
                    with conn.cursor() as cur_update:  # Use a new cursor for update within the loop
                        cur_update.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (order_id,))
                    conn.commit()  # Commit each update
                    pending_orders_info.append({"order_id": order_id, "status": "notification_queued_and_marked_sent"})
                    orders_to_notify_count += 1
                except psycopg2.Error as e_upd:
                    conn.rollback()
                    logger.error(f"Failed to mark order DB-{order_id} as sent: {e_upd}")
                    pending_orders_info.append(
                        {"order_id": order_id, "status": "notification_queued_db_update_failed", "error": str(e_upd)})

            return {
                "message": f"Queued WhatsApp notifications for {orders_to_notify_count} pending order(s).",
                "details": pending_orders_info
            }

    except psycopg2.Error as e:
        logger.error(f"DB error processing pending messages: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error sending from DB: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if conn: conn.close()


@app.post("/whatsapp/quick-send", summary="Send a custom WhatsApp message to predefined contacts")
async def quick_whatsapp_send(request: QuickWhatsAppRequest, background_tasks: BackgroundTasks,
                              current_user: dict = Depends(get_current_user)):
    """
    Sends a custom message to selected contacts from a predefined list.
    NOTE: Relies on problematic desktop automation for WhatsApp.
    """
    contact_dict = {  # This should ideally be in a config or database
        "Narayan": "+919067847003", "Rani Bhise": "+917070242402", "Abhishek": "+919284625240",
        "Damini": "+917499353409", "Sandeep": "+919850030215", "Chandrakant": "+919665934999",
        "Vikas Kumbharkar": "+919284238738", "bhavin": "+916353761393"
    }
    selected_contacts_numbers = []
    invalid_contacts = []
    for name in request.contact_names:
        phone = contact_dict.get(name)
        if phone:
            fmt_num = format_phone_number(phone)
            if fmt_num:
                selected_contacts_numbers.append(fmt_num)
            else:
                invalid_contacts.append({"name": name, "phone": phone, "reason": "Invalid format"})
        else:
            invalid_contacts.append({"name": name, "reason": "Contact not found in predefined list"})

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not selected_contacts_numbers:
        raise HTTPException(status_code=400, detail="No valid contacts selected or found.")

    background_tasks.add_task(active_whatsapp_sender, request.message, selected_contacts_numbers)

    return {
        "message": f"WhatsApp message sending queued for {len(selected_contacts_numbers)} contact(s).",
        "sent_to_numbers": selected_contacts_numbers,
        "invalid_or_not_found_contacts": invalid_contacts if invalid_contacts else "None"
    }


# --- Dashboard Endpoints ---
@app.get("/dashboard/metrics", response_model=Metrics, summary="Get key order metrics")
async def get_dashboard_metrics(current_user: dict = Depends(get_current_user)):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            total_orders = (cur.fetchone() or [0])[0]

            cur.execute("SELECT SUM(price * quantity) FROM orders WHERE payment_status = 'Paid'")
            total_revenue = (cur.fetchone() or [0.0])[0] or 0.0

            cur.execute("SELECT AVG(price * quantity) FROM orders WHERE quantity > 0 AND price > 0")
            avg_order_value = (cur.fetchone() or [0.0])[0] or 0.0

            return Metrics(
                total_orders=total_orders,
                total_revenue_paid=float(total_revenue),
                avg_order_value=float(avg_order_value)
            )
    except psycopg2.Error as e:
        logger.error(f"Dashboard DB error (metrics): {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()


@app.get("/dashboard/status-distribution", response_model=List[StatusDistribution], summary="Get order count by status")
async def get_status_distribution(current_user: dict = Depends(get_current_user)):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT order_status, COUNT(*) as count FROM orders GROUP BY order_status")
            status_data = cur.fetchall()
            return [StatusDistribution(status=row[0] or "Unknown", count=row[1]) for row in status_data]
    except psycopg2.Error as e:
        logger.error(f"Dashboard DB error (status dist): {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()


@app.get("/dashboard/sales-by-category", response_model=List[SalesByCategory], summary="Get total sales by category")
async def get_sales_by_category():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT category, SUM(price * quantity) as total_sales
                FROM orders
                WHERE category IS NOT NULL AND category != '' AND quantity > 0 AND price > 0
                GROUP BY category ORDER BY total_sales DESC
            """)
            cat_sales_data = cur.fetchall()
            return [SalesByCategory(category=row[0], total_sales=float(row[1])) for row in cat_sales_data]
    except psycopg2.Error as e:
        logger.error(f"Dashboard DB error (category sales): {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn: conn.close()


@app.post("/signup", summary="Register a new user")
async def signup(user: UserSignup):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM login WHERE email = %s", (user.email,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")

            hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
            cur.execute(
                "INSERT INTO login (email, password) VALUES (%s, %s) RETURNING id",
                (user.email, hashed_password.decode('utf-8'))
            )
            conn.commit()
            return {"message": "User registered successfully"}
    except Exception as e:
        logger.error(f"Error during signup: {e}")
        raise HTTPException(status_code=500, detail="Error registering user")
    finally:
        conn.close()


@app.post("/signin", summary="Sign in a user")
async def signin(user: UserSignin):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email, password FROM login WHERE email = %s", (user.email,))
            db_user = cur.fetchone()
            if not db_user or not bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid email or password")

            token_expiry = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            token = jwt.encode(
                {"sub": db_user["email"], "exp": token_expiry},
                SECRET_KEY,
                algorithm=ALGORITHM
            )
            return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Error during signin: {e}")
        raise HTTPException(status_code=500, detail="Error signing in")
    finally:
        conn.close()


# --- To run the API (save this script as main.py) ---
# Command: uvicorn main:app --reload
# Make sure to create a .env file with your secrets:
# GMAIL_UNAME="your_email@gmail.com"
# GMAIL_PWD="your_gmail_app_password"
# IMAP_SERVER="imap.gmail.com"
# ADDITIONAL_WHATSAPP_RECIPIENTS="+91xxxxxxxxxx,+91yyyyyyyyyy"
# DB_HOST="localhost"
# DB_NAME="po_orders"
# DB_USER="po_user"
# DB_PASSWORD="postdb123"
# DB_PORT="5432"

if __name__ == "__main__":
    # This part is for development run only.
    # For production, use a process manager like Gunicorn with Uvicorn workers.
    # Example: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    logger.info("Starting API server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)