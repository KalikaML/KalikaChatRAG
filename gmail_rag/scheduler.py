import streamlit as st
import schedule
import time
import threading
import logging
from datetime import datetime
from streamlit import secrets
from proforma_s3store import process_proforma_emails, create_faiss_index as create_proforma_faiss
from PO_s3store import process_po_emails, create_faiss_index as create_po_faiss

# Load secrets from streamlit
EMAIL_ACCOUNT = st.secrets["gmail_uname"]
EMAIL_PASSWORD = st.secrets["gmail_pwd"]
AWS_ACCESS_KEY = st.secrets["access_key_id"]
AWS_SECRET_KEY = st.secrets["secret_access_key"]

# Set up logging
log_file = f'scheduler_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Global variable to track scheduler thread
scheduler_thread = None

def process_proforma():
    """Job to process proforma invoices"""
    try:
        logging.info("Starting Proforma Invoice processing...")
        process_proforma_emails()
        create_proforma_faiss()
        logging.info("Proforma Invoice processing completed successfully")
    except Exception as e:
        logging.error(f"Error in Proforma Invoice processing: {str(e)}")

def process_po():
    """Job to process PO documents"""
    try:
        logging.info("Starting PO processing...")
        process_po_emails()
        create_po_faiss()
        logging.info("PO processing completed successfully")
    except Exception as e:
        logging.error(f"Error in PO processing: {str(e)}")

def run_scheduler():
    """Set up and run the scheduler"""
    schedule.every().day.at("00:00").do(process_proforma)
    schedule.every().day.at("00:00").do(process_po)
    # schedule.every().day.at("21:52").do(process_proforma)
    # schedule.every().day.at("15:45").do(process_po)

    logging.info("Scheduler started...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(60)  # Wait before retrying

def start_scheduler_in_background():
    """Start the scheduler in a background thread"""
    global scheduler_thread
    if scheduler_thread is None or not scheduler_thread.is_alive():
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logging.info("Background scheduler thread started")
        return True
    return False

# Streamlit UI
st.title("Proforma and PO Processing Scheduler")

st.header("Scheduler Control")
if st.button("Start Scheduler"):
    if start_scheduler_in_background():
        st.success("Scheduler started successfully! It will run daily at now.")
    else:
        st.warning("Scheduler is already running.")

st.header("Scheduler Status")
if scheduler_thread and scheduler_thread.is_alive():
    st.write("Status: **Running**")
    st.write("Next run: Daily at 12:00 AM")
else:
    st.write("Status: **Stopped**")

st.header("Logs")
if st.button("Refresh Logs"):
    try:
        with open(log_file, "r") as f:
            logs = f.read()
        st.text_area("Recent Logs", logs, height=300)
    except FileNotFoundError:
        st.warning("No logs available yet.")