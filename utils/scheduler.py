import schedule
import time
import threading
import logging
from datetime import datetime
from PO_s3store import process_po_emails, create_faiss_index_po
from proforma_s3store import process_proforma_emails, create_faiss_index

scheduler_thread = None

def run_jobs():
    """Runs the scheduled tasks: fetches new PO and proforma emails, processes them, and updates FAISS index."""
    jobs = [
        (process_po_emails, "Processing PO Emails"),
        (create_faiss_index_po, "Updating PO FAISS Index"),
        (process_proforma_emails, "Processing Proforma Emails"),
        (create_faiss_index, "Updating Proforma FAISS Index")
    ]
    
    for job, desc in jobs:
        try:
            logging.info(f"Starting: {desc}")
            job()
            logging.info(f"Completed: {desc}")
        except Exception as e:
            logging.error(f"Error in {desc}: {e}")

def run_scheduler():
    """Sets up and runs the scheduler at midnight (12:00 AM) and checks for new emails every 10 minutes."""
    schedule.every().day.at("00:00").do(run_jobs)  # Main job at midnight
    schedule.every(10).minutes.do(run_jobs)  # Check for new emails every 10 minutes
    logging.info("Scheduler started, running at 12:00 AM daily and every 10 minutes...")
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler():
    """Starts the scheduler thread if not already running."""
    global scheduler_thread
    if not scheduler_thread or not scheduler_thread.is_alive():
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logging.info("Scheduler thread started.")
        return True
    return False

if __name__ == "__main__":
    start_scheduler()



#This script runs proforma_s3store.py and PO_s3store.py at 12:00 AM daily.
#It keeps running in the background to trigger the jobs at the scheduled time.
#If new files arrive in the email, the existing scripts will process them automatically.
