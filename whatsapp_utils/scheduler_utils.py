# scheduler_utils.py
import schedule
import time
from datetime import datetime
from streamlit import secrets  # For scheduler config

# Import necessary functions from other modules
# These imports assume all .py files are in the same directory or Python's import path.
from whatsapp_utils import database_manager as dbm
from whatsapp_utils import email_utils as emu
from whatsapp_utils import order_parser as op
from whatsapp_utils import whatsapp_utils as wau
from whatsapp_utils import config  # For defaults


# Orchestrator for email processing (called by scheduler or manually)
def run_email_processing_cycle(db_connection, recent_days_to_check=None):
    """
    Orchestrates one full cycle of fetching, parsing, and storing email data.
    Returns counts of new POs and RFQs processed.
    """
    processed_pos = 0
    processed_rfqs = 0
    mail_connection_obj = None  # To hold the active IMAP connection object

    subjects = secrets.get("EMAIL_SUBJECTS_LIST", config.DEFAULT_EMAIL_SUBJECTS_TO_MONITOR)

    # Step 1: Fetch basic info for new emails
    # fetch_and_prepare_emails now returns a list of dicts, each with a 'mail_connection' key
    # if it intends to keep the connection open.
    # Let's refine fetch_and_prepare_emails to handle its own connection lifecycle for simplicity,
    # or ensure it's properly managed here.
    # For now, assume fetch_and_prepare_emails gives us what we need to proceed.

    email_data_list = emu.fetch_and_prepare_emails(
        db_conn=db_connection,
        subjects_to_monitor=subjects,
        only_recent_days=recent_days_to_check
    )

    if not email_data_list:
        print("SCHED: No new emails fetched for processing in this cycle.")
        return 0, 0  # No new emails, so nothing to do.

    # The IMAP connection might still be open if email_data_list is not empty
    # and fetch_and_prepare_emails was designed to pass it.
    # Assuming the first email's 'mail_connection' is the one to use for all 'store' operations if needed.
    # However, it's safer if fetch_and_prepare_emails handles its logout unless specifically passing the connection.
    # Let's assume fetch_and_prepare_emails closes its own connection after fetching if it doesn't pass it.
    # If it *does* pass mail_connection in each item, we need to manage it.
    # For simplicity, the current emu.fetch_and_prepare_emails tries to close if it fetched nothing.
    # If it *did* fetch emails, it implies the connection might be open.

    active_mail_connection = None
    if email_data_list and 'mail_connection' in email_data_list[0]:
        active_mail_connection = email_data_list[0]['mail_connection']

    for email_content in email_data_list:
        email_uid = email_content["uid"]
        msg = email_content["msg"]
        email_subject = email_content["subject"]
        email_sender = email_content["sender"]

        print(f"SCHED: Processing UID {email_uid}, Subject: {email_subject}")

        has_pdf = False
        parsed_details = None
        processing_status_for_db_log = "NO_RELEVANT_CONTENT"  # Default
        related_order_id = None

        # 1. Check for PDF
        for part in msg.walk():
            if part.get_content_type() == "application/pdf" and part.get('Content-Disposition'):
                filename = emu.decode_mail_header_robust(part.get_filename()) or f'attachment_{email_uid}.pdf'
                pdf_data = part.get_payload(decode=True)
                if pdf_data:
                    print(f"SCHED: Found PDF: {filename} in UID {email_uid}")
                    has_pdf = True
                    pdf_text = emu.extract_text_from_pdf_bytes(pdf_data)
                    if pdf_text:
                        parsed_details = op.parse_order_details_from_pdf_text(pdf_text, email_uid, email_subject,
                                                                              email_sender)
                        if parsed_details:
                            processing_status_for_db_log = "PDF_PARSED_PENDING_STORE"
                        else:
                            processing_status_for_db_log = "PDF_NO_DETAILS"
                    else:
                        processing_status_for_db_log = "PDF_EMPTY_TEXT"
                    break  # Process first PDF found

        # 2. If no PDF, check body text
        if not has_pdf:
            print(f"SCHED: No PDF in UID {email_uid}. Checking body text.")
            email_body = emu.get_email_body_text_content(msg)
            if email_body:
                list_of_parsed_text_items = op.parse_details_from_email_body(email_body, email_uid, email_subject,
                                                                             email_sender)
                if list_of_parsed_text_items:
                    # Store each item found in text
                    temp_po = 0;
                    temp_rfq = 0
                    for item_detail_from_text in list_of_parsed_text_items:
                        new_id = dbm.store_parsed_order_or_rfq(db_connection, item_detail_from_text)
                        if new_id:
                            if not related_order_id: related_order_id = new_id  # Link to first stored item
                            if item_detail_from_text["order_type"].startswith("PO"):
                                temp_po += 1
                            else:
                                temp_rfq += 1  # RFQ or Inquiry
                    if temp_po > 0 or temp_rfq > 0:
                        processing_status_for_db_log = f"TEXT_DB_STORED ({temp_po} PO, {temp_rfq} RFQ)"
                        processed_pos += temp_po;
                        processed_rfqs += temp_rfq
                    else:
                        processing_status_for_db_log = "TEXT_ITEMS_PARSED_BUT_NOT_STORED"
                    # parsed_details is not set here because items are stored directly
                else:
                    processing_status_for_db_log = "TEXT_NO_RELEVANT_ITEMS_FOUND"
            else:
                processing_status_for_db_log = "EMPTY_EMAIL_BODY"

        # 3. If PDF was parsed (and details found), store it now
        if parsed_details and processing_status_for_db_log == "PDF_PARSED_PENDING_STORE":
            new_id = dbm.store_parsed_order_or_rfq(db_connection, parsed_details)
            if new_id:
                related_order_id = new_id
                if parsed_details["order_type"].startswith("PO"):
                    processed_pos += 1
                else:
                    processed_rfqs += 1  # Assuming PDF can be RFQ/Other too
                processing_status_for_db_log = f"PDF_DB_STORED ({parsed_details['order_type']})"
            else:
                processing_status_for_db_log = "PDF_DB_STORE_FAILED"

        # 4. Log processing status for this email UID in DB
        dbm.mark_email_as_processed(db_connection, email_uid, email_subject, email_sender, processing_status_for_db_log,
                                    related_order_id)

        # 5. Mark as SEEN on IMAP server
        # This needs the 'mail' object from fetch_and_prepare_emails or a fresh one.
        # The 'active_mail_connection' should be the one used for UID fetches.
        if active_mail_connection:
            try:
                active_mail_connection.uid('store', email_uid, '+FLAGS', '\\Seen')
                print(f"SCHED: Marked UID {email_uid} as SEEN on server.")
            except Exception as e_seen:
                print(f"SCHED: Failed to mark UID {email_uid} as SEEN on server: {e_seen}")
        else:
            print(
                f"SCHED: No active IMAP connection to mark UID {email_uid} as SEEN. This might indicate an issue in connection passing.")

    # Ensure the main IMAP connection used for fetching UIDs is closed
    if active_mail_connection:
        try:
            active_mail_connection.close()
            active_mail_connection.logout()
            print("SCHED: Main IMAP connection for UID processing closed and logged out.")
        except Exception as e_logout_main:
            print(f"SCHED: Error logging out main IMAP connection: {e_logout_main}")

    return processed_pos, processed_rfqs


def scheduled_email_check_task():  # Renamed
    """Task run by the scheduler."""
    print(f"SCHED TASK: [{datetime.now()}] Automatic email check initiated by scheduler...")
    db_conn = dbm.connect_to_db()  # Get a fresh DB connection
    if not db_conn:
        print(f"SCHED TASK: DB connection failed. Skipping email check cycle.")
        return

    try:
        recent_days = secrets.get("SCHEDULER_RECENT_DAYS", config.DEFAULT_SCHEDULER_RECENT_DAYS)
        pos_found, rfqs_found = run_email_processing_cycle(
            db_connection=db_conn,
            recent_days_to_check=recent_days
        )
        print(f"SCHED TASK: Email check cycle complete. New POs: {pos_found}, New RFQs: {rfqs_found}")

        if pos_found > 0 or rfqs_found > 0:
            print(f"SCHED TASK: New items found, attempting to send notifications.")
            wau.send_pending_db_notifications(db_conn)  # Pass the same DB connection
        else:
            print(f"SCHED TASK: No new items from email to notify immediately.")

    except Exception as e:
        print(f"SCHED TASK: Error during scheduled email check: {e}")
        import traceback;
        traceback.print_exc()
    finally:
        if db_conn:
            db_conn.close()
            print(f"SCHED TASK: DB connection closed.")
    # Update last check time in Streamlit session state (app.py will handle st.session_state)
    # This function is called from a thread, so direct st.session_state manipulation
    # should be done carefully or via a callback if Streamlit's threading model requires it.
    # For now, app.py can update it after calling this or based on a timer.


def run_background_scheduler_thread():  # Renamed
    """Target function for the scheduler thread."""
    print("SCHEDULER THREAD: Started. Waiting for scheduled jobs...")
    while True:
        schedule.run_pending()
        time.sleep(1)