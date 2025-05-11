# app.py
import streamlit as st
from datetime import datetime, timedelta
import schedule
import threading
import time
import pandas as pd  # For displaying data in tables

# Import functions from our modules
from whatsapp_utils import config
import whatsapp_utils.database_manager as dbm
import whatsapp_utils.email_utils as emu  # Not directly called by UI much, but scheduler_utils uses it
import whatsapp_utils.order_parser as op  # Not directly called by UI much
import whatsapp_utils.whatsapp_utils as wau
import whatsapp_utils.scheduler_utils as su

# --- Session State Initialization ---
default_ss_keys = {
    "whatsapp_sent_counter": 0, "whatsapp_errors": [],
    "last_check_time": datetime.now() - timedelta(hours=1),
    "auto_check_enabled": st.secrets.get("SCHEDULER_AUTO_ENABLE", config.DEFAULT_SCHEDULER_AUTO_ENABLE),
    "check_interval_minutes": st.secrets.get("SCHEDULER_INTERVAL_MINUTES", config.DEFAULT_SCHEDULER_INTERVAL_MINUTES),
    "scheduler_started": False, "last_scheduled_interval": None,
    "db_init_success": False, "scheduler_thread_obj": None,  # Renamed to avoid conflict
    "pending_msg_count_val": None, "pending_msg_count_last_updated_val": None,  # Renamed
}
for key, default_val in default_ss_keys.items():
    if key not in st.session_state: st.session_state[key] = default_val
# --- End Session State ---

st.set_page_config(page_title="PO/RFQ Email Processor", layout="wide")
st.title("📧 PO & RFQ Email Processor & Notifier V2")

# --- Initial DB Connection Check ---
if not st.session_state.db_init_success:
    print("APP: UI: Attempting initial DB connection...")
    with st.spinner("Connecting to Database..."):
        db_conn_ui_init = dbm.connect_to_db()  # Use function from database_manager
        if db_conn_ui_init:
            st.session_state.db_init_success = True
            st.sidebar.success("Database Connected & Tables Verified.")
            db_conn_ui_init.close()  # Close after check
        else:
            st.sidebar.error("CRITICAL: DB Connection FAILED! Most features will not work.")
            # st.stop() # Optionally stop the app if DB is essential

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Status")
    if not st.session_state.db_init_success:
        st.warning("Database not connected. Functionality limited.")

    st.subheader("📧 Email Processing")
    email_subjects_monitored = st.secrets.get("EMAIL_SUBJECTS_LIST", config.DEFAULT_EMAIL_SUBJECTS_TO_MONITOR)
    st.info(f"Monitoring {len(email_subjects_monitored)} subject patterns.")
    # with st.expander("Monitored Subjects"):
    #     for subj_pattern in email_subjects_monitored: st.caption(subj_pattern)

    if st.button("Manually Process Emails Now", key="manual_email_check_btn_app",
                 disabled=not st.session_state.db_init_success):
        with st.spinner("Manual Email Check... This may take some time."):
            db_manual_conn = dbm.connect_to_db()
            if db_manual_conn:
                try:
                    # Call the orchestrator from scheduler_utils
                    recent_days_manual = st.secrets.get("MANUAL_CHECK_RECENT_DAYS",
                                                     config.DEFAULT_MANUAL_CHECK_RECENT_DAYS)
                    pos, rfqs = su.run_email_processing_cycle(
                        db_connection=db_manual_conn,
                        recent_days_to_check=recent_days_manual
                    )
                    st.success(f"Manual check: {pos} POs, {rfqs} RFQs/Inquiries processed from new emails.")
                    if pos > 0 or rfqs > 0:
                        st.info("Attempting to send notifications for newly processed items...")
                        wau.send_pending_db_notifications(db_manual_conn)
                    st.session_state.last_check_time = datetime.now()  # Update last check time
                    st.session_state.pending_msg_count_val = None  # Invalidate pending count cache
                except Exception as e_man:
                    st.error(f"Error during manual email processing: {e_man}")
                    import traceback;

                    traceback.print_exc()
                finally:
                    db_manual_conn.close()
            else:
                st.error("Manual check failed: Could not connect to DB.")
        st.rerun()

    st.subheader("⚙️ Automatic Scheduler")
    auto_check_ui_val = st.checkbox("Enable Automatic Email Processing", value=st.session_state.auto_check_enabled,
                                    key="auto_check_cb_app")
    if auto_check_ui_val != st.session_state.auto_check_enabled:
        st.session_state.auto_check_enabled = auto_check_ui_val
        st.rerun()

    interval_ui_val = st.slider(
        "Scheduler Interval (minutes)", 5, 180,
        st.session_state.check_interval_minutes, 5, key="interval_sl_app",
        disabled=not st.session_state.auto_check_enabled
    )
    if interval_ui_val != st.session_state.check_interval_minutes:
        st.session_state.check_interval_minutes = interval_ui_val
        st.rerun()

    st.caption(f"Last manual/auto-check trigger: {st.session_state.last_check_time.strftime('%I:%M %p, %d-%b')}")
    if st.session_state.auto_check_enabled and st.session_state.scheduler_started:
        next_run_time_val = schedule.next_run()
        if next_run_time_val: st.caption(f"Next auto-check: {next_run_time_val.strftime('%I:%M %p, %d-%b')}")

    st.subheader("📱 WhatsApp Notifications")


    def get_displayed_pending_count_sidebar():
        now = datetime.now()
        cache_duration_min = st.secrets.get("PENDING_COUNT_CACHE_MIN", config.DEFAULT_PENDING_COUNT_CACHE_MIN)
        if (st.session_state.pending_msg_count_val is None or
                st.session_state.pending_msg_count_last_updated_val is None or
                (now - st.session_state.pending_msg_count_last_updated_val) > timedelta(minutes=cache_duration_min)):
            print("APP: Sidebar: Refreshing pending notification count from DB.")
            conn = dbm.connect_to_db()
            if conn:
                try:
                    st.session_state.pending_msg_count_val = dbm.get_unnotified_items_count(conn)
                except Exception as e_cnt_sb:
                    print(f"APP: Error getting pending count for sidebar: {e_cnt_sb}")
                    st.session_state.pending_msg_count_val = "Error"
                finally:
                    conn.close()
            else:
                st.session_state.pending_msg_count_val = "DB Err"
            st.session_state.pending_msg_count_last_updated_val = now
        return st.session_state.pending_msg_count_val


    pending_val_sidebar = get_displayed_pending_count_sidebar()
    st.metric("Items Needing Notification",
              pending_val_sidebar if isinstance(pending_val_sidebar, int) else str(pending_val_sidebar))

    if st.button("Send Pending Notifications Now", key="send_pending_wa_btn_app",
                 disabled=not (isinstance(pending_val_sidebar,
                                          int) and pending_val_sidebar > 0 and st.session_state.db_init_success)):
        with st.spinner(f"Sending {pending_val_sidebar} notifications..."):
            conn_notif_app = dbm.connect_to_db()
            if conn_notif_app:
                try:
                    wau.send_pending_db_notifications(conn_notif_app)
                    st.session_state.pending_msg_count_val = None  # Invalidate cache
                finally:
                    conn_notif_app.close()
            else:
                st.error("Notification sending failed: DB Connection error.")
        st.rerun()

    seller_recipients_list = wau.get_seller_team_recipients_wa()  # Get list for display
    st.caption(f"WA Recipients: {', '.join(seller_recipients_list) if seller_recipients_list else 'None Configured'}")
    st.caption(f"Total WA Sent (Session): {st.session_state.whatsapp_sent_counter}")
    if st.session_state.whatsapp_errors:
        with st.expander("WA Errors (Session)"): st.error("\n".join(st.session_state.whatsapp_errors))

# --- Main Content Tabs ---
tab_dashboard_app, tab_items_app, tab_manual_app, tab_qwa_app, tab_logs_app = st.tabs([
    "📊 Dashboard", "📋 Processed Items", "✍️ Manual Entry", "📞 Quick WA", "📜 Email Logs"
])

with tab_dashboard_app:
    st.header("📊 Overview Dashboard")
    if st.session_state.db_init_success:
        # ... (Dashboard content using dbm functions to fetch data) ...
        # Example:
        conn_dash_app = dbm.connect_to_db()
        if conn_dash_app:
            try:
                with conn_dash_app.cursor() as cur:
                    st.subheader("Key Metrics (Last 30 Days)")
                    c1, c2, c3, c4 = st.columns(4)
                    thirty_days_ago = datetime.now() - timedelta(days=30)
                    cur.execute("SELECT COUNT(*) FROM orders WHERE order_type LIKE 'PO%%' AND created_at >= %s",
                                (thirty_days_ago,));
                    po_count = cur.fetchone()[0] or 0
                    c1.metric("POs (30d)", po_count)
                    cur.execute("SELECT COUNT(*) FROM orders WHERE order_type LIKE 'RFQ%%' AND created_at >= %s",
                                (thirty_days_ago,));
                    rfq_count = cur.fetchone()[0] or 0
                    c2.metric("RFQs/Inquiries (30d)", rfq_count)
                    cur.execute(
                        "SELECT SUM(price * quantity) FROM orders WHERE order_type LIKE 'PO%%' AND payment_status = 'Paid' AND created_at >= %s",
                        (thirty_days_ago,));
                    po_rev = cur.fetchone()[0] or 0
                    c3.metric("Paid PO Rev (30d)", f"₹{po_rev:.2f}")
                    cur.execute("SELECT COUNT(*) FROM processed_emails WHERE processed_at >= %s", (thirty_days_ago,));
                    email_log_count = cur.fetchone()[0] or 0
                    c4.metric("Emails Logged (30d)", email_log_count)
            except Exception as e_dash_app:
                st.error(f"Dashboard data error: {e_dash_app}")
            finally:
                conn_dash_app.close()
        else:
            st.warning("Dashboard: DB connection failed.")
    else:
        st.warning("Dashboard: DB not initialized.")

with tab_items_app:
    st.header("📋 Processed Orders, RFQs & Inquiries")
    if st.session_state.db_init_success:
        # ... (Display items from 'orders' table using dbm functions) ...
        conn_items_app = dbm.connect_to_db()
        if conn_items_app:
            try:
                with conn_items_app.cursor() as cur:
                    cur.execute("""SELECT id, created_at, order_type, product_name, specifications, quantity, unit,
                                   customer_name, email, order_status, message_sent, source_email_uid
                                   FROM orders ORDER BY created_at DESC LIMIT 100""")
                    items_rows_app = cur.fetchall()
                    if items_rows_app:
                        cols_items_app = ['ID', 'Logged At', 'Type', 'Product', 'Specs', 'Qty', 'Unit', 'Customer',
                                          'Contact', 'Status', 'Notified', 'Email UID']
                        df_items_app = pd.DataFrame(items_rows_app, columns=cols_items_app)
                        st.dataframe(df_items_app, use_container_width=True, hide_index=True)
                    else:
                        st.write("No processed items found.")
            except Exception as e_items_app:
                st.error(f"Error loading items: {e_items_app}")
            finally:
                conn_items_app.close()
        else:
            st.warning("Items List: DB connection failed.")
    else:
        st.warning("Items List: DB not initialized.")

with tab_manual_app:
    st.header("✍️ Manual Data Entry for PO / RFQ / Inquiry")
    if st.session_state.db_init_success:
        with st.form("manual_entry_form", clear_on_submit=True):
            st.subheader("Item Details")
            entry_type = st.selectbox("Entry Type *", ["PO_MANUAL", "RFQ_MANUAL", "INQUIRY_MANUAL"], key="man_type")
            product_name = st.text_input("Product Name/Service *", key="man_prod")
            specifications = st.text_area("Specifications/Description *", key="man_spec")
            quantity = st.number_input("Quantity", min_value=0, step=1, key="man_qty")
            unit = st.text_input("Unit (e.g., PCS, KG, MTR)", key="man_unit")

            st.subheader("Customer & Dates")
            customer_name = st.text_input("Customer Name *", key="man_cust_name")
            customer_email = st.text_input("Customer Email", key="man_cust_email")
            customer_phone = st.text_input("Customer Phone", key="man_cust_phone")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                order_date = st.date_input("Order/Request Date *", datetime.now().date(), key="man_order_date")
            with col_d2:
                delivery_date = st.date_input("Expected Delivery Date (if PO)", key="man_del_date", value=None)

            is_po = "PO" in entry_type
            if is_po:
                st.subheader("PO Specifics")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    price = st.number_input("Total Price (₹)", min_value=0.0, format="%.2f", key="man_price",
                                            value=None)
                    payment_method = st.text_input("Payment Method", key="man_pay_method")
                with col_p2:
                    address = st.text_area("Delivery Address", key="man_addr")
                    payment_status = st.selectbox("Payment Status", ["Pending", "Paid", "Unpaid", "Partial"],
                                                  key="man_pay_status")

            order_status_options = ["PENDING", "PROCESSING", "CONFIRMED", "PENDING_QUOTE", "QUOTE_SENT", "NEEDS_REVIEW",
                                    "COMPLETED", "CANCELLED"]
            order_status = st.selectbox("Overall Status *", order_status_options, key="man_order_status")

            submitted = st.form_submit_button("Save Entry")

            if submitted:
                if not all([entry_type, product_name or specifications, customer_name, order_date, order_status]):
                    st.error("Please fill all required (*) fields.")
                else:
                    manual_details = {
                        "order_type": entry_type, "product_name": product_name, "specifications": specifications,
                        "quantity": quantity if quantity > 0 else None, "unit": unit,
                        "customer_name": customer_name, "email": customer_email, "customer_phone": customer_phone,
                        "order_date": datetime.combine(order_date, datetime.min.time()),  # Store as timestamp
                        "delivery_date": delivery_date if is_po and delivery_date else None,
                        "price": price if is_po and price is not None else None,
                        "payment_method": payment_method if is_po else None,
                        "address": address if is_po else None,
                        "payment_status": payment_status if is_po else None,
                        "order_status": order_status,
                        "source_email_uid": f"MANUAL_ENTRY_{int(time.time())}"  # Unique ID for manual entries
                    }
                    conn_man = dbm.connect_to_db()
                    if conn_man:
                        new_id = dbm.store_parsed_order_or_rfq(conn_man, manual_details)
                        if new_id:
                            st.success(f"Manual entry saved successfully! ID: {new_id}")
                            st.session_state.pending_msg_count_val = None  # Invalidate cache
                            # Optionally trigger WA notification here
                            # wau.send_pending_db_notifications(conn_man) # Or a specific one for this entry
                        else:
                            st.error("Failed to save manual entry to database.")
                        conn_man.close()
                    else:
                        st.error("Failed to save: DB Connection Error.")
                    st.rerun()  # To refresh lists and counts
    else:
        st.warning("Manual Entry: DB not initialized.")

with tab_qwa_app:
    st.header("📞 Quick WhatsApp Message")
    if st.session_state.db_init_success:  # WA utils don't strictly need DB but good to check app health
        # Your existing Quick WhatsApp UI using wau.send_whatsapp_notification
        # ...
        qwa_message = st.text_area("Enter message:", height=100, key="qwa_msg")
        qwa_recipients_str = st.text_input("Enter recipient numbers (comma-separated, with country code e.g., +91...):",
                                           value=st.secrets.get("DEFAULT_QUICK_WA_RECIPIENTS", ""), key="qwa_rec")

        if st.button("Send Quick WhatsApp", key="qwa_send_btn"):
            if qwa_message and qwa_recipients_str:
                qwa_rec_list_raw = [r.strip() for r in qwa_recipients_str.split(',')]
                qwa_rec_list_valid = [wau.format_phone_number_wa(r) for r in qwa_rec_list_raw if
                                      wau.format_phone_number_wa(r)]

                if qwa_rec_list_valid:
                    with st.spinner(f"Sending to {len(qwa_rec_list_valid)} valid numbers..."):
                        success_qwa = wau.active_whatsapp_sender(qwa_message, qwa_rec_list_valid)
                        if success_qwa:
                            st.success("Quick WhatsApp message(s) initiated!")
                        else:
                            st.error("Failed to send one or more Quick WhatsApp messages.")
                        # Update session state for sent counter/errors if you modify wau.active_whatsapp_sender to return details
                else:
                    st.warning("No valid recipient phone numbers entered.")
            else:
                st.warning("Please enter a message and recipient numbers.")
    else:
        st.warning("Quick WA: DB not initialized (affects overall app health check).")

with tab_logs_app:
    st.header("📜 Email Processing Logs (Recent)")
    if st.session_state.db_init_success:
        # ... (Display logs from 'processed_emails' table using dbm functions) ...
        conn_logs_app = dbm.connect_to_db()
        if conn_logs_app:
            try:
                with conn_logs_app.cursor() as cur:
                    cur.execute("""SELECT processed_at, email_uid, subject, sender, status, related_order_id
                                   FROM processed_emails ORDER BY processed_at DESC LIMIT 200""")  # Show more logs
                    logs_app = cur.fetchall()
                    if logs_app:
                        df_logs_app = pd.DataFrame(logs_app, columns=['Timestamp', 'Email UID', 'Subject', 'Sender',
                                                                      'Processing Status', 'Related Order ID'])
                        st.dataframe(df_logs_app, use_container_width=True, hide_index=True)
                    else:
                        st.write("No email processing logs found.")
            except Exception as e_log_app:
                st.error(f"Error loading logs: {e_log_app}")
            finally:
                conn_logs_app.close()
        else:
            st.warning("Logs: DB connection failed.")
    else:
        st.warning("Logs: DB not initialized.")

# --- Background Scheduler Setup ---
if st.session_state.auto_check_enabled and st.session_state.db_init_success:  # Only run scheduler if DB is up
    current_interval_app = st.session_state.check_interval_minutes
    interval_has_changed_app = st.session_state.last_scheduled_interval != current_interval_app

    if not st.session_state.scheduler_started or interval_has_changed_app:
        schedule.clear()
        job = schedule.every(current_interval_app).minutes.do(
            su.scheduled_email_check_task)  # Call task from scheduler_utils
        st.session_state.last_scheduled_interval = current_interval_app
        print(f"APP: Scheduler: Job (re)defined: Every {current_interval_app} mins. Job: {job}")

        if not st.session_state.scheduler_started:
            if st.session_state.scheduler_thread_obj is None or not st.session_state.scheduler_thread_obj.is_alive():
                thread_app = threading.Thread(target=su.run_background_scheduler_thread, daemon=True,
                                              name="AppBgScheduler")
                thread_app.start()
                st.session_state.scheduler_thread_obj = thread_app
                st.session_state.scheduler_started = True
                print("APP: Scheduler: Background thread started.")
                st.toast(f"Auto-processing ON: every {current_interval_app}m.", icon="⏰")
            else:
                print("APP: Scheduler: Background thread already running. Job definition updated.")
                st.toast(f"Auto-processing updated: every {current_interval_app}m.", icon="🔄")
        else:
            print(f"APP: Scheduler: Job interval updated to {current_interval_app} minutes.")
            st.toast(f"Auto-processing interval updated to {current_interval_app}m.", icon="🔄")

elif not st.session_state.auto_check_enabled and st.session_state.scheduler_started:
    schedule.clear()
    st.session_state.scheduler_started = False
    st.session_state.last_scheduled_interval = None
    print("APP: Scheduler: Automatic email processing disabled. All jobs cleared.")
    st.toast("Auto-processing OFF.", icon="🛑")

print(
    f"APP: Streamlit script rerun completed at {datetime.now()}. Auto-check: {st.session_state.auto_check_enabled}, Scheduler Active: {st.session_state.scheduler_started}")