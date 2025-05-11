# whatsapp_utils.py
import phonenumbers
import webbrowser
import pyautogui  # For web automation
import time
import random
from streamlit import secrets
from whatsapp_utils import config  # For defaults

# Seller team recipients from secrets or config
SELLER_TEAM_RECIPIENTS_STR = secrets.get("ADDITIONAL_WHATSAPP_RECIPIENTS", config.DEFAULT_SELLER_TEAM_RECIPIENTS_STR)


def format_phone_number_wa(phone_str):  # Renamed to avoid conflict
    if not phone_str or not isinstance(phone_str, str): return None
    try:
        p_num = phonenumbers.parse(phone_str, "IN" if not phone_str.startswith('+') else None)
        return phonenumbers.format_number(p_num, phonenumbers.PhoneNumberFormat.E164) if phonenumbers.is_valid_number(
            p_num) else None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None


def get_seller_team_recipients_wa():  # Renamed
    recipients = set()
    if SELLER_TEAM_RECIPIENTS_STR and isinstance(SELLER_TEAM_RECIPIENTS_STR, str):
        for phone in SELLER_TEAM_RECIPIENTS_STR.split(","):
            fmt_seller = format_phone_number_wa(phone.strip())
            if fmt_seller:
                recipients.add(fmt_seller)
            else:
                print(f"WAU: Invalid seller phone in config: {phone.strip()}")
    if not recipients:
        print("WAU: No valid seller WhatsApp recipients configured.")
        # if 'st' in globals(): st.warning("No valid seller WhatsApp recipients configured.") # Avoid st in non-UI modules ideally
    return list(recipients)


def send_whatsapp_message_via_web(message, recipient_numbers_list):  # Renamed
    # status_container = st.empty() # Cannot use st.empty() here directly if called from bg thread
    print(f"WAU: Attempting to send WA message via web to {len(recipient_numbers_list)} recipients.")
    if not isinstance(recipient_numbers_list, (list, set)) or not recipient_numbers_list:
        print("WAU: Invalid/No recipients for web WhatsApp.")
        return False

    try:  # Wrap webbrowser.open in try-except for headless environments or if no browser
        webbrowser.open("https://web.whatsapp.com")
    except Exception as e_wb:
        print(f"WAU: Could not open web browser: {e_wb}. WhatsApp Web sending will likely fail.")
        # return False # Optionally fail early if browser can't be opened

    print("WAU: Opened WhatsApp Web. User needs to scan QR if needed. Waiting 25s...")
    time.sleep(25)  # User time to scan QR code

    all_successful = True
    for recipient in recipient_numbers_list:
        formatted_recipient = format_phone_number_wa(recipient)
        if not formatted_recipient:
            print(f"WAU: Skipping invalid WA phone: {recipient}")
            all_successful = False
            continue
        try:
            import urllib.parse
            encoded_message = urllib.parse.quote(message)
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_recipient}&text={encoded_message}"
            webbrowser.open(whatsapp_url)
            print(f"WAU: Web: Opened chat for {formatted_recipient}. Waiting 15s for page to load...")
            time.sleep(15)
            pyautogui.press("enter")  # This is the riskiest part; requires GUI and correct window focus
            print(f"WAU: Web: 'Enter' key pressed for {formatted_recipient}.")
            # if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.whatsapp_sent_counter += 1 # Manage session state in app.py
            time.sleep(random.uniform(4, 7))  # Be a good netizen
        except Exception as e:
            print(f"WAU: Web WhatsApp Error for {formatted_recipient}: {e}")
            # if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.whatsapp_errors.append(f"Web WA Error for {formatted_recipient}: {str(e)}")
            all_successful = False
    return all_successful


# Fallback or alternative using pywhatkit (if user prefers and handles browser tabs)
# def send_whatsapp_message_via_pywhatkit(message, recipient_numbers_list): ...

# CHOSEN SENDER (can be configured)
active_whatsapp_sender = send_whatsapp_message_via_web


def format_whatsapp_message_for_db_item(item_data_row, db_columns_list):  # Renamed
    data = dict(zip(db_columns_list, item_data_row))
    item_id, item_type = data['id'], data['order_type']
    product_info = data.get('product_name') or data.get('specifications') or "N/A"
    customer = data.get('customer_name') or data.get('source_email_sender', '').split('<')[0].strip() or "Unknown"
    item_date = data.get('order_date') or data.get('created_at')
    date_str = item_date.strftime('%d-%b-%Y %H:%M') if isinstance(item_date, datetime) else str(item_date or "N/A")

    title_emoji = "ℹ️"
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
        lines.append(f"🔢 Qty: {data.get('quantity', 'N/A')} {data.get('unit', '') or ''}")
        if data.get('price') is not None: lines.append(f"💰 Price: ₹{data['price']:.2f}")
        lines.append(f"🚚 Delivery: {data.get('delivery_date', 'N/A')}")
        lines.append(f"📊 Status: {data.get('order_status', 'PROCESSING')}")
    elif "RFQ" in item_type or "INQUIRY" in item_type:
        lines.append(f"📋 Details: {data.get('specifications', product_info)[:200]}")
        if data.get('quantity'): lines.append(f"🔢 Qty: {data.get('quantity')} {data.get('unit', '') or ''}")
        lines.append(f"📊 Status: {data.get('order_status', 'PENDING_QUOTE')}")
    lines.append(f"📧 Subject (Ref): {data.get('source_email_subject', 'N/A')[:70]}")
    return "\n".join(lines)


def send_pending_db_notifications(db_conn):  # Renamed
    print("WAU: Checking for pending DB notifications...")
    if not db_conn: print("WAU: DB connection lost for notifications."); return 0

    from database_manager import get_pending_notification_items, \
        mark_item_as_notified  # Local import to avoid top-level circularity

    notifications_sent_this_run = 0
    pending_items_rows, item_cols = get_pending_notification_items(db_conn, limit=5)  # Limit to 5 per run

    if not pending_items_rows:
        print("WAU: No new items in DB needing WhatsApp notification currently.")
        return 0

    seller_recipients = get_seller_team_recipients_wa()
    if not seller_recipients: return 0

    print(f"WAU: Found {len(pending_items_rows)} item(s) to notify via WhatsApp.")
    for item_row in pending_items_rows:
        item_id_for_notif = item_row[item_cols.index('id')]
        message = format_whatsapp_message_for_db_item(item_row, item_cols)
        print(f"WAU: Preparing WA for Item ID: {item_id_for_notif} ({item_row[item_cols.index('order_type')]})")

        if active_whatsapp_sender(message, seller_recipients):
            if mark_item_as_notified(db_conn, item_id_for_notif):
                print(f"WAU: Notification for item DB-{item_id_for_notif} sent & DB marked.")
                notifications_sent_this_run += 1
                # Invalidate cache in app.py if Streamlit context is available
                # if 'st' in globals() and hasattr(st, 'session_state'): st.session_state.pending_msg_count = None
                time.sleep(random.uniform(10, 15))
            else:
                print(f"WAU: WA sent for {item_id_for_notif} but FAILED to mark in DB.")
        else:
            print(f"WAU: Failed WA for item DB-{item_id_for_notif}. Will retry later.")

    if notifications_sent_this_run > 0:
        print(f"WAU: Successfully sent {notifications_sent_this_run} notifications in this run.")
    return notifications_sent_this_run