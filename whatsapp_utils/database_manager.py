# database_manager.py
import psycopg2
from streamlit import secrets # For DB credentials
from whatsapp_utils import config # For default values

DB_HOST = secrets.get("DB_HOST", config.DB_HOST_DEFAULT)
DB_NAME = secrets.get("DB_NAME", config.DB_NAME_DEFAULT)
DB_USER = secrets.get("DB_USER", config.DB_USER_DEFAULT)
DB_PASSWORD = secrets.get("DB_PASSWORD", config.DB_PASSWORD_DEFAULT)
DB_PORT = secrets.get("DB_PORT", config.DB_PORT_DEFAULT)

def create_db_tables(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY, order_type TEXT, product_name TEXT, category TEXT,
                    price REAL, quantity INTEGER, unit TEXT, specifications TEXT,
                    order_date TIMESTAMP, delivery_date DATE, customer_name TEXT,
                    customer_phone TEXT, email TEXT, address TEXT, payment_method TEXT,
                    payment_status TEXT, order_status TEXT, message_sent BOOLEAN DEFAULT FALSE,
                    source_email_subject TEXT, source_email_sender TEXT,
                    source_email_uid TEXT UNIQUE, raw_text_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_type ON orders(order_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_source_email_uid ON orders(source_email_uid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_message_sent ON orders(message_sent);")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_emails (
                    email_uid TEXT PRIMARY KEY, subject TEXT, sender TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, status TEXT,
                    related_order_id INTEGER REFERENCES orders(id) ON DELETE SET NULL
                );
            """)
        conn.commit()
        print("DBM: Database tables checked/created successfully.")
    except psycopg2.Error as e:
        print(f"DBM: Error creating/checking database tables: {e}")
        print(f"DBM: SQL Error Code: {e.pgcode}, Message: {e.pgerror}")
        if conn: conn.rollback()
    except Exception as e_gen:
        print(f"DBM: Generic error in create_db_tables: {e_gen}")
        if conn: conn.rollback()


def connect_to_db():
    """Connects to the PostgreSQL database and ensures tables exist."""
    conn = None
    try:
        print(f"DBM: Attempting to connect to DB: host={DB_HOST}, dbname={DB_NAME}, user={DB_USER}, port={DB_PORT}")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        print("DBM: Database connection successful.")
        create_db_tables(conn)
        return conn
    except psycopg2.OperationalError as e_op: # More specific error for connection issues
        print(f"DBM: OperationalError connecting to database (check server, network, credentials, pg_hba.conf): {e_op}")
        return None
    except psycopg2.Error as e_psql:
        print(f"DBM: psycopg2.Error connecting to database: {e_psql}")
        print(f"DBM: SQL Error Code: {e_psql.pgcode}, Message: {e_psql.pgerror}")
        return None
    except Exception as e_gen:
        print(f"DBM: Unexpected error during database connection: {e_gen}")
        return None

def is_email_processed(conn, email_uid):
    if not conn or not email_uid: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM processed_emails WHERE email_uid = %s", (email_uid,))
            return cursor.fetchone() is not None
    except psycopg2.Error as e:
        print(f"DBM: Error checking if email UID {email_uid} is processed: {e}")
        return False # On error, safer to assume not processed

def mark_email_as_processed(conn, email_uid, subject, sender, status, related_order_id=None):
    if not conn or not email_uid: return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO processed_emails (email_uid, subject, sender, status, related_order_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (email_uid) DO UPDATE SET
                    subject = EXCLUDED.subject, sender = EXCLUDED.sender,
                    processed_at = CURRENT_TIMESTAMP, status = EXCLUDED.status,
                    related_order_id = EXCLUDED.related_order_id;
            """, (email_uid, subject, sender, status, related_order_id))
        conn.commit()
        print(f"DBM: Email UID {email_uid} marked as processed: {status}")
        return True
    except psycopg2.Error as e:
        print(f"DBM: Error marking email UID {email_uid} as processed: {e}")
        if conn: conn.rollback()
        return False

def store_parsed_order_or_rfq(conn, details):
    if conn is None: print("DBM: DB connection not available for storing."); return None
    try:
        if not details.get("order_type") or (not details.get("product_name") and not details.get("specifications")):
            print(f"DBM: Skipping store: Missing order_type or product details. Data: {details}")
            return None
        if not details.get("source_email_uid") and "TEXT" in details.get("order_type", "") or "PDF" in details.get("order_type", ""):
             print(f"DBM Critical: source_email_uid missing for email-derived item. Data: {details}. Skipping store.")
             return None

        with conn.cursor() as cursor:
            columns = [
                "order_type", "product_name", "category", "price", "quantity", "unit",
                "specifications", "order_date", "delivery_date", "customer_name",
                "customer_phone", "email", "address", "payment_method",
                "payment_status", "order_status", "message_sent",
                "source_email_subject", "source_email_sender", "source_email_uid",
                "raw_text_content"
            ]
            values = [
                details.get("order_type"), details.get("product_name"), details.get("category"),
                details.get("price"), details.get("quantity"), details.get("unit"),
                details.get("specifications"), details.get("order_date"), details.get("delivery_date"),
                details.get("customer_name"), details.get("customer_phone"), details.get("email"),
                details.get("address"), details.get("payment_method"), details.get("payment_status"),
                details.get("order_status", "PENDING"),
                details.get("message_sent", False),
                details.get("source_email_subject"), details.get("source_email_sender"),
                details.get("source_email_uid"), details.get("raw_text_content")
            ]
            sql = f"INSERT INTO orders ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) RETURNING id;"
            cursor.execute(sql, tuple(values))
            new_order_id = cursor.fetchone()[0]
        conn.commit()
        print(f"DBM: {details.get('order_type')} stored (ID: {new_order_id}). Product: {details.get('product_name') or details.get('specifications')}")
        return new_order_id
    except psycopg2.Error as e:
        if conn: conn.rollback()
        if hasattr(e, 'pgcode') and e.pgcode == '23505':
             print(f"DBM Constraint Violation: Email UID {details.get('source_email_uid')} likely already processed. {e}")
        else: print(f"DBM: DB error storing item: {e} (Code: {getattr(e, 'pgcode', 'N/A')})")
        return None
    except Exception as e_gen:
        if conn: conn.rollback()
        print(f"DBM: Unexpected error storing item: {e_gen}")
        return None

def get_unnotified_items_count(conn):
    if not conn: return 0
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM orders WHERE message_sent = FALSE")
            return cursor.fetchone()[0]
    except psycopg2.Error as e:
        print(f"DBM: Error getting unnotified items count: {e}")
        return 0

def get_pending_notification_items(conn, limit=5):
    if not conn: return []
    try:
        with conn.cursor() as cur:
            cols = ["id", "order_type", "product_name", "category", "price", "quantity", "unit",
                    "specifications", "order_date", "delivery_date", "customer_name",
                    "customer_phone", "email", "address", "payment_method",
                    "payment_status", "order_status", "message_sent",
                    "source_email_subject", "source_email_sender", "source_email_uid",
                    "raw_text_content", "created_at"]
            cur.execute(f"SELECT {', '.join(cols)} FROM orders WHERE message_sent = FALSE ORDER BY created_at ASC LIMIT %s", (limit,))
            return cur.fetchall(), cols
    except psycopg2.Error as e:
        print(f"DBM: Error fetching pending notification items: {e}")
        return [], []

def mark_item_as_notified(conn, item_id):
    if not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE orders SET message_sent = TRUE WHERE id = %s", (item_id,))
        conn.commit()
        return True
    except psycopg2.Error as e:
        print(f"DBM: Error marking item {item_id} as notified: {e}")
        if conn: conn.rollback()
        return False