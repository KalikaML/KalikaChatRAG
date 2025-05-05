# This script establishes a connection to a local PostgreSQL database, creates an orders table with detailed fields if it doesn't exist, 
# and provides a function to insert new order records into the table.
import psycopg2

def connect_to_db():
    """Connects to the local PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="po_orders",
            user="po_user",
            password="postdb123",
            port=5432
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def create_orders_table(conn):
    """Creates the 'orders' table if it doesn't exist."""
    with conn.cursor() as cursor: # Use context manager
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


def store_order(conn, order_details):
    try:
        with conn.cursor() as cursor:
            columns = [
                "product_name", "category", "price", "quantity", "order_date", "delivery_date",
                "customer_name", "customer_phone", "email", "address", "payment_method",
                "payment_status", "order_status", "message_sent"
            ]
            values = [
                order_details["Product Name"],
                order_details.get("Category", ""),
                order_details["Price"],
                order_details["Quantity"],
                order_details["Order Date"],
                order_details["Delivery Date"],
                order_details["Customer Name"],
                order_details["Raw Customer Phone"],
                order_details.get("Email", ""),
                order_details["Address"],
                order_details["Payment Method"],
                order_details["Payment Status"],
                order_details["Order Status"],
                False
            ]

            sql = f"""
                INSERT INTO orders ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
            """

            print("SQL Query:", sql) # Keep these for debugging if needed
            print("Values:", values)

            cursor.execute(sql, tuple(values))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Store order error: {e}", type(e))
        return False