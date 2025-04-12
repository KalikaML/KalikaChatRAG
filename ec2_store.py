import psycopg2
import os

# Database credentials from environment variables
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def connect_to_db():
    """Connects to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None


def create_orders_table(conn):
    """Creates the 'orders' table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            product_name TEXT,
            price REAL,
            quantity INTEGER,
            order_date TIMESTAMP,
            delivery_date DATE,
            customer_name TEXT,
            customer_phone TEXT,
            address TEXT,
            payment_method TEXT,
            payment_status TEXT,
            order_status TEXT,
            message_sent BOOLEAN DEFAULT FALSE
        )
    """
    )
    conn.commit()
    cursor.close()


def store_order(conn, order_details):
    """Stores order details in the 'orders' table."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO orders (product_name, price, quantity, order_date, delivery_date,
                              customer_name, customer_phone, address, payment_method,
                              payment_status, order_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                order_details["Product Name"],
                float(order_details["Price"]),  # Ensure price is a float
                int(order_details["Quantity"]),  # Ensure quantity is an integer
                order_details["Order Date"],
                order_details["Delivery Date"],
                order_details["Customer Name"],
                order_details["Raw Customer Phone"],
                order_details["Address"],
                order_details["Payment Method"],
                order_details["Payment Status"],
                order_details["Order Status"],
            ),
        )
        conn.commit()
        print("Order stored successfully!")
        return True

    except Exception as e:  # Catch any exception during the process
        print(f"Error storing order: {e}")
        conn.rollback()  # Roll back transaction in case of error
        return False
    finally:
        cursor.close()


if __name__ == "__main__":
    # Example Usage (for testing purposes)
    conn = connect_to_db()
    if conn:
        create_orders_table(conn)
        # Example order details (replace with your actual data)
        order_details = {
            "Product Name": "Test Product",
            "Price": "25.00",
            "Quantity": "2",
            "Order Date": "2025-04-12 10:00:00",
            "Delivery Date": "2025-04-15",
            "Customer Name": "Test Customer",
            "Raw Customer Phone": "+15551234567",
            "Address": "123 Test St",
            "Payment Method": "Credit Card",
            "Payment Status": "Paid",
            "Order Status": "Shipped",
        }
        if store_order(conn, order_details):
            print("Test order stored successfully.")
        else:
            print("Failed to store test order.")
        conn.close()
    else:
        print("Failed to connect to the database.")
