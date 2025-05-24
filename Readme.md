# PO Order Manager

This project sets up a purchase order management system using PostgreSQL for the database, FastAPI for the backend, and a frontend built with Node.js (React or similar framework).

---

## 📚 Table of Contents

1. [Database Setup](##database-setup)
2. [Python Virtual Environment Setup](##python-virtual-environment-setup)
3. [Install Requirements](##install-requirements)
4. [Run Backend](##run-backend)
5. [Run Frontend](##run-frontend)
6. [Database Usage](##database-usage)

---

## Database Setup

```sql
-- Create user
CREATE USER po_user WITH PASSWORD 'postdb123';
ALTER USER po_user WITH CREATEDB;

-- Create database
CREATE DATABASE po_orders OWNER po_user;
GRANT ALL PRIVILEGES ON DATABASE po_orders TO po_user;

-- Connect to po_orders
\c po_orders

-- Create orders table
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

-- Create login table
CREATE TABLE IF NOT EXISTS login (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE orders TO po_user;
GRANT ALL PRIVILEGES ON TABLE login TO po_user;
GRANT ALL ON SCHEMA public TO po_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO po_user;
```

## Create Python Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

## Install Requirements
```
pip install -r requirements.txt
```

## Run Backend
```
uvicorn whatsapp_backend:app --host 0.0.0.0 --port 8000
```

## Run Frontend
```
cd po-order-manager
npm run dev
```

## Show Table Data
```
SELECT * FROM orders;
SELECT * FROM login;    
```