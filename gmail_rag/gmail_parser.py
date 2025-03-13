"""
Problem Statement : Build RAG on Kalika Enterprises Gmail Data
Tasks
1) Understand the details of various documents
    i) Proforma Invoice:A proforma invoice is a preliminary bill that a seller sends to a buyer before a sale is confirmed. It's a non-binding document that's used for planning, budgeting, and estimates. It's also used for customs clearance and financing
    ii) PO Dump : Pending and Processed data details
2) Create the Parsing process
    1) Proforma Invoice
    - Searching a with proforma invoice for smtp mail python utility download attached file
    - Dump that on scheduling daily to s3
    - Extract the content using ocr engine and valdate
   2)  PO dump
   _ On Daily basis Dump is in mail extract excel store in s3
   - Implement pg vector

##### Prompt:
with gmail Imap server to extract invoice proforma from attached mail to dump in s3 bucket
write a end to end flow with python


"""

import imaplib
import email
from email.parser import BytesParser

# Gmail IMAP server details
IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993
USERNAME =
PASSWORD =

# Connect to the IMAP server
import streamlit as st
import imaplib
import email
from email.parser import BytesParser
import boto3
from streamlit import secrets

# Load secrets from .secrets.toml
secrets_config = secrets.toml_file_config("secrets.toml")

# Gmail IMAP server details
IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

# AWS S3 details
s3 = boto3.client('s3', aws_access_key_id=secrets_config['aws']['access_key_id'],
                  aws_secret_access_key=secrets_config['aws']['secret_access_key'])


def connect_to_imap():
    """Connect to Gmail's IMAP server."""
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(secrets_config['imap']['username'], secrets_config['imap']['password'])
    return mail


def search_emails(mail, query):
    """Search for emails based on the query."""
    mail.select('inbox')
    status, messages = mail.search(None, query)
    return messages


def extract_attachments(mail, message_id):
    """Extract attachments from an email."""
    status, msg = mail.fetch(message_id, '(RFC822)')
    raw_message = msg[0][1]
    message = BytesParser().parsebytes(raw_message)

    attachments = []
    for part in message.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        filename = part.get_filename()
        if bool(filename):
            attachments.append((filename, part.get_payload(decode=True)))
    return attachments


def upload_to_s3(attachments):
    """Upload attachments to S3."""
    for filename, data in attachments:
        try:
            s3.put_object(Body=data, Bucket=secrets_config['aws']['bucket_name'], Key=filename)
            print(f"Uploaded {filename} to S3")
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")


def main():
    st.title("Email Search and Attachment Uploader")

    query = st.text_input("Enter search query (e.g., 'FROM sender@example.com')", value="")

    if st.button("Search Emails"):
        mail = connect_to_imap()
        messages = search_emails(mail, query)

        if messages[0]:
            for num in messages[0].split():
                attachments = extract_attachments(mail, num)
                if attachments:
                    upload_to_s3(attachments)
                    st.success("Attachments uploaded to S3")
                else:
                    st.info("No attachments found in this email")
        else:
            st.info("No emails found matching the query")

        mail.close()
        mail.logout()


if __name__ == "__main__":
    main()

