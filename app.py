import streamlit as st
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from botocore.exceptions import ClientError

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Debugging: Show available secrets (remove this after debugging)
st.write("Available Secrets:", st.secrets)


# Function to load FAISS index from S3 directly into memory
def load_faiss_index_from_s3(bucket_name, key):
    # Initialize the S3 client using credentials from Streamlit Secrets
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws_secret_access_key"],
        region_name=st.secrets["aws_region"]
    )
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        index_binary = response['Body'].read()
        index = faiss.deserialize_index(index_binary)  # Deserialize FAISS index into memory
        return index
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.error(f"The specified key '{key}' does not exist in bucket '{bucket_name}'.")
        else:
            st.error(f"An unexpected error occurred: {e}")
        return None


# Paths to FAISS indexes in S3
bucket_name = "kalika-rag"  # Replace with your actual S3 bucket name
po_index_key = "faiss_indexes/po_faiss_index/index_file.bin"
proforma_index_key = "faiss_indexes/proforma_faiss_index/index_file.bin"


# Load FAISS indexes dynamically based on user selection
def get_faiss_index(query_type):
    if query_type == "Proforma":
        return load_faiss_index_from_s3(bucket_name, proforma_index_key)
    elif query_type == "Purchase Order":
        return load_faiss_index_from_s3(bucket_name, po_index_key)


# Function to retrieve top k results from the FAISS index
def retrieve(query, index, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return distances, indices


# Streamlit app layout
st.title("Proforma and PO Query Chatbot")

# Dropdown for query type selection
query_type = st.selectbox("Select Query Type", ["Proforma", "Purchase Order"])

# Text input for user query
user_query = st.text_input("Enter your query:")
if st.button("Submit"):
    # Load the appropriate FAISS index based on user selection
    faiss_index = get_faiss_index(query_type)

    if faiss_index:  # Proceed only if the index was loaded successfully
        distances, indices = retrieve(user_query, faiss_index)
        st.write(f"Top results for {query_type}:")
        st.write(f"indices: {indices}")
        st.write(f"distances: {distances}")
