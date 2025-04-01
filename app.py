import streamlit as st
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import os
import pandas as pd
import pickle

# AWS S3 configuration
S3_BUCKET = "kalika-rag"  # From your scripts
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/index.faiss"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/index.faiss"
PO_CHUNKS_PATH = "faiss_indexes/po_faiss_index/chunks.pkl"  # Assumed path for PO chunks
PROFORMA_CHUNKS_PATH = "faiss_indexes/proforma_faiss_index/chunks.pkl"  # Assumed path for Proforma chunks

# Initialize S3 client with region
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets['aws_region']  # Replace with your bucket's region, e.g., 'us-east-1'
)


# Initialize sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Function to load FAISS index from S3
def load_faiss_index_from_s3(s3_path):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
        index_bytes = response['Body'].read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(index_bytes)
            tmp_file.flush()
            index = faiss.read_index(tmp_file.name)
        os.unlink(tmp_file.name)
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index from S3: {str(e)}")
        return None


# Function to load document chunks from S3
def load_document_chunks_from_s3(chunks_path):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=chunks_path)
        chunks_bytes = response['Body'].read()
        chunks = pickle.loads(chunks_bytes)  # Assuming chunks are pickled
        return chunks
    except Exception as e:
        st.error(f"Error loading document chunks from S3: {str(e)}")
        return []


# RAG function to generate structured response
def generate_response(query, index, model, document_data):
    try:
        query_embedding = model.encode([query])[0]
        D, I = index.search(np.array([query_embedding]), k=3)  # Top 3 matches

        # Structured output: list of dictionaries
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(D[0], I[0])):
            if doc_idx < len(document_data):
                results.append({
                    "Match Rank": idx + 1,
                    "Document Chunk": document_data[doc_idx],
                    "Similarity Score": float(1 / (1 + distance))  # Convert distance to similarity
                })
            else:
                results.append({
                    "Match Rank": idx + 1,
                    "Document Chunk": "No matching document found",
                    "Similarity Score": float(1 / (1 + distance))
                })
        return results
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None


# Streamlit app
def main():
    st.title("Proforma & PO Query Chatbot (Streamlit Cloud)")
    model = load_embedding_model()
    query_type = st.selectbox("Select Query Type", ["Proforma Query", "PO Query"])
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if not user_query:
            st.warning("Please enter a query!")
            return

        # Select paths based on query type
        s3_index_path = PROFORMA_INDEX_PATH if query_type == "Proforma Query" else PO_INDEX_PATH
        s3_chunks_path = PROFORMA_CHUNKS_PATH if query_type == "Proforma Query" else PO_CHUNKS_PATH
        index_type = "Proforma" if query_type == "Proforma Query" else "PO"

        # Load FAISS index
        with st.spinner(f"Loading {index_type} index from S3..."):
            index = load_faiss_index_from_s3(s3_index_path)

        if index:
            # Load document chunks
            with st.spinner(f"Loading {index_type} document chunks from S3..."):
                document_data = load_document_chunks_from_s3(s3_chunks_path)

            if not document_data:
                st.error(f"No document chunks found for {index_type}")
                return

            # Generate and display response
            with st.spinner(f"Generating {index_type} response..."):
                response = generate_response(user_query, index, model, document_data)

            if response:
                st.success(f"{index_type} Response Generated!")
                df = pd.DataFrame(response)
                st.table(df)
            else:
                st.error("Failed to generate response")
        else:
            st.error("Failed to load FAISS index from S3")


if __name__ == "__main__":
    main()