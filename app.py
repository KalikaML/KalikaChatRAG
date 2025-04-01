import streamlit as st
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import os
import pandas as pd

# AWS S3 configuration
S3_BUCKET = st.secrets['S3_BUCKET_NAME '] # Replace with your real bucket name
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/index.faiss"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/index.faiss"

# Initialize S3 client with region
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets['aws_region']
)


# Initialize sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


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
                    "Document": document_data[doc_idx],
                    "Similarity Score": float(1 / (1 + distance))  # Convert distance to similarity
                })
            else:
                results.append({
                    "Match Rank": idx + 1,
                    "Document": "No matching document found",
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

        s3_path = PROFORMA_INDEX_PATH if query_type == "Proforma Query" else PO_INDEX_PATH
        index_type = "Proforma" if query_type == "Proforma Query" else "PO"

        with st.spinner(f"Loading {index_type} index from S3..."):
            index = load_faiss_index_from_s3(s3_path)

        if index:
            # Placeholder document data (replace with actual retrieval logic)
            document_data = ["Doc1: PO #123 details", "Doc2: Proforma #456 info", "Doc3: PO #789 summary"]

            with st.spinner(f"Generating {index_type} response..."):
                response = generate_response(user_query, index, model, document_data)

            if response:
                st.success(f"{index_type} Response Generated!")
                # Display structured output as a table
                df = pd.DataFrame(response)
                st.table(df)
            else:
                st.error("Failed to generate response")
        else:
            st.error("Failed to load FAISS index from S3")


if __name__ == "__main__":
    main()