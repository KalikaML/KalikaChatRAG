import streamlit as st
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from io import BytesIO
import tempfile

# AWS S3 configuration
S3_BUCKET = "your-s3-bucket-name"  # Replace with your S3 bucket name
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/index.faiss"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/index.faiss"

# Initialize S3 client using Streamlit secrets
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
)


# Initialize sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# Function to load FAISS index directly from S3 into memory
def load_faiss_index_from_s3(s3_path):
    try:
        # Get object from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
        index_bytes = response['Body'].read()

        # Use a temporary file to load the index (Streamlit Cloud compatible)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(index_bytes)
            tmp_file.flush()
            index = faiss.read_index(tmp_file.name)

        # Clean up the temporary file
        os.unlink(tmp_file.name)
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index from S3: {str(e)}")
        return None


# RAG function to generate response
def generate_response(query, index, model, document_data):
    try:
        query_embedding = model.encode([query])[0]
        D, I = index.search(np.array([query_embedding]), k=3)
        relevant_docs = [document_data[i] for i in I[0]]
        response = "Based on the query, here's what I found:\n\n"
        for doc in relevant_docs:
            response += f"- {doc}\n"
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Streamlit app
def main():
    st.title("Proforma & PO Query Chatbot (Streamlit Cloud)")

    model = load_embedding_model()

    query_type = st.selectbox(
        "Select Query Type",
        ["Proforma Query", "PO Query"]
    )

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
            document_data = ["Doc1", "Doc2", "Doc3"]  # Replace with actual data retrieval
            with st.spinner(f"Generating {index_type} response..."):
                response = generate_response(user_query, index, model, document_data)
                st.success("Response generated!")
                st.write(response)
        else:
            st.error("Failed to load FAISS index from S3")


if __name__ == "__main__":
    main()