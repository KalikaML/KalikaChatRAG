import streamlit as st
import boto3
import os
import logging
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration constants
S3_BUCKET = "kalika-rag"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
LOCAL_FAISS_DIR = "C:/znew_chatboat_rag/faiss_index"  # Local folder to store FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-pro"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize S3 client using Streamlit secrets
def init_s3_client():
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["access_key_id"],
            aws_secret_access_key=st.secrets["secret_access_key"],
        )
        return s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {str(e)}")
        st.error("Failed to connect to S3. Please check your credentials.")
        return None

# Initialize Gemini API via LangChain
def init_gemini():
    try:
        model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=st.secrets["gemini_api_key"],
            temperature=0.5  # Adjustable parameter for response creativity
        )
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini: {str(e)}")
        st.error("Failed to connect to Gemini API. Please check your API key.")
        return None

# Download FAISS index from S3 to local folder if not already present
def ensure_faiss_index_local(s3_client):
    try:
        # Check if FAISS index already exists locally
        if os.path.exists(LOCAL_FAISS_DIR) and os.listdir(LOCAL_FAISS_DIR):
            logging.info("FAISS index found locally, loading from disk...")
            vector_store = FAISS.load_local(LOCAL_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            st.write(f"Using existing FAISS index from: {LOCAL_FAISS_DIR}")
            return vector_store

        # If not present, download from S3
        logging.info("No local FAISS index found, downloading from S3...")
        if os.path.exists(LOCAL_FAISS_DIR):
            shutil.rmtree(LOCAL_FAISS_DIR)  # Clear any incomplete folder
        os.makedirs(LOCAL_FAISS_DIR)

        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PROFORMA_INDEX_PATH)

        if 'Contents' not in response:
            logging.warning("No FAISS index found in S3.")
            return None

        total_size = 0
        for obj in response['Contents']:
            key = obj['Key']
            local_path = os.path.join(LOCAL_FAISS_DIR, os.path.basename(key))
            s3_client.download_file(S3_BUCKET, key, local_path)
            file_size = os.path.getsize(local_path)  # Get size in bytes
            total_size += file_size
            logging.info(f"Downloaded FAISS file: {key} ({file_size / 1024:.2f} KB)")

        # Convert total size to human-readable format
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"

        st.write(f"Total size of FAISS index downloaded: {size_str}")
        logging.info(f"Total FAISS index size: {size_str}")

        # Load the FAISS index from local folder
        vector_store = FAISS.load_local(LOCAL_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        logging.info("Successfully loaded FAISS index from local folder")
        return vector_store

    except Exception as e:
        logging.error(f"Failed to ensure FAISS index locally: {str(e)}")
        return None

# Query the FAISS index with adjustable k (number of results)
def query_faiss_index(vector_store, query, k=10):  # Increased k for broader queries
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logging.error(f"Error querying FAISS index: {str(e)}")
        return None

# Generate response using Gemini via LangChain
def generate_response(model, query, faiss_results):
    try:
        if not faiss_results:
            return "No relevant information found in the proforma invoices."

        # Combine FAISS results into a context
        context = "\n\n".join([result.page_content for result in faiss_results])
        prompt = (
            f"You are an expert assistant for a sales team analyzing proforma invoices. "
            f"Based on the following information from proforma invoices:\n\n{context}\n\n"
            f"Answer the query in detail: {query}"
        )

        # Use LangChain's invoke method to generate response
        response = model.invoke(prompt)
        return response.content  # Extract the content from the response object
    except Exception as e:
        logging.error(f"Error generating response with Gemini: {str(e)}")
        return "An error occurred while generating the response."

# Main chatbot interface
def main():
    st.title("Proforma Invoice Chatbot for Sales Team")

    # Initialize S3 client and Gemini model
    s3_client = init_s3_client()
    gemini_model = init_gemini()
    vector_store = None

    if s3_client:
        with st.spinner("Loading FAISS index..."):
            vector_store = ensure_faiss_index_local(s3_client)

        if vector_store:
            st.success("FAISS index successfully loaded!")
            st.write(f"FAISS index location: {LOCAL_FAISS_DIR}")
        else:
            st.error("Failed to load FAISS index from local or S3.")
            return

    if not gemini_model:
        return

    # Query input and response
    if vector_store:
        st.subheader("Ask a Question")
        query = st.text_input("Enter your query about the proforma invoices (e.g., 'How many proformas do we have?', 'List all products', 'Details for a specific company'):")

        if st.button("Submit"):
            if query:
                with st.spinner("Searching and generating response..."):
                    # Search FAISS index with a higher k to capture more context
                    faiss_results = query_faiss_index(vector_store, query, k=10)

                    # Generate response with Gemini
                    response = generate_response(gemini_model, query, faiss_results)

                st.subheader("Response")
                st.write(response)
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()