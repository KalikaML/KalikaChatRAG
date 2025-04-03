import streamlit as st
from load_faiss_from_s3 import initialize_s3_client, load_all_indexes

# Cache the index loading to avoid reloading on every interaction
@st.cache_resource
def load_indexes_with_counts(aws_access_key, aws_secret_key):
    s3_client = initialize_s3_client(aws_access_key, aws_secret_key)
    return load_all_indexes(s3_client)

st.title("RAG for Sales Team")

# Load AWS credentials from Streamlit secrets
try:
    aws_access_key = st.secrets["access_key_id"]
    aws_secret_key = st.secrets["secret_access_key"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please configure secrets in Streamlit Cloud.")
    st.stop()

# Load FAISS indexes and file counts
po_index, po_file_count, proforma_index, proforma_file_count = load_indexes_with_counts(aws_access_key, aws_secret_key)

# Display loading status and file counts
if po_index:
    st.success(f"PO Index Loaded Successfully! ({po_file_count} files)")
else:
    st.error("Failed to load PO Index")

if proforma_index:
    st.success(f"Proforma Index Loaded Successfully! ({proforma_file_count} files)")
else:
    st.error("Failed to load Proforma Index")

# Placeholder for RAG functionality
query = st.text_input("Enter a query for the sales team:")
if query:
    if po_index:
        po_results = po_index.similarity_search(query, k=3)
        st.write("Top PO Results:", po_results)
    if proforma_index:
        proforma_results = proforma_index.similarity_search(query, k=3)
        st.write("Top Proforma Results:", proforma_results)
    if not po_index and not proforma_index:
        st.warning("No indexes available to process the query.")