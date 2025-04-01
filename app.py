'''import streamlit as st
import boto3
import faiss
import pickle
import json
from io import BytesIO
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import nest_asyncio

# Apply nest_asyncio patch for event loop issues
nest_asyncio.apply()

# Initialize S3 client with credentials from Streamlit secrets
s3 = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

# Initialize Hugging Face embeddings model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Use a valid SentenceTransformers model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


def load_faiss_and_metadata(bucket_name, index_path):
    """Load FAISS index and metadata (pickle) from S3."""
    try:
        # Load FAISS index file from S3
        index_response = s3.get_object(Bucket=bucket_name, Key=f"{index_path}/index.faiss")
        index_bytes = BytesIO(index_response['Body'].read())
        faiss_index = faiss.read_index(index_bytes)

        # Load metadata (pickle file) from S3
        metadata_response = s3.get_object(Bucket=bucket_name, Key=f"{index_path}/index.pkl")
        metadata_bytes = BytesIO(metadata_response['Body'].read())
        metadata = pickle.load(metadata_bytes)

    except s3.exceptions.NoSuchKey:
        st.error(f"Error: The specified key does not exist in bucket '{bucket_name}' at path '{index_path}'.")
        return None, None

    return faiss_index, metadata


def query_faiss_index(faiss_index, metadata, query_embedding, k=5):
    """Query FAISS index to retrieve relevant documents."""
    if faiss_index is None or metadata is None:
        st.error("FAISS index or metadata is not loaded.")
        return []

    distances, indices = faiss_index.search(query_embedding, k)
    results = [metadata[idx] for idx in indices[0]]  # Retrieve documents using indices
    return results


def filter_and_format_results(results):
    """Format results for display."""
    formatted_output = [{"Proforma ID": res.get("id"), "Date": res.get("date"), "Details": res.get("details")} for res in results]
    return formatted_output


def get_pdf_count(bucket_name, folder_path):
    """Get the count of PDFs in the S3 folder."""
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    pdf_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
    return len(pdf_files), pdf_files


# Load FAISS index and metadata from S3 bucket dynamically
bucket_name = "kalika-rag"
index_path = "faiss_indexes/proforma_faiss_index"
folder_path = "proforma_invoice/"
faiss_index, metadata = load_faiss_and_metadata(bucket_name, index_path)

# Streamlit app interface
st.title("Proforma Invoice Chatbot & PDF Monitor")

# Section 1: Query Interface
st.header("Query Proforma Invoices")
user_query = st.text_input("Enter your query:")
if user_query:
    start_time = datetime.now()

    query_embedding = embeddings.embed_query(user_query)
    results = query_faiss_index(faiss_index, metadata, query_embedding)

    formatted_results = filter_and_format_results(results)

    end_time = datetime.now()

    st.write(f"Query processed in {(end_time - start_time).total_seconds()} seconds.")

    st.write("Results:")
    for result in formatted_results:
        st.json(result)

# Section 2: PDF Count Monitoring
st.header("PDF Count Monitoring")

if "previous_pdf_count" not in st.session_state:
    st.session_state.previous_pdf_count = 0

current_pdf_count, pdf_files = get_pdf_count(bucket_name, folder_path)

st.write(f"Previous PDF Count: {st.session_state.previous_pdf_count}")
st.write(f"Current PDF Count: {current_pdf_count}")

if current_pdf_count > st.session_state.previous_pdf_count:
    st.write("New PDFs detected!")
else:
    st.write("No new PDFs added.")

st.session_state.previous_pdf_count = current_pdf_count

# List all PDF files
st.write("PDF Files:")
for pdf_file in pdf_files:
    st.write(pdf_file)
'''
import streamlit as st
import boto3
import os
import json
from typing import BinaryIO
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# AWS Configuration
s3 = boto3.client('s3',
                  aws_access_key_id=st.secrets['AWS_ACCESS_KEY'],
                  aws_secret_access_key=st.secrets['AWS_SECRET_KEY'])
bucket_name = st.secrets['S3_BUCKET']


@st.cache_resource
def load_faiss_from_s3(folder: str):
    """Load FAISS index directly from S3 without local download"""
    index_file = f"{folder}/index.faiss"
    config_file = f"{folder}/index.pkl"

    # Load index files from S3
    index_obj = s3.get_object(Bucket=bucket_name, Key=index_file)
    config_obj = s3.get_object(Bucket=bucket_name, Key=config_file)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create FAISS instance from bytes
    return FAISS.deserialize_from_bytes(
        embeddings=embeddings,
        serialized=(
            index_obj['Body'].read(),
            config_obj['Body'].read()
        )
    )


def get_new_file_count(folder: str):
    """Count unprocessed files in S3 folder"""
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/raw/")
    return len(response.get('Contents', []))


# Streamlit UI
st.title("Proforma/PO Chat Assistant")
doc_type = st.selectbox("Select Document Type", ["Proforma", "Purchase Order"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display new file counts
proforma_count = get_new_file_count("proforma_invoice")
po_count = get_new_file_count("PO_Dump")
st.sidebar.markdown(f"**New Proforma Files:** {proforma_count}")
st.sidebar.markdown(f"**New PO Files:** {po_count}")

# Load appropriate FAISS index
folder_map = {
    "Proforma": "proforma_invoice",
    "Purchase Order": "PO_Dump"
}
selected_folder = folder_map[doc_type]
vector_db = load_faiss_from_s3(selected_folder)

# RAG Pipeline
prompt_template = """Use the following context to answer the question:
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Proforma/PO documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = qa_chain.invoke({"query": prompt})

    with st.chat_message("assistant"):
        st.markdown(response['result'])
    st.session_state.messages.append({"role": "assistant", "content": response['result']})
