import streamlit as st
import boto3
import faiss
import io
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA



# AWS S3 Configuration
S3_BUCKET = "kalika-rag"
S3_PO_FOLDER = "Po_Dumps/"
S3_PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
S3_PROFORMA_FOLDER = "proforma_invoice/"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Initialize S3 Client
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
                         # region_name=st.secrets["region"])

# st.write(st.secrets["access_key_id"])
# st.write(st.secrets["secret_access_key"])
# st.write(st.secrets["region"])

def list_s3_pdfs(folder):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
    return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]

def extract_text_from_s3(s3_key):
    text = ""
    pdf_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
    pdf_stream = io.BytesIO(pdf_obj["Body"].read())
    with pdfplumber.open(pdf_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def process_documents(folder):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_texts = []
    pdf_keys = list_s3_pdfs(folder)
    for s3_key in pdf_keys:
        text = extract_text_from_s3(s3_key)
        all_texts.extend(text_splitter.split_text(text))
    return all_texts

def create_vector_store(documents, index_path):
    if not documents:
        st.warning("No data found!")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    faiss_bytes = faiss.serialize_index(vector_store.index)
    faiss_buffer = io.BytesIO(faiss_bytes)
    s3_client.upload_fileobj(faiss_buffer, S3_BUCKET, index_path)
    print("vector store in path", index_path)
    return vector_store

def get_vector_store(index_path, folder):
    try:
        faiss_buffer = io.BytesIO()
        s3_client.download_fileobj(S3_BUCKET, index_path, faiss_buffer)
        faiss_buffer.seek(0)
        index = faiss.deserialize_index(faiss_buffer.read())
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS(embedding_function=embeddings, index=index)
    except Exception as e:
        print(f"⚠️ Error loading FAISS index for {index_path}: {e}")
        print("Rebuilding FAISS index...")
        documents = process_documents(folder)
        return create_vector_store(documents, index_path)
#exception


def query_rag(query, index_path, folder):
    vector_store = get_vector_store(index_path, folder)
    print("vector store in path", index_path)
    print("vector store:", vector_store)
    if not vector_store:
        return "Index not found. Please build the index first."
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama2:latest")
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return chain.run(query)

# Streamlit UI
st.set_page_config(page_title="Proforma & PO Chatbot", layout="wide")
st.title("Proforma Invoice & PO Dump Chatbot")

# Tabs
tab1, tab2 = st.tabs(["Proforma Invoice Chatbot", "PO Dump Chatbot"])

with tab1:
    st.header("Ask about Proforma Invoices")
    proforma_query = st.text_input("Enter your query about Proforma Invoices:")
    if proforma_query:
        proforma_answer = query_rag(proforma_query, S3_PROFORMA_INDEX_PATH, S3_PROFORMA_FOLDER)
        st.write("Answer:", proforma_answer)

with tab2:
    st.header("Ask about PO Dump")
    po_query = st.text_input("Enter your query about PO Dump:")
    if po_query:
        print("PO query entered")
        print("po folder::", S3_PO_FOLDER)
        po_answer = query_rag(po_query, S3_PO_INDEX_PATH, S3_PO_FOLDER)
        st.write("Answer:", po_answer)
