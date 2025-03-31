'''import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, fetch_faiss_index_from_s3

# Access secrets from Streamlit Cloud
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

# Fetch FAISS index from S3 on startup and initialize locally if needed
if fetch_faiss_index_from_s3():
    print("Successfully loaded FAISS index from S3.")
else:
    print("Failed to load FAISS index from S3.")

# Function to process user queries and generate structured output
def query_proforma_rag(query):
    """Query the RAG model using FAISS index and generate structured output."""
    vector_store = get_faiss_index()
    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update.", []

    # Initialize retriever and LLM
    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    )

    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Run query through the chain
    response = chain.run(query)

    # Retrieve relevant documents for structuring output
    sources = retriever.get_relevant_documents(query)

    # Generate structured output
    structured_output = []
    for source in sources:
        structured_output.append({
            "text": source.page_content,
            "source": source.metadata.get("source", "Unknown")
        })

    return response, structured_output

# Streamlit UI Setup
st.title("Proforma Invoice Query System")

# Sidebar Section: Instructions
st.sidebar.header("Instructions")
st.sidebar.write("Enter your query below to fetch relevant information from proforma invoices.")

# Main Section: Query Input and Response Display
st.header("Query Proforma Invoices")

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages (user and bot interactions)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "bot":
        st.chat_message("bot").write(message["content"])

# User input for chat interaction
if user_query := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get bot response and structured output using query_proforma_rag function
    bot_response, structured_output = query_proforma_rag(user_query)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Display bot response immediately in the chat interface without context
    st.chat_message("bot").write(bot_response)

    # Display structured output using Streamlit widgets
    st.header("Structured Output")
    for item in structured_output:
        with st.expander(f"Source: {item['source']}"):
            st.write(item['text'])'''

import streamlit as st
import boto3
import faiss
import json
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

# Initialize S3 client with credentials from Streamlit secrets
s3 = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

# Load Hugging Face model and tokenizer for embeddings
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=st.secrets["HUGGINGFACE_ACCESS_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=st.secrets["HUGGINGFACE_ACCESS_TOKEN"])


def get_embeddings(texts):
    """Generate embeddings using HuggingFaceH4/zephyr-7b-beta."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()


def load_faiss_index(bucket_name, index_path):
    """Load FAISS index and metadata from S3."""
    index_response = s3.get_object(Bucket=bucket_name, Key=f"{index_path}/index.faiss")
    docstore_response = s3.get_object(Bucket=bucket_name, Key=f"{index_path}/docstore.json")

    index_bytes = BytesIO(index_response['Body'].read())
    docstore_data = json.loads(docstore_response['Body'].read())

    # Deserialize FAISS index
    faiss_index = faiss.read_index(index_bytes)

    return faiss_index, docstore_data["docstore"], docstore_data["mapping"]


def query_faiss_index(faiss_index, docstore, query_embedding, k=5):
    """Query FAISS index to retrieve relevant documents."""
    distances, indices = faiss_index.search(query_embedding, k)
    results = [docstore[str(idx)] for idx in indices[0]]
    return results


def filter_and_format_results(results):
    """Format results for display."""
    formatted_output = [{"Proforma ID": res["id"], "Date": res["date"], "Details": res["details"]} for res in results]
    return formatted_output


def get_pdf_count(bucket_name, folder_path):
    """Get the count of PDFs in the S3 folder."""
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    pdf_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
    return len(pdf_files), pdf_files


# Load FAISS index and docstore from S3 bucket dynamically
bucket_name = "kalika-rag"
index_path = "faiss_indexes/proforma_faiss_index"
folder_path = "proforma_invoice"
faiss_index, docstore, mapping = load_faiss_index(bucket_name, index_path)

# Streamlit app interface
st.title("Proforma Invoice Chatbot & PDF Monitor")

# Section 1: Query Interface
st.header("Query Proforma Invoices")
user_query = st.text_input("Enter your query:")
if user_query:
    start_time = datetime.now()

    query_embedding = get_embeddings([user_query])
    results = query_faiss_index(faiss_index, docstore, query_embedding)

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
