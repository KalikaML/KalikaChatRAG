import streamlit as st
import schedule
import time
import threading
import toml
import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, update_faiss_index_from_emails, fetch_faiss_index_from_s3, get_faiss_stats
from s3_uploader import get_s3_file_count

# Load secrets from secrets.toml
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
#secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = st.secrets["api_token"]

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = None
if 's3_file_count' not in st.session_state:
    st.session_state['s3_file_count'] = 0
if 'newly_indexed_files' not in st.session_state:
    st.session_state['newly_indexed_files'] = []

# Function to update FAISS index and S3 count
def update_faiss_and_s3_count():
    st.session_state['newly_indexed_files'] = update_faiss_index_from_emails()
    st.session_state['s3_file_count'] = get_s3_file_count()
    st.session_state['faiss_index'] = get_faiss_index()

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    schedule.every(24).hours.do(update_faiss_and_s3_count)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Load FAISS index on startup
with st.spinner("Loading FAISS index..."):
    if not st.session_state['faiss_index']:
        fetch_faiss_index_from_s3()
        st.session_state['faiss_index'] = get_faiss_index()

# Sidebar for history
with st.sidebar:
    st.header("Query History")
    for query, answer in reversed(st.session_state['history']):  # Reverse for chronological order
        st.markdown(f"**Q:** {query}")
        st.markdown(f"**A:** {answer}")
        st.write("---")

# Main app
st.title("Proforma Invoice RAG")

# S3 file count display
st.write(f"Total Proforma Invoices in S3: {st.session_state['s3_file_count']}")

# Display newly indexed files
if st.session_state['newly_indexed_files']:
    st.success(f"Newly indexed files: {', '.join(st.session_state['newly_indexed_files'])}")

# FAISS update scheduler button
if st.button("Update FAISS Index"):
    with st.spinner("Updating FAISS index..."):
        update_faiss_and_s3_count()
        st.success("FAISS index updated!")

# Query input
query = st.text_input("Enter your query:")

# Question suggestions
if query:
    suggested_questions = [
        f"Tell me about {query}?",
        f"Explain {query} in detail?",
        f"What are the key aspects of {query}?",
        f"Give me a summary of {query}."
    ]
    st.write("Suggested questions:")
    for suggestion in suggested_questions:
        st.markdown(f"- {suggestion}")

if query:
    vector_store = get_faiss_index()

    if not vector_store:
        st.warning("FAISS index not available. Please wait for initialization or the next scheduled update.")
    else:
        retriever = vector_store.as_retriever()
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"max_new_tokens": 512},
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
        )
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = qa.run(query)
        st.write("Answer:", answer)

        # Update history
        st.session_state['history'].append((query, answer))

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
