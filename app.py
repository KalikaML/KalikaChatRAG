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
SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = secrets["api_token"]

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize FAISS index and S3 file count in session state
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = None
    fetch_faiss_index_from_s3() # Load FAISS index at startup
    st.session_state['faiss_index'] = get_faiss_index()

if 's3_file_count' not in st.session_state:
    st.session_state['s3_file_count'] = get_s3_file_count()

def update_faiss_and_s3_count():
    """Update FAISS index and S3 file count, and store indexed files."""
    newly_indexed = update_faiss_index_from_emails()
    st.session_state['s3_file_count'] = get_s3_file_count()
    st.session_state['faiss_index'] = get_faiss_index()
    return newly_indexed

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    schedule.every(24).hours.do(update_faiss_and_s3_count)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Sidebar for history
with st.sidebar:
    st.header("Query History")
    for query, answer in st.session_state['history']:
        st.markdown(f"**Q:** {query}")
        st.markdown(f"**A:** {answer}")
        st.write("---")

# Main app
st.title("Proforma Invoice RAG")

# S3 file count display
s3_count = st.session_state['s3_file_count']
st.write(f"Total Proforma Invoices in S3: {s3_count}")

# FAISS update scheduler
if st.button("Run Scheduler"):
    newly_indexed_files = update_faiss_and_s3_count() # Run the update and get indexed files

    if newly_indexed_files:
        st.success(f"FAISS index updated. New files indexed: {', '.join(newly_indexed_files)}")
    else:
        st.info("No new files were indexed during this update.")

# Query input
query = st.text_input("Enter your query:")

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
        st.session_state['history'].insert(0, (query, answer))
# Suggestion according to the enter query.
        suggested_questions = [
        f"Tell me about {query} ?",
        f"Explain about {query}?",
        f"Describe {query}?"
            ]
        st.write("Suggestion:")
        for sugges in suggested_questions:
            st.markdown(f"- {sugges}")

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
