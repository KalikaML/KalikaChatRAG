import streamlit as st
import schedule
import time
import threading
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, update_faiss_index_from_emails, fetch_faiss_index_from_s3
from s3_uploader import get_s3_file_count

# Access secrets from Streamlit Cloud environment via st.secrets
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

# Initialize file counts in session state
if 'total_files' not in st.session_state:
    st.session_state.total_files = get_s3_file_count()  # Initial count from S3 bucket
if 'new_files' not in st.session_state:
    st.session_state.new_files = 0  # Track new files added during update
if 'last_update_message' not in st.session_state:
    st.session_state.last_update_message = "Waiting for first update..."


# Scheduler setup to update FAISS index and file counts periodically
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    schedule.every(24).hours.do(faiss_update_job)


def faiss_update_job():
    """Job to update FAISS index and track new files."""
    initial_count = get_s3_file_count()
    update_faiss_index_from_emails()
    final_count = get_s3_file_count()

    # Calculate new files added during the update process
    new_files = final_count - initial_count

    # Update session state values for UI display
    st.session_state.new_files = new_files
    st.session_state.total_files = final_count  # Update total count after adding new files

    # Set last update message for user feedback
    st.session_state.last_update_message = f"Updated! {new_files} new files added. Total files: {final_count}"

    # Force Streamlit app to re-render with updated values
    st.experimental_rerun()


def run_scheduler():
    """Run scheduler in a separate thread."""
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)


def query_proforma_rag(query):
    """Query the RAG model using the local FAISS index."""
    vector_store = get_faiss_index()

    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update.", "Unknown"

    retriever = vector_store.as_retriever()

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    )

    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = chain.run(query)

    sources = retriever.get_relevant_documents(query)

    source_file = sources[0].metadata.get("source", "Unknown") if sources else "Unknown"

    return response, source_file


# Fetch FAISS index from S3 on startup and initialize locally if needed
if fetch_faiss_index_from_s3():
    print("Successfully loaded FAISS index from S3.")
else:
    print("Failed to load FAISS index from S3. Will attempt to create it from new emails.")

# Start scheduler thread for periodic updates of FAISS index and file counts
schedule_faiss_update()
scheduler_thread.start()

# Streamlit UI Setup for Chatbot and Statistics Display
st.title("Chatbot for Proforma Invoice Analysis")

# Sidebar Section: File Statistics Display
st.sidebar.header("Proforma Invoice Statistics")
st.sidebar.metric("Total Proforma Files in S3", st.session_state.total_files)
st.sidebar.write(st.session_state.last_update_message)

# Chat Section: User Interaction with Chatbot
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.header("Chat with the Bot")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])

    elif message["role"] == "bot":
        st.chat_message("bot").write(message["content"])

if user_query := st.chat_input("Type your question here..."):
    # Add user query to chat history and process it using RAG model query function.

    st.session_state.messages.append({"role": "user", "content": user_query})

    bot_response, _ = query_proforma_rag(user_query)

    # Add bot response to chat history and display it.

    st.session_state.messages.append({"role": "bot", "content": bot_response})

