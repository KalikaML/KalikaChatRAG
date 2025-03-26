'''import streamlit as st
import schedule
import time
import threading
import toml
import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, update_faiss_index_from_emails, fetch_faiss_index_from_s3

# Load secrets from secrets.toml
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
#secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = st.secrets["api_token"]

new_files_count = 0

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    #schedule.every().day.at("19:35").do(update_faiss_index_from_emails)
    def update_and_reload_index():
        global new_files_count
        # Update FAISS index from emails and get count of new files added
        new_files_count = update_faiss_index_from_emails()
        # Reload FAISS index locally after update
        fetch_faiss_index_from_s3()
        get_faiss_index()

    schedule.every(24).hours.do(update_and_reload_index)
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

def query_proforma_rag(query):
    """Query the RAG model using the local FAISS index."""
    vector_store = get_faiss_index()
    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update.", "Unknown"

    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = chain.run(query)

    # Get the source file from the retriever (simplified assumption: first retrieved doc)
    sources = retriever.get_relevant_documents(query)
    source_file = sources[0].metadata.get("source", "Unknown") if sources else "Unknown"

    return response, source_file

# Fetch FAISS index from S3 and initialize locally on startup
fetch_faiss_index_from_s3()
get_faiss_index()

# Start the scheduler
schedule_faiss_update()

# Streamlit UI Setup
st.title("Chatbot for Proforma Invoice Analysis")

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chat Section
st.header("Chat with the Bot")

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

    # Get bot response using query_proforma_rag function
    bot_response, _ = query_proforma_rag(user_query)  # Ignore source_file

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Display bot response immediately in the chat interface without context
    st.chat_message("bot").write(bot_response)

'''

import streamlit as st
import schedule
import time
import threading
import toml
import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, update_faiss_index_from_emails, fetch_faiss_index_from_s3

# Load secrets from secrets.toml
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
#secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = st.secrets["api_token"]

# Global variable to track new files added during updates
new_files_count = 0


# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""

    def update_and_reload_index():
        global new_files_count
        # Update FAISS index from emails and get count of new files added
        new_files_count = update_faiss_index_from_emails()
        # Reload FAISS index locally after update
        fetch_faiss_index_from_s3()
        get_faiss_index()

    schedule.every(24).hours.do(update_and_reload_index)


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()


# Query function
def query_proforma_rag(query):
    """Query the RAG model using the local FAISS index."""
    vector_store = get_faiss_index()
    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update.", "Unknown"

    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = chain.run(query)

    # Get the source file from the retriever (simplified assumption: first retrieved doc)
    sources = retriever.get_relevant_documents(query)
    source_file = sources[0].metadata.get("source", "Unknown") if sources else "Unknown"

    return response, source_file


# Fetch FAISS index from S3 and initialize locally on startup
fetch_faiss_index_from_s3()
get_faiss_index()

# Start the scheduler
schedule_faiss_update()

# Streamlit UI Setup
st.title("Chatbot for Proforma Invoice Analysis")

# Chat Section
st.header("Chat with the Bot")

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

    # Get bot response using query_proforma_rag function
    bot_response, _ = query_proforma_rag(user_query)  # Ignore source_file

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Display bot response immediately in the chat interface without context
    st.chat_message("bot").write(bot_response)

# Section: Total Files and New Files Added During Updates
st.header("Proforma Invoice Statistics")

# Display total number of files indexed in FAISS
vector_store = get_faiss_index()
if vector_store:
    total_files = len(vector_store.index_to_docstore_id.keys())
else:
    total_files = 0

st.write(f"Total Number of Files Indexed: {total_files}")

# Display count of new files added during last scheduler update
st.write(f"New Files Added During Last Update: {new_files_count}")
