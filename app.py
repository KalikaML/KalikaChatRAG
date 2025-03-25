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

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    schedule.every().day.at("19:35").do(update_faiss_index_from_emails)

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

