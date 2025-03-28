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
from langchain.prompts import PromptTemplate

# Load secrets from secrets.toml
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
#secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = st.secrets["api_token"]

# Global variable to track new files added during updates
new_files_count = 0
new_file_names = []  # List to store new file names

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    def update_and_reload_index():
        global new_files_count, new_file_names
        # Update FAISS index from emails and get count and names of new files added
        new_files_count, new_file_names = update_faiss_index_from_emails()
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

### UPDATED SECTION: Suggestion Generation ###
def generate_suggested_questions(new_files):
    """Generates suggested questions based on the names of new files."""
    suggested_questions = []
    for file_name in new_files:
        # Example questions - customize these based on your data
        suggested_questions.append(f"What is the total amount in {file_name}?")
        suggested_questions.append(f"Who is the supplier in {file_name}?")
        suggested_questions.append(f"What is the invoice number in {file_name}?")
    return suggested_questions

### UPDATED SECTION: Professional RAG Query Function ###
def query_proforma_rag(query):
    """Query the RAG model and return only the answer in structured format."""
    vector_store = get_faiss_index()
    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update."

    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )

    # Custom prompt to return only the answer
    prompt_template = """You are an expert Proforma invoice analyst. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you do not know, don't try to make up an answer.
    Your goal is to provide concise, accurate, and professional answers. You should not include any context or explanation in your responses.
    Just output the direct answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False  # Do not return source documents
    )

    result = chain({"query": query})
    return result["result"].strip()

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

### UPDATED SECTION: Display suggested questions ###
if new_file_names:
    st.subheader("Suggested Questions Based on New Invoices:")
    suggested_questions = generate_suggested_questions(new_file_names)
    for question in suggested_questions:
        st.markdown(f"- *{question}*")

# User input for chat interaction
if user_query := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get bot response using query_proforma_rag function
    bot_response = query_proforma_rag(user_query)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Display bot response immediately in the chat interface
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
'''
import streamlit as st
import schedule
import time
import threading
import toml
import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, update_faiss_index_from_emails, get_last_updated_file_count
from s3_uploader import get_s3_file_count

# Load secrets from secrets.toml
#SECRETS_FILE_PATH = os.path.join(os.getcwd(), "secrets.toml")
#secrets = toml.load(SECRETS_FILE_PATH)
HUGGINGFACE_API_TOKEN = st.secrets["api_token"]

# Scheduler setup
def schedule_faiss_update():
    """Schedule FAISS index update every 24 hours."""
    schedule.every(24).hours.do(update_faiss_index_from_emails)

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

# Initialize FAISS index on startup
get_faiss_index()

# Start the scheduler
schedule_faiss_update()

# Streamlit UI Setup
st.title("Chatbot for Proforma Invoice Analysis")

# Sidebar for Status and Information
with st.sidebar:
    st.header("Proforma Invoice Status")
    total_files = get_s3_file_count()
    st.write(f"Total Proforma Files in S3: {total_files}")

    last_updated_count = get_last_updated_file_count()
    st.write(f"FAISS Index Last Updated with: {last_updated_count} files")

    if total_files > last_updated_count:
        st.warning("New Proforma files detected! The FAISS index might be outdated.")
        if st.button("Update FAISS Index"):
            update_faiss_index_from_emails()
            st.success("FAISS Index is being updated. Please refresh after a while.")
    else:
        st.success("FAISS Index is up to date!")

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
