import streamlit as st
import boto3
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from datetime import datetime
import threading
import time
import re  # Import the regular expression module

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Load secrets from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Initialize embeddings - use cache for better performance
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

embeddings = get_embeddings()

# Mock Gemini LLM with better extraction and follow-up suggestions
class GeminiLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, prompt, temperature=0.7, max_length=None):
        # Mock response with bullet points and follow-up questions
        response_text = f"""
        - Detail 1: This is a detail extracted from the context.
        - Detail 2: Another key piece of information.
        - Detail 3: More relevant data.

        Follow-up Questions:
        1. What is the status of the order?
        2. Can you provide more details on pricing?
        3. What are the delivery dates?
        """
        return response_text

    def extract_response(self, response_text):
        """Extract bullet point answers from the response."""
        bullet_pattern = r"^- (.*)$"
        bullets = re.findall(bullet_pattern, response_text, re.MULTILINE)
        return bullets

    def extract_followup_questions(self, response_text):
        """Extract follow-up questions from the response."""
        question_pattern = r"^\d+\. (.*)\?$"
        questions = re.findall(question_pattern, response_text, re.MULTILINE)
        return questions

@st.cache_resource
def get_llm():
    return GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])

gemini_llm = get_llm()

# Enhanced prompt template for sales team queries with document count information
prompt_template = PromptTemplate(
    input_variables=["documents", "question", "doc_count"],
    template="""
    You are an assistant designed to support a sales team. Using the provided information from {doc_count} documents, answer the user's question with accurate, concise, and actionable details in a well-structured bullet-point format.

    Make your response as comprehensive as needed to fully address the query - don't artificially limit length. Provide follow-up questions that would help the user get more specific information.

    Information from documents: {documents}
    Question: {question}

    Important: Your response must ONLY include the answer in bullet points and a list of follow-up questions. Do NOT include:
    - The documents or source information you used
    - Any preamble or introduction to your answer
    - Any mention of the context you're referencing
    - This instruction itself

    Respond directly with bullet points and follow-up questions:
    - [Relevant detail addressing the user's question]
    - [Additional relevant detail, if applicable]
    - [Further relevant detail, if applicable]

    Follow-up Questions:
    1. [Suggested question 1]
    2. [Suggested question 2]
    3. [Suggested question 3]
    (Include as many bullet points and follow-up questions as necessary)
    """
)

# Function to load FAISS index from S3 - cached for performance
@st.cache_resource
def load_faiss_index_from_s3(index_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Preload both indexes at startup to avoid loading delays
@st.cache_resource
def get_all_indexes():
    proforma_index = load_faiss_index_from_s3(PROFORMA_INDEX_PATH)
    po_index = load_faiss_index_from_s3(PO_INDEX_PATH)
    return {
        "Proforma Invoices": proforma_index,
        "Purchase Orders": po_index,
        "Combined": None  # Will be created when needed
    }

# Function to create a combined index from both sources
def get_combined_index(proforma_index, po_index):
    # This is a placeholder implementation
    # Depending on your FAISS setup, you might need to:
    # 1. Merge the indexes programmatically, or
    # 2. Run combined queries against both and merge results
    # For now, we'll return a simple dict to be used for dual querying
    return {"proforma": proforma_index, "po": po_index}

# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    if 'Contents' not in response:
        return 0
    new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
    return new_files

# Function to retrieve documents from combined indexes
def retrieve_from_all_indexes(query, k=10):
    """Retrieve documents from both indexes and combine results"""
    proforma_index = st.session_state.all_indexes["Proforma Invoices"]
    po_index = st.session_state.all_indexes["Purchase Orders"]

    # Retrieve from both indexes
    proforma_docs = proforma_index.similarity_search(query, k=k // 2)
    po_docs = po_index.similarity_search(query, k=k // 2)

    # Combine results
    all_docs = proforma_docs + po_docs

    # Sort by relevance if there's a relevance score
    # This is placeholder logic - implement actual relevance sorting based on your needs
    if hasattr(all_docs[0], 'metadata') and 'score' in all_docs[0].metadata:
        all_docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

    return all_docs

# Background thread to periodically refresh file counts
def background_refresh():
    while True:
        try:
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time.sleep(300)  # Refresh every 5 minutes
        except Exception as e:
            print(f"Error in background refresh: {e}")
            time.sleep(60)  # Retry after a minute if there's an error

# Custom styling
def load_css():
    return """
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .chat-message {
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            font-size: 16px;
        }
        .user-message {
            background-color: #2D2D2D;
            color: #BB86FC;
            text-align: right;
            border: 1px solid #BB86FC;
        }
        .bot-message {
            background-color: #333333;
            color: #E0E0E0;
            text-align: left;
            border: 1px solid #03DAC6;
        }
        .sidebar .sidebar-content {
            background-color: #252525;
            padding: 20px;
            color: #E0E0E0;
        }
        .context-panel {
            background-color: #252525;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #555555;
            margin-top: 10px;
        }
        .context-title {
            color: #03DAC6;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .context-item {
            background-color: #333333;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 8px;
            font-size: 14px;
            border-left: 3px solid #BB86FC;
        }
        .document-source {
            font-size: 12px;
            color: #BB86FC;
            margin-bottom: 5px;
        }
        .stTextInput > div > div > input {
            background-color: #333333;
            color: #E0E0E0;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        .stButton > button {
            background-color: #BB86FC;
            color: #1E1E1E;
            border: none;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #03DAC6;
            color: #1E1E1E;
        }
        h1, h2, h3 {
            color: #BB86FC;
        }
        .stSpinner > div > div {
            color: #03DAC6;
        }
        .loading-message {
            background-color: #2D2D2D;
            color: #03DAC6;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 20px 0;
        }
        .doc-stats {
            background-color: #252525;
            padding: 8px 12px;
            border-radius: 4px;
            display: inline-block;
            margin-right: 10px;
            border-left: 3px solid #03DAC6;
        }
        .followup-question {
            background-color: #333333;
            color: #E0E0E0;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            cursor: pointer;
        }
        .followup-question:hover {
            background-color: #444444;
        }
        </style>
    """

# Main Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.markdown(load_css(), unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "context_documents" not in st.session_state:
        st.session_state.context_documents = []
    if "indexes_loaded" not in st.session_state:
        st.session_state.indexes_loaded = False
    if "proforma_new" not in st.session_state:
        st.session_state.proforma_new = 0
    if "po_new" not in st.session_state:
        st.session_state.po_new = 0
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if "background_thread_started" not in st.session_state:
        st.session_state.background_thread_started = False
    if "response_length" not in st.session_state:
        st.session_state.response_length = "Auto"  # Default to auto length
    if "followup_question" not in st.session_state:
        st.session_state.followup_question = None

    # Start background thread for file count updates
    if not st.session_state.background_thread_started:
        thread = threading.Thread(target=background_refresh, daemon=True)
        thread.start()
        st.session_state.background_thread_started = True

    # Create a two-column layout
    col1, col2 = st.columns([7, 3])

    # Main chat interface in the left column
    with col1:
        st.title("RAG Chatbot")
        st.write("Ask anything related to proforma invoices and purchase orders.")

        # Load indexes in the background if not already loaded
        if not st.session_state.indexes_loaded:
            with st.spinner("Loading FAISS indexes..."):
                st.session_state.all_indexes = get_all_indexes()
                st.session_state.indexes_loaded = True

        # Follow-up question handling
        if st.session_state.followup_question:
            user_input = st.session_state.followup_question
            st.session_state.followup_question = None  # Clear the follow-up question
        else:
            user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")

        # Options for query
        input_col1, input_col2, input_col3 = st.columns([2, 2, 1])

        with input_col1:
            option = st.selectbox(
                "Data Source",
                ["Combined (Both Sources)", "Proforma Invoices Only", "Purchase Orders Only"],
                index=0
            )

        with input_col2:
            doc_count = st.slider("Number of documents to retrieve", min_value=3, max_value=20, value=10)

        with input_col3:
            response_length = st.selectbox(
                "Response Length",
                ["Auto", "Concise", "Detailed"],
                index=0
            )
            st.session_state.response_length = response_length

        # Function to display follow-up questions
        def display_followup_questions(questions):
            st.write("Suggested Follow-up Questions:")
            for question in questions:
                if st.button(question, key=f"followup_{question}"):
                    st.session_state.followup_question = question
                    st.experimental_rerun()

        if st.button("Send") and user_input:
            # Append user question to chat history immediately for better UX
            st.session_state.chat_history.append(("user", user_input))

            # Create a placeholder for the bot's response
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="loading-message">Retrieving documents and generating response...</div>',
                unsafe_allow_html=True
            )

            try:
                # Determine which index to use based on user selection
                if option == "Combined (Both Sources)":
                    # Use both indexes
                    documents = retrieve_from_all_indexes(user_input, k=doc_count)
                elif option == "Proforma Invoices Only":
                    vector_store = st.session_state.all_indexes["Proforma Invoices"]
                    documents = vector_store.similarity_search(user_input, k=doc_count)
                else:  # Purchase Orders Only
                    vector_store = st.session_state.all_indexes["Purchase Orders"]
                    documents = vector_store.similarity_search(user_input, k=doc_count)

                st.session_state.context_documents = documents

                # Calculate max_length based on selected response length
                max_length = None  # Default: no limit (Auto)
                if response_length == "Concise":
                    max_length = 300  # Shorter response
                elif response_length == "Detailed":
                    max_length = 2000  # Longer, more detailed response

                # Generate response using the documents
                prompt_instance = prompt_template.format(
                    documents=documents,
                    question=user_input,
                    doc_count=len(documents)
                )

                response_text = gemini_llm.generate(prompt_instance, max_length=max_length)
                bullets = gemini_llm.extract_response(response_text)
                followup_questions = gemini_llm.extract_followup_questions(response_text)

                # Add document source information to the response for context
                doc_sources = {}
                for doc in documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = doc.metadata['source']
                        if source in doc_sources:
                            doc_sources[source] += 1
                        else:
                            doc_sources[source] = 1

                source_info = "Based on: "
                for source, count in doc_sources.items():
                    source_info += f"{source.split('/')[-1]} ({count}), "
                source_info = source_info.rstrip(", ")

                # Update chat history with bot response including document stats
                st.session_state.chat_history.append(("bot_metadata", f"Using {len(documents)} documents"))
                st.session_state.chat_history.append(("bot", bullets))

                # Clear the placeholder
                response_placeholder.empty()

                # Display follow-up questions
                display_followup_questions(followup_questions)

                # Force a rerun to update the UI
                st.experimental_rerun()

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)

        # Display chat history
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            elif role == "bot":
                if isinstance(message, list):
                    st.markdown('<div class="chat-message bot-message"><ul>' +
                                ''.join(f'<li>{item}</li>' for item in message) +
                                '</ul></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)
            elif role == "bot_metadata":
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)

    # Context panel in the right column
    with col2:
        st.sidebar.header("Context Panel")
        st.sidebar.markdown(f"""
            <div class="context-panel">
            <h3 class="context-title">Document Stats</h3>
            <div class="context-item">
                <strong>New Proforma Invoices:</strong> {st.session_state.proforma_new}
            </div>
            <div class="context-item">
                <strong>New Purchase Orders:</strong> {st.session_state.po_new}
            </div>
            <div class="context-item">
                <strong>Last Updated:</strong> {st.session_state.last_updated}
            </div>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.context_documents:
            st.sidebar.markdown('<div class="context-panel"><h3 class="context-title">Relevant Documents</h3>',
                                unsafe_allow_html=True)
            for doc in st.session_state.context_documents:
                source = doc.metadata.get('source', 'N/A').split('/')[-1] if hasattr(doc, 'metadata') else 'N/A'
                page = doc.metadata.get('page', 'N/A') if hasattr(doc, 'metadata') else 'N/A'
                content = doc.page_content[:100] + "..." if hasattr(doc, 'page_content') else 'No Content'
                st.sidebar.markdown(f"""
                    <div class="context-item">
                    <div class="document-source">Source: {source}, Page: {page}</div>
                    {content}
                    </div>
                    """, unsafe_allow_html=True)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
