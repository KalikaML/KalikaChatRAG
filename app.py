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


# Initialize Gemini LLM with proper API implementation
class GeminiLLM:
    def __init__(self, api_key):
        self.api_key = api_key
        # Import inside the class to handle potential import errors gracefully
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
            self.initialized = True
        except ImportError:
            st.error("Failed to import Google Generative AI. Please install with: pip install google-generativeai")
            self.initialized = False
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            self.initialized = False

    def generate(self, prompt, temperature=0.7, max_length=None):
        """
        Generate response with Gemini AI
        """
        if not self.initialized:
            return "Error: Gemini LLM not properly initialized."

        try:
            generation_config = {
                "temperature": temperature,
            }
            if max_length:
                generation_config["max_output_tokens"] = max_length

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)}"


# Cache the LLM to avoid reinitializing
@st.cache_resource
def get_llm():
    try:
        return GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")

        # Return a dummy LLM as fallback
        class DummyLLM:
            def generate(self, prompt, **kwargs):
                return "Error: Could not initialize the LLM. Please check your API key and network connection."

        return DummyLLM()


# Enhanced prompt template for sales team queries with document count information
prompt_template = PromptTemplate(
    input_variables=["documents", "question", "doc_count"],
    template="""
    You are an assistant designed to support a sales team. Using the provided information from {doc_count} documents (including proforma invoices and purchase orders), answer the user's question with accurate, concise, and actionable details in a well-structured bullet-point format.

    Make your response as comprehensive as needed to fully address the query - don't artificially limit length.

    Information from documents: {documents}
    Question: {question}

    Important: Your response must ONLY include the answer in bullet points. Do NOT include:
    - The documents or source information you used
    - Any preamble or introduction to your answer
    - Any mention of the context you're referencing
    - This instruction itself

    Respond directly with bullet points:
    - [Relevant detail addressing the user's question]
    - [Additional relevant detail, if applicable]
    - [Further relevant detail, if applicable]
    (Include as many bullet points as necessary to fully answer the question)
    """
)


# Function to load FAISS index from S3 - cached for performance
@st.cache_resource
def load_faiss_index_from_s3(index_path):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_name in ["index.faiss", "index.pkl"]:
                s3_key = f"{index_path}{file_name}"
                local_path = os.path.join(temp_dir, file_name)
                s3_client.download_file(S3_BUCKET, s3_key, local_path)
            vector_store = FAISS.load_local(temp_dir, get_embeddings(), allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        # Return a minimal working FAISS index as fallback
        return FAISS.from_texts(["Error loading index"], get_embeddings())


# Preload both indexes at startup to avoid loading delays
@st.cache_resource
def get_all_indexes():
    try:
        proforma_index = load_faiss_index_from_s3(PROFORMA_INDEX_PATH)
        po_index = load_faiss_index_from_s3(PO_INDEX_PATH)
        return {
            "Proforma Invoices": proforma_index,
            "Purchase Orders": po_index,
            "Combined": None  # Will be created when needed
        }
    except Exception as e:
        st.error(f"Error loading indexes: {str(e)}")
        return None


# Function to retrieve documents from combined indexes
def retrieve_from_all_indexes(query, k=10):
    """Retrieve documents from both indexes and combine results"""
    try:
        proforma_index = st.session_state.all_indexes["Proforma Invoices"]
        po_index = st.session_state.all_indexes["Purchase Orders"]

        # Retrieve from both indexes
        proforma_docs = proforma_index.similarity_search(query, k=k // 2)
        po_docs = po_index.similarity_search(query, k=k // 2)

        # Combine results
        all_docs = proforma_docs + po_docs

        # Sort by relevance if there's a relevance score
        if hasattr(all_docs[0], 'metadata') and 'score' in all_docs[0].metadata:
            all_docs.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)

        return all_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []


# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
        if 'Contents' not in response:
            return 0
        new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
        return new_files
    except Exception as e:
        st.error(f"Error counting files: {str(e)}")
        return 0


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
        .error-message {
            background-color: #2D2D2D;
            color: #CF6679;
            padding: 10px;
            border-radius: 5px;
            text-align: left;
            margin: 20px 0;
            border-left: 3px solid #CF6679;
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
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()

    # Start background thread for file count updates if not already started
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

        # User input area
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

                if not documents:
                    raise Exception("No relevant documents found.")

                st.session_state.context_documents = documents

                # Calculate max_length based on selected response length
                max_length = None  # Default: no limit (Auto)
                if response_length == "Concise":
                    max_length = 300  # Shorter response
                elif response_length == "Detailed":
                    max_length = 2000  # Longer, more detailed response

                # Create document content for the prompt
                docs_content = "\n\n".join([doc.page_content for doc in documents])

                # Generate response using the documents
                prompt_instance = prompt_template.format(
                    documents=docs_content,
                    question=user_input,
                    doc_count=len(documents)
                )

                # Use direct LLM interface instead of chain
                response = st.session_state.llm.generate(prompt_instance, max_length=max_length)

                # Add document source information to the response for context
                doc_sources = {}
                for doc in documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = doc.metadata['source'].split('/')[-1]
                        if source in doc_sources:
                            doc_sources[source] += 1
                        else:
                            doc_sources[source] = 1

                source_info = "Based on: "
                for source, count in doc_sources.items():
                    source_info += f"{source} ({count}), "
                source_info = source_info.rstrip(", ")

                # Update chat history with bot response including document stats
                st.session_state.chat_history.append(("bot_metadata", f"Using {len(documents)} documents"))
                st.session_state.chat_history.append(("bot", response))

                # Clear the placeholder
                response_placeholder.empty()

                # Force a rerun to update the UI
                st.experimental_rerun()

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.session_state.chat_history.append(("error", error_message))
                response_placeholder.empty()
                st.experimental_rerun()

        # Display chat history
        for idx, (sender, message) in enumerate(reversed(st.session_state.chat_history)):
            if sender == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot_metadata":
                st.markdown(f'<div class="doc-stats">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot":
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "error":
                st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

    # Right panel for context
    with col2:
        st.markdown("<h3>Retrieved Context</h3>", unsafe_allow_html=True)

        # Display the context documents
        if st.session_state.context_documents:
            st.markdown('<div class="context-panel">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="context-title">Source Documents ({len(st.session_state.context_documents)})</div>',
                unsafe_allow_html=True)

            for i, doc in enumerate(st.session_state.context_documents):
                # Determine source type (Proforma or PO)
                source_type = "Unknown"
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    if PROFORMA_FOLDER.lower() in doc.metadata['source'].lower():
                        source_type = "Proforma"
                    elif PO_FOLDER.lower() in doc.metadata['source'].lower():
                        source_type = "PO"

                st.markdown(
                    f'<div class="document-source">Document {i + 1} - {source_type}</div>' +
                    f'<div class="context-item">{doc.page_content}</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="context-panel">No context retrieved yet. Ask a question to see relevant documents.</div>',
                unsafe_allow_html=True)

        # Options and stats in the right panel below context
        st.markdown("<h3>Stats</h3>", unsafe_allow_html=True)
        st.write(f"Proforma Invoices: {st.session_state.proforma_new} new files")
        st.write(f"Purchase Orders: {st.session_state.po_new} new files")
        st.write(f"Last Updated: {st.session_state.last_updated}")

        # Add a refresh button for stats
        if st.button("Refresh Stats"):
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.experimental_rerun()

        # Add a clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.context_documents = []
            st.experimental_rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")