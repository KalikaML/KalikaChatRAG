import streamlit as st
import boto3
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import threading
import time

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"  # Upgraded model for better embeddings
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
)


# Initialize embeddings with better configuration
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Changed to True for better similarity
    )


embeddings = get_embeddings()


# Gemini LLM implementation
class GeminiLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, prompt, temperature=0.5, max_length=None):
        # Placeholder - replace with actual Gemini API implementation
        return f"Generated response for: {prompt}"


@st.cache_resource
def get_llm():
    return GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])


gemini_llm = get_llm()

# Enhanced prompt template
prompt_template = PromptTemplate(
    input_variables=["documents", "question", "doc_count"],
    template="""
    You are a sales team assistant. Using information from {doc_count} documents (proforma invoices and purchase orders), 
    provide a comprehensive, detailed answer to the user's question in bullet-point format. Include all relevant details 
    without omission.

    Documents: {documents}
    Question: {question}

    - [Detail 1]
    - [Detail 2]
    - [Detail 3]
    (Continue with all relevant details)
    """
)


# Improved FAISS index loading with document counting
@st.cache_resource
def load_faiss_index_from_s3(index_path, folder_prefix):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download index files
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)

        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)

        # Get document count from S3 folder
        doc_count = count_processed_files(folder_prefix)
        return vector_store, doc_count


# Count processed files
def count_processed_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    return sum(1 for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf'))


# Count new (unprocessed) files
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    return sum(1 for obj in response.get('Contents', []) if not obj['Key'].endswith('_processed.pdf'))


# Load all indexes with document counts
@st.cache_resource
def get_all_indexes():
    proforma_index, proforma_count = load_faiss_index_from_s3(PROFORMA_INDEX_PATH, PROFORMA_FOLDER)
    po_index, po_count = load_faiss_index_from_s3(PO_INDEX_PATH, PO_FOLDER)
    return {
        "Proforma Invoices": {"index": proforma_index, "count": proforma_count},
        "Purchase Orders": {"index": po_index, "count": po_count}
    }


# Enhanced document retrieval
def retrieve_documents(query, index_info, k=15):  # Increased k for more comprehensive results
    vector_store = index_info["index"]
    # Get more documents and sort by relevance
    docs = vector_store.similarity_search_with_score(query, k=k)
    # Sort by score (lower is better) and return all documents
    sorted_docs = sorted(docs, key=lambda x: x[1])
    return [doc for doc, score in sorted_docs]


# Background refresh for stats
def background_refresh():
    while True:
        try:
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time.sleep(300)
        except Exception as e:
            print(f"Error in background refresh: {e}")
            time.sleep(60)


# CSS styling (unchanged from original)
def load_css():
    return """
        <style>
        /* ... (keeping your original CSS) ... */
        </style>
    """


# Main app
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

    # Start background thread
    if not st.session_state.background_thread_started:
        thread = threading.Thread(target=background_refresh, daemon=True)
        thread.start()
        st.session_state.background_thread_started = True

    col1, col2 = st.columns([7, 3])

    with col1:
        st.title("RAG Chatbot")
        st.write("Ask about proforma invoices and purchase orders.")

        if not st.session_state.indexes_loaded:
            with st.spinner("Loading FAISS indexes..."):
                st.session_state.all_indexes = get_all_indexes()
                st.session_state.indexes_loaded = True

        user_input = st.text_input("Your Question:", key="input")

        input_col1, input_col2 = st.columns([2, 1])
        with input_col1:
            option = st.selectbox(
                "Data Source",
                ["Combined (Both Sources)", "Proforma Invoices Only", "Purchase Orders Only"]
            )
        with input_col2:
            doc_count = st.slider("Documents to retrieve", 5, 30, 15)

        if st.button("Send") and user_input:
            st.session_state.chat_history.append(("user", user_input))
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="loading-message">Processing...</div>',
                unsafe_allow_html=True
            )

            try:
                # Retrieve documents based on selection
                if option == "Combined (Both Sources)":
                    proforma_docs = retrieve_documents(user_input, st.session_state.all_indexes["Proforma Invoices"],
                                                       k=doc_count // 2)
                    po_docs = retrieve_documents(user_input, st.session_state.all_indexes["Purchase Orders"],
                                                 k=doc_count // 2)
                    documents = proforma_docs + po_docs
                elif option == "Proforma Invoices Only":
                    documents = retrieve_documents(user_input, st.session_state.all_indexes["Proforma Invoices"],
                                                   k=doc_count)
                else:
                    documents = retrieve_documents(user_input, st.session_state.all_indexes["Purchase Orders"],
                                                   k=doc_count)

                st.session_state.context_documents = documents

                # Generate comprehensive response
                prompt_instance = prompt_template.format(
                    documents="\n".join([doc.page_content for doc in documents]),
                    question=user_input,
                    doc_count=len(documents)
                )

                response = gemini_llm.generate(prompt_instance, temperature=0.5,
                                               max_length=4000)  # Increased max_length

                st.session_state.chat_history.append(("bot_metadata", f"Using {len(documents)} documents"))
                st.session_state.chat_history.append(("bot", response))
                response_placeholder.empty()
                st.experimental_rerun()

            except Exception as e:
                st.session_state.chat_history.append(("error", f"Error: {str(e)}"))
                response_placeholder.empty()
                st.experimental_rerun()

        # Display chat history
        for sender, message in reversed(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot_metadata":
                st.markdown(f'<div class="doc-stats">{message}</div>', unsafe_allow_html=True)
            elif sender == "bot":
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)
            elif sender == "error":
                st.error(message)

    with col2:
        st.markdown("<h3>Context & Stats</h3>", unsafe_allow_html=True)

        if st.session_state.context_documents:
            st.markdown(f'<div class="context-panel">Documents ({len(st.session_state.context_documents)})</div>',
                        unsafe_allow_html=True)
            for i, doc in enumerate(st.session_state.context_documents):
                source_type = "Proforma" if PROFORMA_FOLDER in doc.metadata.get('source', '') else "PO"
                st.markdown(
                    f'<div class="document-source">Doc {i + 1} - {source_type}</div>' +
                    f'<div class="context-item">{doc.page_content[:500]}...</div>',
                    unsafe_allow_html=True
                )

        st.write(
            f"Proforma: {st.session_state.all_indexes['Proforma Invoices']['count']} total, {st.session_state.proforma_new} new")
        st.write(f"PO: {st.session_state.all_indexes['Purchase Orders']['count']} total, {st.session_state.po_new} new")
        st.write(f"Last Updated: {st.session_state.last_updated}")

        if st.button("Refresh Stats"):
            st.session_state.proforma_new = count_new_files(PROFORMA_FOLDER)
            st.session_state.po_new = count_new_files(PO_FOLDER)
            st.session_state.last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.experimental_rerun()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.context_documents = []
            st.experimental_rerun()


if __name__ == "__main__":
    main()