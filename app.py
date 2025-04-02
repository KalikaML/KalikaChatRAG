import streamlit as st
import boto3
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from datetime import datetime

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

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


# Initialize Gemini 1.5 Pro LLM
class GeminiLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, prompt, temperature=0.7, max_length=512):
        # Replace this with actual API calls to Gemini 1.5 Pro
        # Example: Make a POST request to Gemini's endpoint with the prompt and parameters.
        return f"Generated response for: {prompt}"  # Placeholder response


gemini_llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])

# Prompt template for sales team queries
prompt_template = PromptTemplate(
    input_variables=["documents", "question"],
    template="""
    You are an assistant designed to support a sales team. Using the provided information from proforma invoices and purchase orders, answer the user's question with accurate, concise, and actionable details in a well-structured bullet-point format. Ensure the response includes all relevant details requested by the user, covering every aspect of the question comprehensively. Do not include the raw data, source information, or this prompt in your response—only provide the relevant answer formatted as requested.
    Information: {documents}
    Question: {question}
    Answer in the following format:
    - [Relevant detail addressing the user's question]
    - [Additional relevant detail, if applicable]
    - [Further relevant detail, if applicable]
    (Include as many bullet points as necessary to fully answer the question)
    """
)


# Function to load FAISS index from S3
def load_faiss_index_from_s3(index_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_name in ["index.faiss", "index.pkl"]:
            s3_key = f"{index_path}{file_name}"
            local_path = os.path.join(temp_dir, file_name)
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
        vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    if 'Contents' not in response:
        return 0
    new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
    return new_files


# Main Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")

    # Custom CSS for black theme with additional styling for context panel
    st.markdown("""
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
        </style>
    """, unsafe_allow_html=True)

    # Create a two-column layout
    col1, col2 = st.columns([7, 3])

    # Right panel for context
    with col2:
        st.markdown("<h3>Retrieved Context</h3>", unsafe_allow_html=True)

        # Initialize context session state
        if "context_documents" not in st.session_state:
            st.session_state.context_documents = []

        # Display the context documents
        if st.session_state.context_documents:
            st.markdown('<div class="context-panel">', unsafe_allow_html=True)
            st.markdown('<div class="context-title">Source Documents</div>', unsafe_allow_html=True)
            for i, doc in enumerate(st.session_state.context_documents):
                st.markdown(f'<div class="context-item">Document {i + 1}:<br>{doc.page_content}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="context-panel">No context retrieved yet. Ask a question to see relevant documents.</div>',
                unsafe_allow_html=True)

        # Options and stats in the right panel below context
        st.markdown("<h3>Options</h3>", unsafe_allow_html=True)
        option = st.radio("Select Data Source", ("Proforma Invoices", "Purchase Orders"))

        st.markdown("<h3>Stats</h3>", unsafe_allow_html=True)
        proforma_new = count_new_files(PROFORMA_FOLDER)
        po_new = count_new_files(PO_FOLDER)
        st.write(f"Proforma Invoices: {proforma_new} new files")
        st.write(f"Purchase Orders: {po_new} new files")
        st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Main chat interface in the left column
    with col1:
        st.title("RAG Chatbot")
        st.write("Ask anything based on the selected data source.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "current_option" not in st.session_state:
            st.session_state.current_option = None

        if "qa_chain" not in st.session_state or option != st.session_state.current_option:
            st.session_state.current_option = option
            index_path = PROFORMA_INDEX_PATH if option == "Proforma Invoices" else PO_INDEX_PATH
            with st.spinner(f"Loading {option} FAISS index from S3..."):
                vector_store = load_faiss_index_from_s3(index_path)
                retriever = vector_store.as_retriever()

                def gemini_chain_run(question):
                    # Store the retrieved documents in session state for display in right panel
                    documents = retriever.retrieve(question)
                    st.session_state.context_documents = documents

                    # Generate response using the documents
                    prompt_instance = prompt_template.format(documents=documents, question=question)
                    return gemini_llm.generate(prompt_instance)

                st.session_state.qa_chain_run = gemini_chain_run

        user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")

        if st.button("Send") and user_input:
            with st.spinner("Generating response..."):
                response = st.session_state.qa_chain_run(user_input)

                # Append chat history (only user question and bot response)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response))

                # Trigger a rerun to update the context panel
                st.experimental_rerun()

        # Display chat history
        for sender, message in reversed(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()