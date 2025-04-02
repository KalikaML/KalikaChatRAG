'''import streamlit as st
import boto3
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from datetime import datetime

# Configuration constants
S3_BUCKET = "kalika-rag"
PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
PO_INDEX_PATH = "faiss_indexes/po_faiss_index/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Replace with your preferred model
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Load secrets from secrets.toml
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]

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

# Initialize LLM (HuggingFace Hub)
llm = HuggingFaceHub(
    repo_id=LLM_MODEL,
    huggingfacehub_api_token=HUGGINGFACE_TOKEN,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

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

    # Custom CSS for black theme
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

    # Sidebar for options and stats
    with st.sidebar:
        st.title("Chatbot Options")
        option = st.radio("Select Data Source", ("Proforma Invoices", "Purchase Orders"))
        st.subheader("New Files Processed")
        proforma_new = count_new_files(PROFORMA_FOLDER)
        po_new = count_new_files(PO_FOLDER)
        st.write(f"Proforma Invoices: {proforma_new} new files")
        st.write(f"Purchase Orders: {po_new} new files")
        st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Chat interface
    st.title("RAG Chatbot")
    st.write("Ask anything based on the selected data source.")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_option" not in st.session_state:
        st.session_state.current_option = None

    # Load FAISS index and initialize qa_chain if not set or option changes
    if "qa_chain" not in st.session_state or option != st.session_state.current_option:
        st.session_state.current_option = option
        index_path = PROFORMA_INDEX_PATH if option == "Proforma Invoices" else PO_INDEX_PATH
        with st.spinner(f"Loading {option} FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(index_path)
            retriever = vector_store.as_retriever()  # Search entire FAISS index
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt_template,
                    "document_variable_name": "documents"
                }
            )

    # Chat input
    user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            response = st.session_state.qa_chain.run(user_input)
            # Append to chat history (latest entry added last)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", response))

    # Display chat history with latest at top, oldest at bottom
    for sender, message in reversed(st.session_state.chat_history):
        if sender == "user":
            st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()'''
import streamlit as st
import boto3
import os
import tempfile
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
S3_BUCKET = "kalika-rag"
INDEX_PATHS = {
    "Proforma Invoices": "faiss_indexes/proforma_faiss_index/",
    "Purchase Orders": "faiss_indexes/po_faiss_index/"
}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
PROFORMA_FOLDER = "proforma_invoice/"
PO_FOLDER = "PO_Dump/"

# Load secrets from secrets.toml
try:
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your secrets.toml configuration.")
    logger.error(f"Missing secret: {e}")
    st.stop()

# Initialize S3 client
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
    )
except Exception as e:
    st.error("Failed to initialize S3 client. Please check AWS credentials.")
    logger.error(f"S3 client initialization failed: {str(e)}")
    st.stop()

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
except Exception as e:
    st.error("Failed to initialize embeddings. Please check the model name or network connection.")
    logger.error(f"Embeddings initialization failed: {str(e)}")
    st.stop()

# Initialize LLM (HuggingFace Hub)
try:
    llm = HuggingFaceHub(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HUGGINGFACE_TOKEN,
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
except Exception as e:
    st.error("Failed to initialize LLM. Please check the HuggingFace token or model availability.")
    logger.error(f"LLM initialization failed: {str(e)}")
    st.stop()

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
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_name in ["index.faiss", "index.pkl"]:
                s3_key = f"{index_path}{file_name}"
                local_path = os.path.join(temp_dir, file_name)
                s3_client.download_file(S3_BUCKET, s3_key, local_path)
            vector_store = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load FAISS index from S3: {str(e)}")
        raise


# Function to count new files in S3 folder
def count_new_files(folder_prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
        if 'Contents' not in response:
            return 0
        new_files = sum(1 for obj in response['Contents'] if not obj['Key'].endswith('_processed.pdf'))
        return new_files
    except Exception as e:
        logger.error(f"Failed to count new files in S3: {str(e)}")
        return 0


# Retry decorator for HuggingFace API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(huggingface_hub.errors.HfHubHTTPError),
    before_sleep=lambda retry_state: logger.info(f"Retrying API call, attempt {retry_state.attempt_number}")
)
def run_qa_chain(qa_chain, user_input):
    return qa_chain.run(user_input)


# Load all FAISS indexes at startup
def load_all_faiss_indexes():
    vector_stores = {}
    for source, path in INDEX_PATHS.items():
        try:
            vector_stores[source] = load_faiss_index_from_s3(path)
        except Exception as e:
            st.error(f"Failed to load FAISS index for {source}. Skipping this source.")
            logger.error(f"FAISS index loading failed for {source}: {str(e)}")
    return vector_stores


# Main Streamlit app
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")

    # Custom CSS for black theme (unchanged)
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

    # Sidebar for stats
    with st.sidebar:
        st.title("Chatbot Stats")
        st.subheader("New Files Processed")
        proforma_new = count_new_files(PROFORMA_FOLDER)
        po_new = count_new_files(PO_FOLDER)
        st.write(f"Proforma Invoices: {proforma_new} new files")
        st.write(f"Purchase Orders: {po_new} new files")
        st.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Chat interface
    st.title("RAG Chatbot")
    st.write("Ask anything based on Proforma Invoices and Purchase Orders.")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_stores" not in st.session_state:
        with st.spinner("Loading all FAISS indexes from S3..."):
            st.session_state.vector_stores = load_all_faiss_indexes()
    if "qa_chain" not in st.session_state:
        try:
            # Combine all retrievers into a single QA chain
            retrievers = {source: vs.as_retriever(search_kwargs={"k": 4}) for source, vs in
                          st.session_state.vector_stores.items()}
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=None,  # We'll dynamically set documents later
                chain_type_kwargs={
                    "prompt": prompt_template,
                    "document_variable_name": "documents"
                }
            )
            st.session_state.retrievers = retrievers
        except Exception as e:
            st.error("Failed to initialize QA chain.")
            logger.error(f"QA chain initialization failed: {str(e)}")
            st.stop()

    # Chat input
    user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            try:
                # Search across all vector stores
                all_docs = []
                for source, retriever in st.session_state.retrievers.items():
                    docs = retriever.get_relevant_documents(user_input)
                    all_docs.extend(docs)

                # Combine document content
                combined_docs = "\n".join([doc.page_content for doc in all_docs])

                # Run QA chain with combined documents
                response = st.session_state.qa_chain({"documents": combined_docs, "question": user_input})
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response["result"]))
            except huggingface_hub.errors.HfHubHTTPError as e:
                st.error(
                    "Failed to get a response from the LLM. This could be due to API rate limits or model unavailability. Please try again later or contact support.")
                logger.error(f"HuggingFace API error: {str(e)}")
            except Exception as e:
                st.error("An unexpected error occurred while generating the response. Please try again.")
                logger.error(f"Unexpected error during QA chain run: {str(e)}")

    # Display chat history with latest at top, oldest at bottom
    for sender, message in reversed(st.session_state.chat_history):
        if sender == "user":
            st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()