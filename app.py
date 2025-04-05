import logging
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration constants
S3_BUCKET = "kalika-rag"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
LOCAL_FAISS_DIR = "C:/znew_chatboat_rag/faiss_index"  # Local folder to store FAISS index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-pro"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize S3 client using Streamlit secrets
def init_s3_client():
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["access_key_id"],
            aws_secret_access_key=st.secrets["secret_access_key"],
        )
        return s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {str(e)}")
        st.error("Failed to connect to S3. Please check your credentials.")
        return None

# Initialize Gemini API via LangChain
def init_gemini():
    try:
        model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=st.secrets["gemini_api_key"],
            temperature=0.5  # Adjustable parameter for response creativity
        )
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Gemini: {str(e)}")
        st.error("Failed to connect to Gemini API. Please check your API key.")
        return None

# Download FAISS index from S3 to local folder if not already present
def ensure_faiss_index_local(s3_client):
    try:
        # Check if FAISS index already exists locally
        if os.path.exists(LOCAL_FAISS_DIR) and os.listdir(LOCAL_FAISS_DIR):
            logging.info("FAISS index found locally, loading from disk...")
            vector_store = FAISS.load_local(LOCAL_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            st.write(f"Using existing FAISS index from: {LOCAL_FAISS_DIR}")
            return vector_store

        # If not present, download from S3
        logging.info("No local FAISS index found, downloading from S3...")
        if os.path.exists(LOCAL_FAISS_DIR):
            shutil.rmtree(LOCAL_FAISS_DIR)  # Clear any incomplete folder
        os.makedirs(LOCAL_FAISS_DIR)

        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PROFORMA_INDEX_PATH)

        if 'Contents' not in response:
            logging.warning("No FAISS index found in S3.")
            return None

        total_size = 0
        for obj in response['Contents']:
            key = obj['Key']
            local_path = os.path.join(LOCAL_FAISS_DIR, os.path.basename(key))
            s3_client.download_file(S3_BUCKET, key, local_path)
            file_size = os.path.getsize(local_path)  # Get size in bytes
            total_size += file_size
            logging.info(f"Downloaded FAISS file: {key} ({file_size / 1024:.2f} KB)")

        # Convert total size to human-readable format
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"

        st.write(f"Total size of FAISS index downloaded: {size_str}")
        logging.info(f"Total FAISS index size: {size_str}")

        # Load the FAISS index from local folder
        vector_store = FAISS.load_local(LOCAL_FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
        logging.info("Successfully loaded FAISS index from local folder")
        return vector_store

    except Exception as e:
        logging.error(f"Failed to ensure FAISS index locally: {str(e)}")
        return None

# Query the FAISS index with adjustable k (number of results)
def query_faiss_index(vector_store, query, k=10):  # Increased k for broader queries
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logging.error(f"Error querying FAISS index: {str(e)}")
        return None

# Generate response using Gemini via LangChain
def generate_response(model, query, faiss_results):
    try:
        if not faiss_results:
            return "No relevant information found in the proforma invoices."

        # Combine FAISS results into a context
        context = "\n\n".join([result.page_content for result in faiss_results])
        prompt = (
            f"You are an expert assistant for a sales team analyzing proforma invoices. "
            f"Based on the following information from proforma invoices:\n\n{context}\n\n"
            f"Answer the query in detail: {query}"
        )

        # Use LangChain's invoke method to generate response
        response = model.invoke(prompt)
        return response.content  # Extract the content from the response object
    except Exception as e:
        logging.error(f"Error generating response with Gemini: {str(e)}")
        return "An error occurred while generating the response."

# Main chatbot interface
def main():
    st.title("Proforma Invoice Chatbot for Sales Team")

    # Initialize S3 client and Gemini model
    s3_client = init_s3_client()
    gemini_model = init_gemini()
    vector_store = None

    if s3_client:
        with st.spinner("Loading FAISS index..."):
            vector_store = ensure_faiss_index_local(s3_client)

        if vector_store:
            st.success("FAISS index successfully loaded!")
            st.write(f"FAISS index location: {LOCAL_FAISS_DIR}")
        else:
            st.error("Failed to load FAISS index from local or S3.")
            return

    if not gemini_model:
        return

    # Query input and response
    if vector_store:
        st.subheader("Ask a Question")
        query = st.text_input("Enter your query about the proforma invoices (e.g., 'How many proformas do we have?', 'List all products', 'Details for a specific company'):")

        if st.button("Submit"):
            if query:
                with st.spinner("Searching and generating response..."):
                    # Search FAISS index with a higher k to capture more context
                    faiss_results = query_faiss_index(vector_store, query, k=10)

                    # Generate response with Gemini
                    response = generate_response(gemini_model, query, faiss_results)

                st.subheader("Response")
                st.write(response)
            else:
                st.warning("Please enter a query.")

=======
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import toml
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

# Prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.
    Context: {context}
    Question: {question}
    Answer:
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
        /* General page styling */
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        /* Chat message styling */
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
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #252525;
            padding: 20px;
            color: #E0E0E0;
        }
        /* Input field */
        .stTextInput > div > div > input {
            background-color: #333333;
            color: #E0E0E0;
            border: 1px solid #555555;
            border-radius: 5px;
        }
        /* Button styling */
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
        /* Titles and headers */
        h1, h2, h3 {
            color: #BB86FC;
        }
        /* Spinner text */
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
    st.write("Ask anything based on the selected data source!")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_option" not in st.session_state:
        st.session_state.current_option = None

    # Load FAISS index based on selected option
    if option != st.session_state.current_option:
        st.session_state.current_option = option
        index_path = PROFORMA_INDEX_PATH if option == "Proforma Invoices" else PO_INDEX_PATH
        with st.spinner(f"Loading {option} FAISS index from S3..."):
            vector_store = load_faiss_index_from_s3(index_path)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )

    # Chat input
    user_input = st.text_input("Your Question:", key="input", placeholder="Type your question here...")
    if st.button("Send") and user_input:
        with st.spinner("Generating response..."):
            response = st.session_state.qa_chain.run(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()