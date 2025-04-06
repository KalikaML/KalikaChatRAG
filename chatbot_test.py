import streamlit as st
import boto3
import os
import tempfile
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import toml

# Load secrets from secrets.toml
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
secrets = toml.load(SECRETS_FILE_PATH)

# Configuration constants
S3_BUCKET = "kalika-rag"
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index/"
EMBEDDING_MODEL = "BAAI/BGE-small-en-v1.5"  # Embedding model used
AWS_ACCESS_KEY = secrets["access_key_id"]
AWS_SECRET_KEY = secrets["secret_access_key"]
GEMINI_MODEL = "gemini-1.5-pro"  # Specify Gemini model version
GEMINI_API_KEY = secrets["gemini_api_key"]

# Initialize S3 client
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
except Exception as e:
    logging.error(f"Error initializing S3 client: {str(e)}")
    s3_client = None

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize Gemini model for LLM responses
gemini_model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.5  # Adjustable parameter for response creativity
)

# Set up logging
logging.basicConfig(level=logging.INFO)


@st.cache_resource
def download_faiss_index_from_s3(bucket, prefix):
    """
    Download and load the FAISS index from S3 to a local directory.
    """
    if not s3_client:
        logging.error("S3 client not initialized. Check your AWS credentials.")
        return None

    try:
        # Create a temporary directory to store the downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # List all objects in the specified S3 prefix
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' not in response:
                logging.warning("No FAISS index files found in S3.")
                return None

            # Download all FAISS-related files from S3 to the local temp directory
            for obj in response['Contents']:
                key = obj.get('Key')
                if isinstance(key, str):  # Ensure key is a string
                    local_file_path = os.path.join(temp_dir, os.path.basename(key))
                    s3_client.download_file(bucket, key, local_file_path)
                    logging.info(f"Downloaded {key} to {local_file_path}")
                else:
                    logging.warning(f"Invalid key type: {key}")

            # Load the FAISS index from the local directory with dangerous deserialization enabled
            vector_store = FAISS.load_local(
                temp_dir,
                embeddings,
                allow_dangerous_deserialization=True  # Enable dangerous deserialization explicitly
            )
            return vector_store

    except Exception as e:
        logging.error(f"Error downloading or loading FAISS index: {str(e)}")
        return None


def query_faiss_index(vector_store, query_text):
    """
    Query the FAISS index and return results.
    """
    try:
        results = vector_store.similarity_search(query_text, k=5)
        return results
    except Exception as e:
        logging.error(f"Error querying FAISS index: {str(e)}")
        return []


def count_faiss_files(bucket, prefix):
    """
    Count the number of FAISS-related files in the S3 bucket.
    """
    if not s3_client:
        logging.error("S3 client not initialized. Check your AWS credentials.")
        return 0

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return 0

        # Ensure only valid keys are counted (e.g., strings)
        file_count = sum(1 for obj in response['Contents'] if isinstance(obj.get('Key'), str))
        return file_count

    except Exception as e:
        logging.error(f"Error counting FAISS files: {str(e)}")
        return 0


# Streamlit Chatbot UI
st.title("Personalized Chatbot")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load FAISS index from S3 with caching for faster access
st.write("Loading FAISS index from S3...")
vector_store = download_faiss_index_from_s3(S3_BUCKET, S3_PROFORMA_INDEX_PATH)

if vector_store:
    st.success("FAISS index loaded successfully!")
else:
    st.error("Failed to load FAISS index.")

# Display count of indexed files in S3
faiss_file_count = count_faiss_files(S3_BUCKET, S3_PROFORMA_INDEX_PATH)
st.write(f"Number of indexed files in S3: {faiss_file_count}")

# Chat input box
user_input = st.text_input("You:", placeholder="Enter your message")

if user_input:
    # Append user input to conversation history as HumanMessage objects
    st.session_state.conversation_history.append(HumanMessage(content=user_input))

    # Query FAISS index for relevant results (if available)
    context_messages = []
    if vector_store:
        results = query_faiss_index(vector_store, user_input)
        if results:
            context_messages.append(SystemMessage(content=f"Context: {results[0].page_content}"))

    # Generate response using Gemini model with structured messages (HumanMessage + SystemMessage)
    messages_to_send = context_messages + [HumanMessage(content=user_input)]

    # Generate AI response using invoke method and append it directly to conversation history
    ai_response_message: AIMessage = gemini_model.invoke(messages_to_send)  # Response is an AIMessage object

    st.session_state.conversation_history.append(ai_response_message)

    # Display conversation history
    st.write("Conversation:")

    for message in st.session_state.conversation_history:
        role_prefix = "AI:" if isinstance(message, AIMessage) else "You:"
        st.write(f"{role_prefix} {message}")
