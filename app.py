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
import hashlib

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Secrets ---
#SECRETS_FILE_PATH = ".streamlit/secrets.toml"

try:
    #secrets = toml.load(SECRETS_FILE_PATH)
    # Core application settings
    S3_BUCKET = "kalika-rag"
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
    MODEL_DIRECTORY = "BAAI/BAAI-bge-base-en-v1.5"
    AWS_ACCESS_KEY = st.secrets["access_key_id"]
    AWS_SECRET_KEY = st.secrets["secret_access_key"]
    GEMINI_MODEL = "gemini-1.5-pro"
    GEMINI_API_KEY = st.secrets["gemini_api_key"]

    # Authentication credentials
    if "credentials" in st.secrets:
        CREDENTIALS = st.secrets["credentials"]["usernames"]
    else:
        CREDENTIALS = {
            "user1": {
                "name": "User",
                "email": "user@example.com",
                "password": hashlib.sha256("user@123".encode()).hexdigest()
            }
        }

except FileNotFoundError:
    st.error(f"Secrets file not found at . App cannot run.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret key in : {e}. App cannot run.")
    st.stop()

# --- Authentication Functions ---
def verify_password(username, password):
    """Verify the password for a given username"""
    if username not in CREDENTIALS:
        logging.warning(f"Login attempt for non-existent user: {username}")
        return False

    stored_hashed_password = CREDENTIALS[username]["password"]
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()
    st.write(f"Input Hash (from typed password): {input_password_hash}")
    is_match = input_password_hash == stored_hashed_password
    if not is_match:
        logging.warning(f"Password mismatch for user: {username}")
    return is_match

def get_user_info(username):
    """Get user info for a given username"""
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None

# --- Initialize S3 client ---
@st.cache_resource
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        s3.list_buckets()
        logging.info("S3 client initialized successfully.")
        return s3
    except Exception as e:
        logging.error(f"Error initializing S3 client: {str(e)}")
        st.error(f"Failed to connect to S3. Check AWS credentials and permissions. Error: {e}")
        return None

# --- Initialize Embeddings Model ---
@st.cache_resource
def get_embeddings_model():
    model_path = MODEL_DIRECTORY
    if not os.path.isdir(model_path):
        st.error(f"Local model directory not found at '{os.path.abspath(model_path)}'. "
                 f"Please ensure the directory '{MODEL_DIRECTORY}' exists in the same "
                 f"location as the Streamlit script or provide the correct path.")
        logging.error(f"Model directory {model_path} not found.")
        return None

    try:
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        logging.info(f"Set TRANSFORMERS_CACHE to: {cache_dir}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': 'cpu',
                "local_files_only": True
            },
            encode_kwargs={'normalize_embeddings': True}
        )

        _ = embeddings.embed_query("Test query")
        logging.info(f"Embeddings model '{model_path}' loaded successfully from local directory.")
        return embeddings

    except Exception as e:
        st.error(f"Failed to load embeddings model from '{model_path}'. Error: {e}")
        logging.error(f"Failed to load embeddings model from '{model_path}': {e}", exc_info=True)
        if "ConnectionError" in str(e) or "offline" in str(e):
            st.error("It seems the application tried to connect to Hugging Face Hub. "
                     "Ensure 'local_files_only=True' is correctly set and the "
                     f"'{MODEL_DIRECTORY}' contains all necessary model files (config.json, pytorch_model.bin/model.safetensors, tokenizer files etc.).")
        elif "snapshot" in str(e):
            st.error(
                f"The library could not find the model files within the expected structure in '{os.path.abspath(model_path)}' "
                f"even with TRANSFORMERS_CACHE set. Please verify the contents of the '{MODEL_DIRECTORY}' directory.")
        return None

# --- Initialize Gemini LLM ---
@st.cache_resource
def get_gemini_model():
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        logging.info(f"Gemini model {GEMINI_MODEL} initialized.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini model {GEMINI_MODEL}. Check API Key. Error: {e}")
        logging.error(f"Failed to initialize Gemini model: {e}")
        return None

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _embeddings:
        st.error("Embeddings model failed to load. Cannot load FAISS index.")
        logging.error("Attempted to load FAISS index, but embeddings are not available.")
        return None
    if not _s3_client:
        st.error("S3 client not initialized. Cannot load index.")
        return None

    s3_index_key = f"{prefix}.faiss"
    s3_pkl_key = f"{prefix}.pkl"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")
            local_pkl_path = os.path.join(temp_dir, "index.pkl")
            logging.info(f"Attempting to download index from s3://{bucket}/{prefix} (.faiss and .pkl)")
            _s3_client.download_file(bucket, s3_index_key, local_index_path)
            _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
            logging.info(f"Successfully downloaded index files to {temp_dir}")
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("FAISS index loaded successfully into memory.")
            return vector_store

    except _s3_client.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            st.error(
                f"FAISS index files ({s3_index_key}, {s3_pkl_key}) not found in s3://{bucket}/. Please run the indexing script.")
            logging.error(f"FAISS index files not found at s3://{bucket}/{prefix}.")
        elif error_code == 'NoSuchBucket':
            st.error(f"S3 bucket '{bucket}' not found. Please check the S3_BUCKET name.")
            logging.error(f"S3 bucket '{bucket}' not found.")
        elif error_code in ['NoCredentialsError', 'InvalidAccessKeyId', 'SignatureDoesNotMatch']:
            st.error(f"AWS S3 Authentication Error: {e}. Please check your AWS credentials in secrets.toml.")
            logging.error(f"S3 Authentication Error: {e}")
        else:
            st.error(f"Error downloading FAISS index from S3: {e}")
            logging.error(f"S3 ClientError downloading index: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        if "Pickle" in str(e) or "deserialization" in str(e):
            st.error("Potential deserialization issue with the FAISS index (.pkl file). "
                     "Ensure the index was created with the same version of LangChain/FAISS "
                     "and that 'allow_dangerous_deserialization=True' is set.")
        return None

# --- Querying Functions ---
def query_faiss_index(vector_store, query_text,k=10, use_mmr=False):
    if not vector_store:
        logging.warning("query_faiss_index called but vector_store is None.")
        return []
    try:
        search_kwargs = {'k': k}
        search_type = 'similarity'
        if use_mmr:
            search_type = 'mmr'
        logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")
        if use_mmr:
            results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k * 4)
        else:
            results = vector_store.similarity_search(query_text, k=k)
        logging.info(f"Retrieved {len(results)} chunks using {search_type} search.")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        logging.error("generate_follow_up_questions called but llm is None.")
        return []
    context = ""
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
    try:
        follow_up_prompt = f"""Based on the following user query and the response provided, 
        generate 3 specific, relevant follow-up questions the user might want to ask next.
        Make questions clear, concise, and directly relevant to the invoice context. 

        Previous Query: {query_text}

        Response: {response_text}

        {f'Context (excerpt): {context[:500]}...' if context else ''}

        Provide ONLY three follow-up questions in a simple list format, one per line.
        Each question should be directly related to invoices, specific details mentioned, 
        or natural next steps in understanding the invoice information.
        """
        messages = [
            SystemMessage(content="You are an AI assistant that generates relevant follow-up questions about invoices."),
            HumanMessage(content=follow_up_prompt)
        ]
        ai_response: AIMessage = llm.invoke(messages)
        questions = []
        for line in ai_response.content.strip().split('\n'):
            clean_line = line.strip()
            clean_line = clean_line.lstrip("0123456789.-*• ").strip()
            if clean_line and '?' in clean_line:
                questions.append(clean_line)
        return questions[:3]
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}", exc_info=True)
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    if not llm:
        logging.error("generate_llm_response called but llm is None.")
        return "LLM model is not available."
    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        context_sources = ", ".join(list(set(doc.metadata.get('source', 'N/A') for doc in retrieved_docs if hasattr(doc, 'metadata') and 'source' in doc.metadata)))
        context_log_msg = f"Context from sources: {context_sources}" if context_sources else "Context from retrieved chunks."
        system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based *only* on the provided context documents.
        Analyze the user's query and the following context documents carefully.
        Synthesize the information from the relevant parts of the context to provide a comprehensive and accurate answer.
        Quote relevant snippets or data points from the context to support your answer where appropriate.
        If the answer cannot be fully found within the provided context, clearly state what information is available and specify what parts of the query cannot be answered based on the context.
        Do not make assumptions or use any external knowledge outside of the provided context.

        Context Documents:
        ---
        {context}
        ---
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        logging.info(f"Generating response for query: '{query_text}' with {len(retrieved_docs)} context chunks. {context_log_msg}")
    else:
        messages = [
            SystemMessage(content="You are an AI assistant answering questions about Proforma Invoices. "
                                  "No relevant context documents were found in the knowledge base for the user's query."),
            HumanMessage(content=query_text + "\n\nSince no relevant documents were found, please state that you cannot answer the question based on the available knowledge base.")
        ]
        logging.info(f"Generating response for query: '{query_text}' without context documents.")
    try:
        ai_response: AIMessage = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        st.error(f"Error generating response from LLM: {e}")
        logging.error(f"LLM invocation error: {e}", exc_info=True)
        return "Sorry, I encountered an error while generating the response."

# --- Login Page ---
def login_page():
    st.title("📄 Proforma Invoice Assistant - Login")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Login to Access the System")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login", use_container_width=True)
        if login_button:
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = get_user_info(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# --- Main Application ---
def main_app():
    # Sidebar with New Chat and Logout
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        # Initialize chat sessions
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
            st.session_state.current_chat_id = "chat_1"
            st.session_state.chat_counter = 1
            st.session_state.chat_sessions["chat_1"] = {
                'query_history': [],
                'response_history': [],
                'follow_up_questions': []
            }
        # New Chat Button
        if st.button("New Chat"):
            st.session_state.chat_counter += 1
            new_chat_id = f"chat_{st.session_state.chat_counter}"
            st.session_state.chat_sessions[new_chat_id] = {
                'query_history': [],
                'response_history': [],
                'follow_up_questions': []
            }
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        # Logout Button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()

    # Main app UI
    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown("Ask questions about the proforma invoices processed from email attachments.")

    # Initialize resources
    s3_client = get_s3_client()
    embeddings = get_embeddings_model()
    gemini_model = get_gemini_model()

    # --- Resource Loading and Status ---
    s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
    embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
    gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"

    with st.status("Initializing resources...", expanded=False) as status_container:
        st.write(s3_status)
        st.write(embeddings_status)
        st.write(gemini_status)
        if not s3_client or not embeddings or not gemini_model:
            st.error("Core components failed to initialize. Application cannot proceed. Check logs for details.")
            status_container.update(label="Initialization Failed!", state="error")
            st.stop()
        else:
            st.write("Loading Knowledge Base Index...")
            vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
            if vector_store:
                st.write("✅ Knowledge Base Index Loaded")
                status_container.update(label="Initialization Complete!", state="complete", expanded=False)
            else:
                st.write("❌ Knowledge Base Index Failed to Load")
                status_container.update(label="Initialization Failed!", state="error")
                st.error("Failed to load the knowledge base index. Querying is disabled. Check S3 path and permissions.")
                st.stop()

    # --- Query Interface ---
    st.markdown("---")

    # Use current chat session
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
    if 'query_history' not in current_chat:
        current_chat['query_history'] = []
    if 'response_history' not in current_chat:
        current_chat['response_history'] = []
    if 'follow_up_questions' not in current_chat:
        current_chat['follow_up_questions'] = []

    # Display current chat history
    if current_chat['query_history']:
        for i in range(len(current_chat['query_history'])):
            st.markdown(f"**Question:**")
            st.markdown(f"> {current_chat['query_history'][i]}")
            st.markdown(f"**Answer:**")
            st.markdown(current_chat['response_history'][i])
            st.markdown("---")

    # Display input box for query
    if 'follow_up_clicked' in st.session_state and st.session_state.follow_up_clicked:
        query_text = st.session_state.follow_up_clicked
        st.session_state.follow_up_clicked = None
    else:
        query_text = st.text_input("Enter your query:",
                                   placeholder="e.g., What is the total amount for invoice [filename]? or List all products in [filename].",
                                   key="query_input",
                                   value=st.session_state.get('follow_up_clicked', ''),
                                   disabled=not vector_store)

    # Using fixed settings for simplicity
    k_results = 15
    use_mmr_search = False

    if query_text and vector_store:
        # 1. Query FAISS index
        with st.spinner(f"Searching knowledge base for relevant info (k={k_results}, MMR={use_mmr_search})..."):
            retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

        # 2. Generate LLM response
        with st.spinner("🧠 Synthesizing answer using retrieved context..."):
            response = generate_llm_response(gemini_model, query_text, retrieved_docs)

        # Store in current chat
        current_chat['query_history'].append(query_text)
        current_chat['response_history'].append(response)

        # 3. Generate follow-up questions
        with st.spinner("Generating follow-up questions..."):
            follow_up_questions = generate_follow_up_questions(gemini_model, query_text, response, retrieved_docs)
            current_chat['follow_up_questions'] = follow_up_questions

        # 4. Display response
        st.markdown("### Response:")
        st.markdown(response)

        # 5. Display follow-up questions as clickable buttons
        if follow_up_questions:
            st.markdown("### You might want to ask:")
            cols = st.columns(len(follow_up_questions))
            for i, question in enumerate(follow_up_questions):
                if cols[i].button(question, key=f"follow_up_{i}_{st.session_state.current_chat_id}"):
                    st.session_state.follow_up_clicked = question
                    st.rerun()

        st.markdown("---")

    elif query_text and not vector_store:	
        st.error("Cannot process query because the knowledge base index is not loaded.")

# --- Main Entry Point ---
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
