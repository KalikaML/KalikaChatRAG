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
import streamlit_authenticator as stauth # Import the authenticator library

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
MODEL_DIRECTORY = "BAAI-bge-base-en-v1.5" # Local embedding model directory
S3_BUCKET = "kalika-rag"  # Your S3 bucket name
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index" # Path within bucket (no extension)
GEMINI_MODEL = "gemini-1.5-pro" # Or your preferred Gemini model

try:
    # Load all secrets from the TOML file
    secrets = toml.load(SECRETS_FILE_PATH)

    # AWS Credentials
    AWS_ACCESS_KEY = secrets["access_key_id"]
    AWS_SECRET_KEY = secrets["secret_access_key"]

    # Gemini API Key
    GEMINI_API_KEY = secrets["gemini_api_key"]

    # --- Load Streamlit Authenticator Config ---
    # It expects a structure like:
    # [credentials.usernames.USERNAME]
    # name = "Display Name"
    # email = "email@example.com"
    # password = "HASHED_PASSWORD"
    # [cookie]
    # name = "cookie_name"
    # key = "secret_signature_key"
    # expiry_days = 30
    credentials = secrets.get('credentials')
    cookie_config = secrets.get('cookie')

    # --- Validate Essential Secrets ---
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
         st.error("AWS credentials (access_key_id, secret_access_key) missing in secrets.toml.")
         st.stop()
    if not GEMINI_API_KEY:
         st.error("Gemini API key (gemini_api_key) missing in secrets.toml.")
         st.stop()
    if not credentials or not cookie_config or not cookie_config.get('key'):
        st.error("Authentication 'credentials' or 'cookie' configuration missing/incomplete in secrets.toml. "
                 "Please ensure the [credentials.usernames...] and [cookie] sections are correctly defined using valid TOML.")
        st.stop()
    # Optional: Warn if default cookie key is used
    if cookie_config.get('key') == "choose_a_strong_random_secret_key" or cookie_config.get('key') == "a_very_secret_random_string_12345":
         st.warning("⚠️ Security Warning: Please change the default cookie 'key' in secrets.toml to a unique, strong, random string!")

except FileNotFoundError:
    st.error(f"❌ Critical Error: Secrets file not found at {SECRETS_FILE_PATH}. App cannot run.")
    st.stop()
except KeyError as e:
    st.error(f"❌ Critical Error: Missing required key in {SECRETS_FILE_PATH}: {e}. App cannot run.")
    st.stop()
except toml.TomlDecodeError as e:
     st.error(f"❌ Critical Error: Invalid TOML format in {SECRETS_FILE_PATH}. Please check the syntax. Error: {e}")
     st.stop()
except Exception as e:
     st.error(f"❌ Critical Error loading secrets or configuration: {e}")
     st.stop()

# --- Set up logging ---
# Logs will show up in the terminal where you run Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application starting up...")

# --- Initialize Authenticator ---
# This uses the credentials and cookie settings loaded from secrets.toml
authenticator = stauth.Authenticate(
    credentials,
    cookie_config['name'],
    cookie_config['key'],
    cookie_config['expiry_days'],
    # preauthorized= # Optional: Add list of emails for pre-authorized users if needed
)

# --- Render Login Form & Handle Authentication ---
# This must happen before the main app logic that requires authentication
# It renders the login form and returns the authentication status
name, authentication_status, username = authenticator.login(location='main') # Render login in the main page area

# --- Authentication Status Check ---
if authentication_status == False:
    st.error('Username/password is incorrect')
    logging.warning(f"Failed login attempt for username: {username}")
    st.stop() # Stop execution if login fails
elif authentication_status == None:
    st.warning('Please enter your username and password to access the application.')
    st.stop() # Stop execution if user hasn't tried logging in yet
elif authentication_status:
    # --- User is Authenticated --- #
    # Now we can proceed with the main application logic and resource loading

    # Display welcome message and logout button (typically in the sidebar)
    st.sidebar.success(f"Welcome *{name}* 👋")
    authenticator.logout('Logout', 'sidebar')
    logging.info(f"User '{username}' authenticated successfully.")

    # --- Resource Initialization Functions (Cached) ---

    @st.cache_resource  # Cache S3 client resource across reruns for this session
    def get_s3_client():
        """Initializes and returns the Boto3 S3 client."""
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                # Consider adding region_name='your-region' if needed
            )
            # Quick check to verify credentials and connectivity
            s3.list_buckets()
            logging.info("S3 client initialized successfully.")
            return s3
        except Exception as e:
            logging.error(f"Error initializing S3 client: {str(e)}", exc_info=True)
            st.error(f"Failed to connect to S3. Check AWS credentials and permissions in secrets.toml. Error: {e}")
            return None

    @st.cache_resource # Cache embeddings model for this session
    def get_embeddings_model():
        """
        Loads the HuggingFace embeddings model from the local directory specified
        by MODEL_DIRECTORY. Uses TRANSFORMERS_CACHE strategy.
        """
        model_path = MODEL_DIRECTORY # Relative path to the local model directory

        # Check if the specified local directory exists
        abs_model_path = os.path.abspath(model_path)
        if not os.path.isdir(abs_model_path):
            st.error(f"Local embedding model directory not found at '{abs_model_path}'. "
                     f"Please ensure the directory '{MODEL_DIRECTORY}' exists relative to the script "
                     f"and contains the model files.")
            logging.error(f"Model directory '{model_path}' (abs: {abs_model_path}) not found.")
            return None

        try:
            # Set TRANSFORMERS_CACHE to the *parent* directory (current dir in this case)
            # This helps the library locate the model folder correctly with local_files_only=True
            cache_dir = os.path.abspath('.')
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logging.info(f"Setting TRANSFORMERS_CACHE to: {cache_dir} to aid local model discovery.")

            # Initialize HuggingFaceEmbeddings, forcing local file usage
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path, # Use the directory name/path as the identifier
                model_kwargs={
                    'device': 'cpu', # Use CPU (adjust if GPU is available and configured)
                    "local_files_only": True # CRUCIAL: prevent any download attempts
                },
                encode_kwargs={'normalize_embeddings': True} # Recommended for BGE models
            )

            # Perform a dummy embedding to ensure the model loads correctly now
            _ = embeddings.embed_query("Initialization test query")
            logging.info(f"Embeddings model '{model_path}' loaded successfully from local directory.")
            return embeddings

        except Exception as e:
            st.error(f"Failed to load local embeddings model from '{model_path}'. Error: {e}")
            logging.error(f"Failed to load embeddings model from '{model_path}': {e}", exc_info=True)
            if "offline" in str(e) or "ConnectionError" in str(e):
                 st.error("Error suggests an attempt to connect online. Double-check 'local_files_only=True' is effective and all necessary model files are present locally.")
            elif "snapshot" in str(e):
                 st.error(f"Could not find the expected model file structure within '{abs_model_path}'. Verify the directory contents.")
            return None

    @st.cache_resource # Cache LLM model for this session
    def get_gemini_model():
        """Initializes and returns the Gemini LLM client."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,  # Lower temperature for more factual RAG answers
                convert_system_message_to_human=True # Good practice for some models
            )
            # Simple check (optional, as invoke will fail later if key is invalid)
            # llm.invoke("Hi")
            logging.info(f"Gemini model '{GEMINI_MODEL}' initialized successfully.")
            return llm
        except Exception as e:
            st.error(f"Failed to initialize Gemini model '{GEMINI_MODEL}'. Check API Key in secrets.toml. Error: {e}")
            logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            return None

    @st.cache_resource(ttl=3600)  # Cache the loaded index for 1 hour
    def download_and_load_faiss_index(_s3_client, _embeddings, bucket, s3_prefix):
        """
        Downloads FAISS index files (.faiss, .pkl) from S3, loads into memory.
        Requires allow_dangerous_deserialization=True for FAISS.load_local.
        """
        # --- Essential Checks ---
        if not _s3_client:
            st.error("S3 client is not available. Cannot download FAISS index.")
            logging.error("FAISS load skipped: S3 client not initialized.")
            return None
        if not _embeddings:
            st.error("Embeddings model is not available. Cannot load FAISS index.")
            logging.error("FAISS load skipped: Embeddings model not initialized.")
            return None
        # --- End Checks ---

        # Define the specific file keys FAISS uses when saving
        s3_index_key = f"{s3_prefix}.faiss"
        s3_pkl_key = f"{s3_prefix}.pkl"
        local_faiss_filename = "index.faiss" # FAISS expects these exact names on load_local
        local_pkl_filename = "index.pkl"

        try:
            # Create a temporary directory that persists for the cached resource lifespan
            with tempfile.TemporaryDirectory() as temp_dir:
                local_index_path = os.path.join(temp_dir, local_faiss_filename)
                local_pkl_path = os.path.join(temp_dir, local_pkl_filename)

                logging.info(f"Attempting to download index files: "
                             f"s3://{bucket}/{s3_index_key} -> {local_index_path} and "
                             f"s3://{bucket}/{s3_pkl_key} -> {local_pkl_path}")

                # Download the files from S3
                _s3_client.download_file(bucket, s3_index_key, local_index_path)
                _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
                logging.info(f"Successfully downloaded index files to temporary directory: {temp_dir}")

                # Load the FAISS index from the temporary directory containing the files
                vector_store = FAISS.load_local(
                    folder_path=temp_dir, # Pass the directory path
                    embeddings=_embeddings,
                    index_name="index", # Base name used during save_local
                    allow_dangerous_deserialization=True  # *** Required ***
                )
                logging.info(f"FAISS index loaded successfully into memory from {s3_prefix}.")
                # The vector store object is returned; temp_dir is cleaned up automatically after 'with' block exits
                return vector_store

        except _s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                st.error(f"FAISS index files ('{s3_index_key}', '{s3_pkl_key}') not found in s3://{bucket}/. "
                         f"Please ensure the S3 bucket ('{S3_BUCKET}') and index path ('{S3_PROFORMA_INDEX_PATH}') "
                         f"are correct and the index has been generated and uploaded.")
                logging.error(f"FAISS index files not found at s3://{bucket}/{s3_prefix}.")
            elif error_code == 'NoSuchBucket':
                 st.error(f"S3 bucket '{bucket}' not found. Please check the S3_BUCKET name in the script and ensure the bucket exists.")
                 logging.error(f"S3 bucket '{bucket}' not found.")
            elif error_code in ['NoCredentialsError', 'InvalidAccessKeyId', 'SignatureDoesNotMatch', 'ExpiredToken']:
                 st.error(f"AWS S3 Authentication/Authorization Error: {e}. Please check your AWS credentials in secrets.toml and ensure they have S3 read permissions for the bucket.")
                 logging.error(f"S3 Authentication/Authorization Error: {e}")
            else:
                st.error(f"Error downloading FAISS index from S3: {e}")
                logging.error(f"S3 ClientError downloading index: {e}", exc_info=True)
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while loading the FAISS index: {e}")
            logging.error(f"Error loading FAISS index: {e}", exc_info=True)
            if "allow_dangerous_deserialization" in str(e):
                 st.error("Error suggests deserialization issue. Ensure 'allow_dangerous_deserialization=True' is set during FAISS.load_local.")
            elif "Can't get attribute" in str(e) or "ModuleNotFoundError" in str(e):
                 st.error("Deserialization Error: The environment loading the index might be missing classes/modules used when the index was created/saved. Ensure library versions (e.g., LangChain, FAISS) are consistent.")
                 logging.error("Potential library mismatch during FAISS deserialization.")
            return None

    # --- Helper Functions ---

    def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
        """Queries the FAISS index, returns retrieved documents."""
        if not vector_store:
            logging.warning("query_faiss_index called but vector_store is None.")
            st.warning("Knowledge base index is not loaded. Cannot perform search.")
            return []
        try:
            search_type = 'MMR' if use_mmr else 'Similarity'
            logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")

            if use_mmr:
                # Fetch more initial results for MMR to select diverse ones from
                results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k*4)
            else:
                results = vector_store.similarity_search(query_text, k=k)

            logging.info(f"Retrieved {len(results)} chunks using {search_type} search.")
            return results
        except Exception as e:
            st.error(f"Error querying FAISS index: {str(e)}")
            logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
            return []

    def generate_llm_response(llm, query_text, retrieved_docs):
        """Generates a response from the LLM using the query and retrieved context."""
        if not llm:
            logging.error("generate_llm_response called but llm is None.")
            return "LLM is not available. Cannot generate response."

        if not retrieved_docs:
            logging.info(f"Generating response for query '{query_text}' without context documents.")
            # Handle case where no relevant documents were found
            messages = [
                SystemMessage(content="You are an AI assistant answering questions about Proforma Invoices. "
                                      "No relevant context documents were found in the knowledge base for the user's query."),
                HumanMessage(content=query_text + "\n\nBased on the available knowledge base, I could not find specific information relevant to your query.")
            ]
        else:
             # Prepare context string
            context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
                                         for doc in retrieved_docs if hasattr(doc, 'metadata')])
            # Create the prompt for the LLM
            system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based *only* on the provided context documents.
            Analyze the user's query and the following context documents carefully.
            Synthesize the information from the relevant parts of the context to provide a comprehensive and accurate answer.
            Quote relevant snippets or data points from the context to support your answer where appropriate (e.g., mentioning the source document).
            If the answer cannot be fully found within the provided context, clearly state what information is available and specify what parts of the query cannot be answered based *solely* on the provided context.
            Do not make assumptions or use any external knowledge outside of the provided context. Stick strictly to the information given below.

            Provided Context Documents:
            ---
            {context}
            ---
            """
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query_text)
            ]
            logging.info(f"Generating response for query: '{query_text}' with {len(retrieved_docs)} context chunks.")

        try:
            # Invoke the LLM
            ai_response: AIMessage = llm.invoke(messages)
            logging.info(f"Successfully generated LLM response for query: '{query_text}'")
            return ai_response.content
        except Exception as e:
            st.error(f"Error generating response from LLM: {e}")
            logging.error(f"LLM invocation error for query '{query_text}': {e}", exc_info=True)
            # Provide specific feedback if possible, e.g., quota issues
            if "quota" in str(e).lower():
                 return "Sorry, I could not generate a response due to API quota limits."
            return "Sorry, I encountered an error while generating the response."


    # --- Main Application UI (Authenticated Access) ---

    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown(f"Welcome, {name}! Ask questions about the proforma invoices processed from email attachments.")

    # --- Load Resources & Display Status ---
    vector_store = None # Initialize vector_store to None
    with st.status("Initializing resources...", expanded=True) as status_container:
        st.write("Initializing S3 Connection...")
        s3_client = get_s3_client()
        s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
        st.write(s3_status)

        st.write("Loading Local Embedding Model...")
        embeddings = get_embeddings_model()
        embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
        st.write(embeddings_status)

        st.write("Initializing LLM...")
        gemini_model = get_gemini_model()
        gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"
        st.write(gemini_status)

        # Proceed only if core components loaded
        if s3_client and embeddings and gemini_model:
            st.write("Loading Knowledge Base Index from S3...")
            vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
            index_status = "✅ Knowledge Base Index Loaded" if vector_store else "❌ Knowledge Base Index Failed"
            st.write(index_status)

            if vector_store:
                status_container.update(label="Initialization Complete!", state="complete", expanded=False)
            else:
                 status_container.update(label="Initialization Failed (Index Load)", state="error", expanded=True)
                 st.error("Failed to load the knowledge base index. Querying is disabled. Check logs and S3 configuration.")
                 # No st.stop() here, allows user to see the error message clearly
        else:
            status_container.update(label="Initialization Failed!", state="error", expanded=True)
            st.error("Core components (S3, Embeddings, or LLM) failed to initialize. Application cannot proceed. Check logs.")
            # No st.stop() here, allows user to see the error message clearly

    # --- Query Interface ---
    st.markdown("---")

    # Disable input if the vector store failed to load
    query_disabled = not vector_store
    query_placeholder = "e.g., What is the total amount for invoice [filename]?"
    if query_disabled:
        query_placeholder = "Knowledge Base not loaded. Querying disabled."

    query_text = st.text_input(
        "Enter your query:",
        placeholder=query_placeholder,
        key="query_input",
        disabled=query_disabled # Disable input if index isn't loaded
    )

    # --- Advanced Settings (Consider Sidebar) ---
    # Using fixed settings for simplicity now
    k_results = 15  # Number of context chunks to retrieve
    use_mmr_search = False # Use similarity search by default

    # --- Execute Query and Display Results ---
    if query_text and vector_store: # Only proceed if query entered and index loaded
        # 1. Query FAISS index
        with st.spinner(f"Searching knowledge base..."):
            retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

        # 2. Generate LLM response
        with st.spinner("🧠 Thinking..."):
            response = generate_llm_response(gemini_model, query_text, retrieved_docs)

        # 3. Display response
        st.markdown("### Response:")
        st.markdown(response) # Use markdown for better formatting
        st.markdown("---")

        # 4. Optional: Display retrieved context
        if retrieved_docs:
            with st.expander("🔍 Show Retrieved Context Snippets"):
                st.markdown(f"Retrieved {len(retrieved_docs)} snippets:")
                for i, doc in enumerate(retrieved_docs):
                    source_info = "Unknown Source"
                    page_info = ""
                    # Attempt to extract metadata robustly
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                         source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                         page_num = doc.metadata.get('page')
                         if page_num is not None:
                              page_info = f", Page: {page_num + 1}" # Often 0-indexed

                    st.text_area(
                        label=f"**Snippet {i + 1}** ({source_info}{page_info})",
                        value=doc.page_content,
                        height=150,
                        key=f"snippet_{i}",
                        disabled=True # Make text area read-only
                    )
        else:
            # Provide feedback even if LLM handled the "no context" case
            st.info("No specific context snippets were found in the knowledge base for this query to display.")

    elif query_text and not vector_store:
         # This case should ideally be prevented by disabling the input, but added as a safeguard
         st.error("Cannot process query because the knowledge base index is not loaded.")

# --- End of Authenticated Section ---

logging.info("Application finished processing request or waiting for input.")