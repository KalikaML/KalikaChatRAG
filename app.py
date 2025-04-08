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

# --- Configuration and Constants ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
MODEL_DIRECTORY = "BAAI-bge-base-en-v1.5" # Local embedding model directory
S3_BUCKET = "kalika-rag"  # Your S3 bucket name
S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index" # Path within bucket (no extension)
GEMINI_MODEL = "gemini-1.5-pro" # Or your preferred Gemini model

# --- Set up logging ---
# Logs will show up in the terminal where you run Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application starting up...")

# --- Load Secrets and Basic Configuration ---
try:
    # Load all secrets from the TOML file
    secrets = toml.load(SECRETS_FILE_PATH)

    # AWS Credentials
    AWS_ACCESS_KEY = secrets.get("access_key_id") # Use .get for safer access
    AWS_SECRET_KEY = secrets.get("secret_access_key")

    # Gemini API Key
    GEMINI_API_KEY = secrets.get("gemini_api_key")

    # Authentication Config (will be validated more thoroughly later)
    credentials_config = secrets.get('credentials')
    cookie_config = secrets.get('cookie')

    # --- Validate Essential Secrets ---
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
         st.error("AWS credentials (access_key_id, secret_access_key) missing in secrets.toml.")
         logging.error("Missing AWS credentials in secrets.")
         st.stop()
    if not GEMINI_API_KEY:
         st.error("Gemini API key (gemini_api_key) missing in secrets.toml.")
         logging.error("Missing Gemini API key in secrets.")
         st.stop()
    # Basic check for auth sections - more detailed checks happen before init
    if not credentials_config:
         st.error("Authentication 'credentials' section missing in secrets.toml.")
         logging.error("'credentials' section not found in secrets.")
         st.stop()
    if not cookie_config:
        st.error("Authentication 'cookie' section missing in secrets.toml.")
        logging.error("'cookie' section not found in secrets.")
        st.stop()

except FileNotFoundError:
    st.error(f"❌ Critical Error: Secrets file not found at {SECRETS_FILE_PATH}. App cannot run.")
    logging.error(f"Secrets file not found at {SECRETS_FILE_PATH}.")
    st.stop()
except KeyError as e:
    # This might happen if using secrets['key'] instead of secrets.get('key') and key is missing
    st.error(f"❌ Critical Error: Missing required key in {SECRETS_FILE_PATH}: {e}. App cannot run.")
    logging.error(f"Missing key {e} in secrets file.")
    st.stop()
except toml.TomlDecodeError as e:
     st.error(f"❌ Critical Error: Invalid TOML format in {SECRETS_FILE_PATH}. Please check the syntax. Error: {e}")
     logging.error(f"Invalid TOML format in secrets file: {e}")
     st.stop()
except Exception as e:
     # Catch-all for other unexpected loading errors
     st.error(f"❌ Critical Error loading secrets or initial configuration: {e}")
     logging.error(f"Unexpected error loading secrets: {e}", exc_info=True)
     st.stop()


# --- Initialize Authenticator ---
# Includes detailed validation of config values before initializing the object
try:
    # --- Validate Cookie Config values and types ---
    cookie_name = cookie_config.get('name')
    cookie_key = cookie_config.get('key')
    cookie_expiry_days_raw = cookie_config.get('expiry_days')
    cookie_expiry_days = 30 # Sensible default

    valid_config = True
    if not cookie_name or not isinstance(cookie_name, str):
        st.error("Config Error: Invalid or missing 'name' in [cookie] section (must be a string).")
        logging.error(f"Invalid cookie name: {cookie_name} (type: {type(cookie_name)})")
        valid_config = False
    if not cookie_key or not isinstance(cookie_key, str):
        st.error("Config Error: Invalid or missing 'key' in [cookie] section (must be a string).")
        # Avoid logging the actual key for security, log its type if it exists
        key_type = type(cookie_key) if cookie_key else 'None'
        logging.error(f"Invalid cookie key type: {key_type}")
        valid_config = False
        # Optional: Warn if default cookie key is used
        if cookie_key == "choose_a_strong_random_secret_key" or cookie_key == "a_very_secret_random_string_12345":
            st.warning("⚠️ Security Warning: Please change the default cookie 'key' in secrets.toml to a unique, strong, random string!")
            logging.warning("Default cookie key is being used.")

    if cookie_expiry_days_raw is None:
        st.error("Config Error: Missing 'expiry_days' in [cookie] section.")
        logging.error("Missing cookie expiry_days.")
        valid_config = False
    else:
        # Try converting expiry_days to int, handle potential errors
        try:
            cookie_expiry_days = int(cookie_expiry_days_raw)
            if cookie_expiry_days <= 0:
                 st.warning(f"Config Warning: 'expiry_days' ({cookie_expiry_days}) should be positive. Using default {30}.")
                 logging.warning(f"Non-positive expiry_days '{cookie_expiry_days_raw}', defaulting to 30.")
                 cookie_expiry_days = 30
        except (ValueError, TypeError):
            # Use the default value if conversion fails
            st.warning(f"Config Warning: Could not parse 'expiry_days' as integer from secrets.toml (value: '{cookie_expiry_days_raw}'). Defaulting to {cookie_expiry_days} days. Please ensure it's a number in the file.")
            logging.warning(f"Could not parse cookie expiry_days '{cookie_expiry_days_raw}', defaulting to {cookie_expiry_days}.")
            # cookie_expiry_days already has the default value of 30 assigned earlier

    if not valid_config:
        st.error("Authentication configuration errors found. Please check secrets.toml and logs.")
        st.stop() # Stop if fundamental config issues are found

    # --- Log values just before Authenticate init ---
    logging.info("Attempting to initialize Authenticate with:")
    logging.info(f"  Credentials type: {type(credentials_config)}") # Should be dict
    logging.info(f"  Cookie Name: '{cookie_name}' (type: {type(cookie_name)})") # Should be str
    logging.info(f"  Cookie Key Type: {type(cookie_key)}") # Should be str - DO NOT LOG VALUE
    logging.info(f"  Cookie Expiry Days: {cookie_expiry_days} (type: {type(cookie_expiry_days)})") # Should be int

    # --- Initialize Authenticator Object ---
    authenticator = stauth.Authenticate(
        credentials_config,   # Use the variable holding the credentials dict
        cookie_name,          # Use validated variable
        cookie_key,           # Use validated variable
        cookie_expiry_days,   # Use validated/converted variable
        # preauthorized= # Optional: Add list of emails for pre-authorized users if needed
    )
    logging.info("Authenticator object initialized successfully.")

except Exception as auth_init_e:
    # Catch errors during the config validation or Authenticate init
    st.error(f"Fatal Error: Failed during authentication setup: {auth_init_e}")
    logging.error(f"Authenticator setup failed: {auth_init_e}", exc_info=True)
    st.stop()


# --- Render Login Form & Handle Authentication ---
# This section should now execute only if authenticator was initialized successfully
try:
    # The line that previously caused TypeError
    name, authentication_status, username = authenticator.login(location='main')
    logging.info(f"Login check complete. Status: {authentication_status}, User: {username}")

except TypeError as login_te:
    # Catch the specific TypeError if it still happens, providing more context
    st.error(f"Internal Error: TypeError during authenticator.login(): {login_te}. This might indicate an issue within the library or unexpected data state after initialization.")
    logging.error(f"TypeError during login call: {login_te}", exc_info=True)
    # Log the state again just before the call might help diagnose
    logging.error(f"State just before login call - Credentials type: {type(credentials_config)}, Cookie Name: {cookie_name}, Key Type: {type(cookie_key)}, Expiry: {cookie_expiry_days}")
    st.info("Please check the application logs for more details and ensure your secrets.toml file is correctly formatted, especially the [cookie] section.")
    st.stop()
except Exception as login_e:
     # Catch any other unexpected errors during login
     st.error(f"An unexpected error occurred during the login process: {login_e}")
     logging.error(f"Exception during login call: {login_e}", exc_info=True)
     st.info("Please check the application logs for more details.")
     st.stop()


# --- Authentication Status Check and Main Application Logic ---

if authentication_status == False:
    st.error('Username/password is incorrect')
    logging.warning(f"Failed login attempt for username: {username}")
    st.stop() # Stop execution if login fails

elif authentication_status == None:
    st.warning('Please enter your username and password to access the application.')
    # No st.stop() needed here, as the login form is already displayed
    # and the rest of the app logic is in the 'elif authentication_status:' block

elif authentication_status:
    # --- User is Authenticated --- #
    # Proceed with the main application logic and resource loading

    # Display welcome message and logout button (typically in the sidebar)
    st.sidebar.success(f"Welcome *{name}* 👋")
    authenticator.logout('Logout', 'sidebar')
    logging.info(f"User '{username}' authenticated successfully. Loading main application.")

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
            s3.list_buckets() # Quick check
            logging.info("S3 client initialized successfully.")
            return s3
        except Exception as e:
            logging.error(f"Error initializing S3 client: {str(e)}", exc_info=True)
            # Display error in the main area as well
            st.error(f"Failed to connect to S3. Check AWS credentials/permissions. Error: {e}")
            return None

    @st.cache_resource # Cache embeddings model for this session
    def get_embeddings_model():
        """
        Loads the HuggingFace embeddings model from the local directory specified
        by MODEL_DIRECTORY. Uses TRANSFORMERS_CACHE strategy.
        """
        model_path = MODEL_DIRECTORY
        abs_model_path = os.path.abspath(model_path)
        if not os.path.isdir(abs_model_path):
            st.error(f"Local embedding model directory not found: '{abs_model_path}'.")
            logging.error(f"Model directory '{abs_model_path}' not found.")
            return None
        try:
            cache_dir = os.path.abspath('.')
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logging.info(f"Setting TRANSFORMERS_CACHE to: {cache_dir}")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu', "local_files_only": True},
                encode_kwargs={'normalize_embeddings': True}
            )
            _ = embeddings.embed_query("Initialization test query") # Warm-up/check
            logging.info(f"Embeddings model '{model_path}' loaded successfully.")
            return embeddings
        except Exception as e:
            st.error(f"Failed to load local embeddings model from '{model_path}'. Error: {e}")
            logging.error(f"Failed to load embeddings model '{model_path}': {e}", exc_info=True)
            return None

    @st.cache_resource # Cache LLM model for this session
    def get_gemini_model():
        """Initializes and returns the Gemini LLM client."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            logging.info(f"Gemini model '{GEMINI_MODEL}' initialized.")
            return llm
        except Exception as e:
            st.error(f"Failed to initialize Gemini LLM '{GEMINI_MODEL}'. Check API Key. Error: {e}")
            logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            return None

    @st.cache_resource(ttl=3600)  # Cache the loaded index for 1 hour
    def download_and_load_faiss_index(_s3_client, _embeddings, bucket, s3_prefix):
        """
        Downloads FAISS index files (.faiss, .pkl) from S3, loads into memory.
        """
        if not _s3_client:
            st.error("S3 client unavailable. Cannot download FAISS index.")
            logging.error("FAISS load skipped: S3 client missing.")
            return None
        if not _embeddings:
            st.error("Embeddings model unavailable. Cannot load FAISS index.")
            logging.error("FAISS load skipped: Embeddings model missing.")
            return None

        s3_index_key = f"{s3_prefix}.faiss"
        s3_pkl_key = f"{s3_prefix}.pkl"
        local_faiss_filename = "index.faiss"
        local_pkl_filename = "index.pkl"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_index_path = os.path.join(temp_dir, local_faiss_filename)
                local_pkl_path = os.path.join(temp_dir, local_pkl_filename)
                logging.info(f"Downloading index: s3://{bucket}/{s3_index_key} & .pkl")
                _s3_client.download_file(bucket, s3_index_key, local_index_path)
                _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)
                logging.info(f"Downloaded index files to temp dir: {temp_dir}")
                vector_store = FAISS.load_local(
                    folder_path=temp_dir,
                    embeddings=_embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True # Required
                )
                logging.info(f"FAISS index loaded from {s3_prefix}.")
                return vector_store
        except _s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                st.error(f"FAISS index not found in s3://{bucket}/{s3_prefix}.(faiss/pkl). Check bucket, path, and ensure index exists.")
                logging.error(f"FAISS index files not found at s3://{bucket}/{s3_prefix}.")
            elif error_code == 'NoSuchBucket':
                 st.error(f"S3 bucket '{bucket}' not found. Check S3_BUCKET name.")
                 logging.error(f"S3 bucket '{bucket}' not found.")
            elif error_code in ['NoCredentialsError','InvalidAccessKeyId','SignatureDoesNotMatch','ExpiredToken']:
                 st.error(f"AWS S3 Auth Error: {error_code}. Check credentials/permissions.")
                 logging.error(f"S3 Auth Error: {e}")
            else:
                st.error(f"S3 Error downloading FAISS index: {e}")
                logging.error(f"S3 ClientError downloading index: {e}", exc_info=True)
            return None
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            logging.error(f"Error loading FAISS index: {e}", exc_info=True)
            if "allow_dangerous_deserialization" in str(e):
                 st.error("Deserialization error. Ensure 'allow_dangerous_deserialization=True'.")
            elif "Can't get attribute" in str(e) or "ModuleNotFoundError" in str(e):
                 st.error("Deserialization Error: Library mismatch? Ensure consistent LangChain/FAISS versions.")
                 logging.error("Potential library mismatch during FAISS deserialization.")
            return None

    # --- Helper Functions ---

    def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
        """Queries the FAISS index, returns retrieved documents."""
        if not vector_store:
            st.warning("Knowledge base is not loaded. Cannot search.")
            return []
        try:
            search_type = 'MMR' if use_mmr else 'Similarity'
            logging.info(f"Performing {search_type} search (k={k}): '{query_text}'")
            if use_mmr:
                results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k*4)
            else:
                results = vector_store.similarity_search(query_text, k=k)
            logging.info(f"Retrieved {len(results)} chunks.")
            return results
        except Exception as e:
            st.error(f"Error querying knowledge base: {str(e)}")
            logging.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
            return []

    def generate_llm_response(llm, query_text, retrieved_docs):
        """Generates a response from the LLM using the query and retrieved context."""
        if not llm:
            return "LLM is not available. Cannot generate response."

        if not retrieved_docs:
            logging.info(f"Generating response for '{query_text}' without context.")
            messages = [
                SystemMessage(content="You are an AI assistant answering questions about Proforma Invoices. No relevant context documents were found."),
                HumanMessage(content=query_text + "\n\nBased on the available knowledge base, I could not find specific information relevant to your query.")
            ]
        else:
            context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
                                         for doc in retrieved_docs if hasattr(doc, 'metadata')])
            system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based *only* on the provided context documents.
            Analyze the user's query and the following context documents carefully.
            Synthesize information from relevant parts of the context for a comprehensive answer.
            Quote relevant snippets or data points, mentioning the source document if possible.
            If the answer cannot be fully found, state what is available and what is missing based *solely* on the provided context.
            Do not make assumptions or use any external knowledge. Stick strictly to the information given below.

            Provided Context Documents:
            ---
            {context}
            ---
            """
            messages = [ SystemMessage(content=system_prompt), HumanMessage(content=query_text) ]
            logging.info(f"Generating response for '{query_text}' with {len(retrieved_docs)} chunks.")

        try:
            ai_response: AIMessage = llm.invoke(messages)
            logging.info(f"Successfully generated LLM response for '{query_text}'")
            return ai_response.content
        except Exception as e:
            st.error(f"Error generating LLM response: {e}")
            logging.error(f"LLM invocation error for '{query_text}': {e}", exc_info=True)
            if "quota" in str(e).lower():
                 return "Sorry, cannot generate response due to API quota limits."
            return "Sorry, an error occurred while generating the response."

    # --- Main Application UI (Authenticated Access) ---

    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown(f"Ask questions about the proforma invoices.") # Welcome message in sidebar now

    # --- Load Resources & Display Status ---
    vector_store = None # Initialize
    initialization_ok = False
    with st.status("Initializing resources...", expanded=True) as status_container:
        st.write("Connecting to S3...")
        s3_client = get_s3_client()
        st.write("✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed")

        st.write("Loading Local Embedding Model...")
        embeddings = get_embeddings_model()
        st.write("✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed")

        st.write("Initializing LLM...")
        gemini_model = get_gemini_model()
        st.write("✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed")

        if s3_client and embeddings and gemini_model:
            st.write("Loading Knowledge Base Index from S3...")
            vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
            st.write("✅ Knowledge Base Index Loaded" if vector_store else "❌ Knowledge Base Index Failed")

            if vector_store:
                status_container.update(label="Initialization Complete!", state="complete", expanded=False)
                initialization_ok = True
            else:
                 status_container.update(label="Initialization Failed (Index Load)", state="error", expanded=True)
                 st.error("Failed to load the knowledge base index. Querying disabled.")
        else:
            status_container.update(label="Initialization Failed (Core Components)", state="error", expanded=True)
            st.error("Core components failed (S3, Embeddings, or LLM). Querying disabled.")

    # --- Query Interface ---
    st.markdown("---")
    query_disabled = not initialization_ok # Disable if any core step failed
    query_placeholder = "e.g., What is the total amount for invoice [filename]?" if initialization_ok else "Application initialization failed. Querying disabled."

    query_text = st.text_input(
        "Enter your query:",
        placeholder=query_placeholder,
        key="query_input",
        disabled=query_disabled
    )

    # --- Advanced Settings (Consider Sidebar) ---
    k_results = 15
    use_mmr_search = False

    # --- Execute Query and Display Results ---
    if query_text and initialization_ok: # Check initialization_ok flag
        # 1. Query FAISS index
        with st.spinner(f"Searching knowledge base..."):
            retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

        # 2. Generate LLM response
        with st.spinner("🧠 Thinking..."):
            response = generate_llm_response(gemini_model, query_text, retrieved_docs)

        # 3. Display response
        st.markdown("### Response:")
        st.markdown(response)
        st.markdown("---")

        # 4. Optional: Display retrieved context
        if retrieved_docs:
            with st.expander("🔍 Show Retrieved Context Snippets"):
                st.markdown(f"Retrieved {len(retrieved_docs)} snippets:")
                for i, doc in enumerate(retrieved_docs):
                    source_info = "Unknown Source"
                    page_info = ""
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                         source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                         page_num = doc.metadata.get('page')
                         if page_num is not None:
                              try:
                                   page_info = f", Page: {int(page_num) + 1}" # Often 0-indexed
                              except (ValueError, TypeError):
                                   page_info = f", Page: {page_num}" # Display as is if not int

                    st.text_area(
                        label=f"**Snippet {i + 1}** ({source_info}{page_info})",
                        value=doc.page_content,
                        height=150,
                        key=f"snippet_{i}",
                        disabled=True
                    )
        else:
            st.info("No specific context snippets were found in the knowledge base for this query to display.")

# --- End of Authenticated Section ---

logging.info("Application loop finished or waiting for user input/action.")