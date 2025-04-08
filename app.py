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
# Use DEBUG level for detailed authenticator library logs and credential structure
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("streamlit_authenticator").setLevel(logging.DEBUG) # Get library logs
logger = logging.getLogger(__name__) # Get logger for this script
logger.info("Application starting up...")

# --- Load Secrets and Basic Configuration ---
try:
    secrets = toml.load(SECRETS_FILE_PATH)

    AWS_ACCESS_KEY = secrets.get("access_key_id")
    AWS_SECRET_KEY = secrets.get("secret_access_key")
    GEMINI_API_KEY = secrets.get("gemini_api_key")
    credentials_config = secrets.get('credentials')
    cookie_config = secrets.get('cookie')

    # --- Validate Essential Secrets ---
    missing_secrets = []
    if not AWS_ACCESS_KEY: missing_secrets.append("access_key_id")
    if not AWS_SECRET_KEY: missing_secrets.append("secret_access_key")
    if not GEMINI_API_KEY: missing_secrets.append("gemini_api_key")
    if not credentials_config: missing_secrets.append("[credentials] section")
    if not cookie_config: missing_secrets.append("[cookie] section")

    if missing_secrets:
         error_message = f"Missing required configuration in {SECRETS_FILE_PATH}: {', '.join(missing_secrets)}."
         st.error(error_message)
         logger.error(error_message)
         st.stop()

    logger.info("Basic secrets loaded.")

except FileNotFoundError:
    st.error(f"❌ Critical Error: Secrets file not found at {SECRETS_FILE_PATH}. App cannot run.")
    logger.error(f"Secrets file not found at {SECRETS_FILE_PATH}.")
    st.stop()
except toml.TomlDecodeError as e:
     st.error(f"❌ Critical Error: Invalid TOML format in {SECRETS_FILE_PATH}. Please check the syntax. Error: {e}")
     logger.error(f"Invalid TOML format in secrets file: {e}")
     st.stop()
except Exception as e:
     st.error(f"❌ Critical Error loading secrets or initial configuration: {e}")
     logger.error(f"Unexpected error loading secrets: {e}", exc_info=True)
     st.stop()

# --- Initialize Authenticator ---
authenticator = None # Initialize to None
try:
    # --- Validate Cookie Config ---
    cookie_name = cookie_config.get('name')
    cookie_key = cookie_config.get('key')
    cookie_expiry_days_raw = cookie_config.get('expiry_days')
    cookie_expiry_days = 30 # Sensible default

    valid_config = True
    error_messages = []
    if not cookie_name or not isinstance(cookie_name, str):
        error_messages.append("Invalid or missing 'name' in [cookie] (must be string).")
        logger.error(f"Invalid cookie name: {cookie_name} (type: {type(cookie_name)})")
        valid_config = False
    if not cookie_key or not isinstance(cookie_key, str):
        error_messages.append("Invalid or missing 'key' in [cookie] (must be string).")
        key_type = type(cookie_key) if cookie_key else 'None'
        logger.error(f"Invalid cookie key type: {key_type}")
        valid_config = False
        # Warn about default keys
        if cookie_key in ["choose_a_strong_random_secret_key", "a_very_secret_random_string_12345", ""]:
            st.warning("⚠️ Security Warning: Please change the default/empty cookie 'key' in secrets.toml to a unique, strong, random string!")
            logger.warning("Default or empty cookie key is being used.")

    if cookie_expiry_days_raw is None:
        error_messages.append("Missing 'expiry_days' in [cookie].")
        logger.error("Missing cookie expiry_days.")
        valid_config = False
    else:
        try:
            cookie_expiry_days = int(cookie_expiry_days_raw)
            if cookie_expiry_days <= 0:
                 st.warning(f"Config Warning: 'expiry_days' ({cookie_expiry_days}) should be positive. Using default {30}.")
                 logger.warning(f"Non-positive expiry_days '{cookie_expiry_days_raw}', defaulting to 30.")
                 cookie_expiry_days = 30
        except (ValueError, TypeError):
            st.warning(f"Config Warning: Could not parse 'expiry_days' as integer from secrets.toml (value: '{cookie_expiry_days_raw}'). Defaulting to {cookie_expiry_days} days.")
            logger.warning(f"Could not parse cookie expiry_days '{cookie_expiry_days_raw}', defaulting to {cookie_expiry_days}.")
            # Default already set

    if not valid_config:
        for msg in error_messages:
            st.error(f"Config Error: {msg}")
        st.error("Authentication configuration errors found. Please check secrets.toml and logs.")
        st.stop()

    # --- Log values just before Authenticate init ---
    logger.info("Attempting to initialize Authenticate with:")
    logger.info(f"  Credentials type: {type(credentials_config)}")
    # *** CRUCIAL DEBUG LOG ***: Check the output of this log in your terminal
    logger.debug(f"  Credentials value structure being passed: {credentials_config}")
    # *************************
    logger.info(f"  Cookie Name: '{cookie_name}' (type: {type(cookie_name)})")
    logger.info(f"  Cookie Key Type: {type(cookie_key)}") # DO NOT LOG VALUE
    logger.info(f"  Cookie Expiry Days: {cookie_expiry_days} (type: {type(cookie_expiry_days)})")

    # --- Initialize Authenticator Object ---
    authenticator = stauth.Authenticate(
        credentials_config,
        cookie_name,
        cookie_key,
        cookie_expiry_days,
        # preauthorized= # Optional: secrets.get('preauthorized', {'emails': []})
    )
    logger.info("Authenticator object initialized successfully.")

except Exception as auth_init_e:
    st.error(f"Fatal Error: Failed during authentication setup: {auth_init_e}")
    logger.error(f"Authenticator setup failed: {auth_init_e}", exc_info=True)
    # Specific check for common credential structure error
    if isinstance(auth_init_e, KeyError) and 'usernames' in str(auth_init_e):
         st.error("Potential Cause: The '[credentials]' section in secrets.toml might be missing the required 'usernames' structure.")
         logger.error("KeyError 'usernames' suggests incorrect credentials structure in secrets.toml.")
    st.stop()

# --- Render Login Form & Handle Authentication ---
# This section requires the authenticator object to be initialized successfully
if authenticator:
    name = None
    authentication_status = None
    username = None
    try:
        # Assign the return value to a single variable first
        login_result = authenticator.login(location='main')

        # Log what was returned - vital for debugging the NoneType error
        logger.info(f"authenticator.login() returned: {login_result} (type: {type(login_result)})")

        # **** This is the crucial check that handles the reported error ****
        if login_result is None:
            st.error("Authentication service encountered an issue (login returned None). Cannot proceed.")
            # This critical log confirms the check is working when you see the error
            logger.error("CRITICAL: authenticator.login() unexpectedly returned None. Aborting.")
            st.info("Possible causes: Incorrect [credentials] structure in secrets.toml (check debug logs above), session state problems, or browser cookie issues. Try clearing browser cookies for this site. Verify library versions.")
            st.stop() # Stop execution as we cannot determine login state
        else:
            # If login_result is not None, proceed with unpacking
            name, authentication_status, username = login_result
            logger.info(f"Login result unpacked. Status: {authentication_status}, User: {username}, Name: {name}")

    except Exception as login_e:
         st.error(f"An unexpected error occurred during the login process: {login_e}")
         logger.error(f"Exception during login call: {login_e}", exc_info=True)
         st.info("Please check the application logs for more details.")
         st.stop()

    # --- Authentication Status Check and Main Application Logic ---
    if authentication_status == False:
        st.error('Username/password is incorrect')
        logger.warning(f"Failed login attempt for username: {username}") # username might be None if field was empty
        # No st.stop() here, allows user to retry

    elif authentication_status == None:
        st.warning('Please enter your username and password to access the application.')
        logger.info("Login form displayed, awaiting user input.")
        # No st.stop() here, login form is active

    elif authentication_status:
        # --- User is Authenticated --- #
        logger.info(f"User '{username}' authenticated successfully. Loading main application.")

        # Display welcome message and logout button
        st.sidebar.success(f"Welcome *{name}* 👋")
        authenticator.logout('Logout', 'sidebar', key='logout_button') # Add a key for uniqueness

        # --- Resource Initialization Functions (Cached) ---

        @st.cache_resource
        def get_s3_client():
            """Initializes and returns the Boto3 S3 client."""
            try:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    # region_name='your-region' # Optional: Add region
                )
                s3.list_buckets() # Quick check to verify connection and credentials
                logger.info("S3 client initialized successfully.")
                return s3
            except Exception as e:
                logger.error(f"Error initializing S3 client: {str(e)}", exc_info=True)
                st.error(f"Failed to connect to S3. Check AWS credentials/permissions. Error: {e}")
                return None

        @st.cache_resource
        def get_embeddings_model():
            """Loads the HuggingFace embeddings model locally."""
            model_path = MODEL_DIRECTORY
            abs_model_path = os.path.abspath(model_path)
            if not os.path.isdir(abs_model_path):
                st.error(f"Local embedding model directory not found: '{abs_model_path}'.")
                logger.error(f"Model directory '{abs_model_path}' not found.")
                return None
            try:
                # Cache directory for transformers - use project root or a dedicated cache dir
                cache_dir = os.environ.get('TRANSFORMERS_CACHE', os.path.abspath('.'))
                os.makedirs(cache_dir, exist_ok=True) # Ensure cache dir exists
                logger.info(f"Using TRANSFORMERS_CACHE: {cache_dir}")

                embeddings = HuggingFaceEmbeddings(
                    model_name=model_path, # Use the path to the local directory
                    model_kwargs={'device': 'cpu'}, # Or 'cuda' if GPU is available
                    encode_kwargs={'normalize_embeddings': True} # Recommended for similarity search
                    # Removed local_files_only=True as it might cause issues if model files need checking/update
                    # Rely on model_name being a *path* to use local files primarily.
                )
                # Test embedding
                _ = embeddings.embed_query("Initialization test query")
                logger.info(f"Embeddings model '{model_path}' loaded successfully.")
                return embeddings
            except Exception as e:
                st.error(f"Failed to load local embeddings model from '{model_path}'. Error: {e}")
                logger.error(f"Failed to load embeddings model '{model_path}': {e}", exc_info=True)
                return None

        @st.cache_resource
        def get_gemini_model():
            """Initializes and returns the Gemini LLM client."""
            try:
                llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3,
                    convert_system_message_to_human=True
                )
                # Optional: Add a simple test invocation if needed
                # llm.invoke("Hello!")
                logger.info(f"Gemini model '{GEMINI_MODEL}' initialized.")
                return llm
            except Exception as e:
                st.error(f"Failed to initialize Gemini LLM '{GEMINI_MODEL}'. Check API Key/Quota. Error: {e}")
                logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
                return None

        @st.cache_resource(ttl=3600)  # Cache the loaded index for 1 hour
        def download_and_load_faiss_index(_s3_client, _embeddings, bucket, s3_prefix):
            """Downloads FAISS index files (.faiss, .pkl) from S3, loads into memory."""
            if not _s3_client:
                st.error("S3 client unavailable. Cannot download FAISS index.")
                logger.error("FAISS load skipped: S3 client missing.")
                return None
            if not _embeddings:
                st.error("Embeddings model unavailable. Cannot load FAISS index.")
                logger.error("FAISS load skipped: Embeddings model missing.")
                return None

            s3_index_key = f"{s3_prefix}.faiss"
            s3_pkl_key = f"{s3_prefix}.pkl"
            local_index_name = "index" # Base name for local files

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_index_path = os.path.join(temp_dir, f"{local_index_name}.faiss")
                    local_pkl_path = os.path.join(temp_dir, f"{local_index_name}.pkl")

                    logger.info(f"Attempting download: s3://{bucket}/{s3_index_key} -> {local_index_path}")
                    _s3_client.download_file(bucket, s3_index_key, local_index_path)

                    logger.info(f"Attempting download: s3://{bucket}/{s3_pkl_key} -> {local_pkl_path}")
                    _s3_client.download_file(bucket, s3_pkl_key, local_pkl_path)

                    logger.info(f"Downloaded index files to temp dir: {temp_dir}. Loading FAISS index.")
                    vector_store = FAISS.load_local(
                        folder_path=temp_dir,
                        embeddings=_embeddings,
                        index_name=local_index_name, # Match local base name
                        allow_dangerous_deserialization=True # Required for loading pickle
                    )
                    logger.info(f"FAISS index loaded successfully from S3 path {s3_prefix}.")
                    return vector_store
            except _s3_client.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                error_message = e.response.get('Error', {}).get('Message', 'Unknown S3 Client Error')
                logger.error(f"S3 ClientError downloading/accessing index '{s3_prefix}': {error_code} - {error_message}", exc_info=True)
                if error_code == '404' or 'NoSuchKey' in str(e):
                    st.error(f"FAISS index not found in s3://{bucket}/{s3_prefix}.(faiss/pkl). Verify path and file existence.")
                elif error_code == 'NoSuchBucket':
                     st.error(f"S3 bucket '{bucket}' not found. Check S3_BUCKET name.")
                elif error_code in ['NoCredentialsError','InvalidAccessKeyId','SignatureDoesNotMatch','ExpiredToken','AccessDenied']:
                     st.error(f"AWS S3 Authentication/Authorization Error: {error_code}. Check credentials and bucket permissions.")
                else:
                    st.error(f"S3 Error downloading FAISS index ({error_code}): {error_message}")
                return None
            except ImportError as e:
                 st.error(f"ImportError during FAISS load: {e}. Required library missing? Ensure 'faiss-cpu' or 'faiss-gpu' is installed.")
                 logger.error(f"ImportError loading FAISS index: {e}", exc_info=True)
                 return None
            except Exception as e:
                st.error(f"Error loading FAISS index: {e}")
                logger.error(f"Error loading FAISS index from {temp_dir}: {e}", exc_info=True)
                if "allow_dangerous_deserialization" in str(e):
                     st.warning("Suggestion: Ensure 'allow_dangerous_deserialization=True' is set during FAISS.load_local().") # Already set, but good reminder
                elif "Can't get attribute" in str(e) or "ModuleNotFoundError" in str(e) or isinstance(e, pickle.UnpicklingError):
                     st.error("Deserialization Error: Possible mismatch in library versions (e.g., langchain, faiss, sentence-transformers) between index creation and loading environment.")
                     logger.error("Potential library mismatch during FAISS deserialization.", exc_info=True)
                return None

        # --- Helper Functions ---

        def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
            """Queries the FAISS index, returns retrieved documents."""
            if not vector_store:
                st.warning("Knowledge base is not loaded. Cannot search.")
                return []
            try:
                search_type = 'MMR' if use_mmr else 'Similarity'
                logger.info(f"Performing {search_type} search (k={k}): '{query_text[:50]}...'") # Log truncated query
                if use_mmr:
                    # Fetch more initially for MMR to re-rank from
                    results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=max(k*4, 20))
                else:
                    results = vector_store.similarity_search(query_text, k=k)
                logger.info(f"Retrieved {len(results)} chunks.")
                return results
            except Exception as e:
                st.error(f"Error querying knowledge base: {str(e)}")
                logger.error(f"Error querying FAISS index: {str(e)}", exc_info=True)
                return []

        def generate_llm_response(llm, query_text, retrieved_docs):
            """Generates a response from the LLM using the query and retrieved context."""
            if not llm:
                 logger.warning("LLM is not available, cannot generate response.")
                 return "Error: LLM client is not initialized."

            if not retrieved_docs:
                logger.info(f"Generating response for '{query_text[:50]}...' without specific context.")
                prompt = f"""You are an AI assistant answering questions about Proforma Invoices.
                No specific context documents were found in the knowledge base for the user's query.
                Answer the user's query based on your general knowledge about proforma invoices, but clearly state that you could not find specific information in the provided documents.

                User Query: {query_text}
                """
                messages = [HumanMessage(content=prompt)] # Simple prompt for no context
            else:
                context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
                                             for doc in retrieved_docs if hasattr(doc, 'metadata') and hasattr(doc, 'page_content')])
                system_prompt = f"""You are an AI assistant specialized in answering questions about Proforma Invoices based *only* on the provided context documents.
                Analyze the user's query and the following context documents carefully.
                Synthesize information from relevant parts of the context for a comprehensive answer.
                Quote relevant snippets or data points directly from the context, mentioning the source document and page number (e.g., "According to Source: X, Page: Y, ...").
                If the answer cannot be fully found within the provided context, state what information is available and clearly indicate what is missing based *solely* on the provided documents.
                Do not make assumptions or use any external knowledge beyond what is given in the context. Stick strictly to the information given below.

                Provided Context Documents:
                ---
                {context}
                ---
                """
                messages = [ SystemMessage(content=system_prompt), HumanMessage(content=query_text) ]
                logger.info(f"Generating response for '{query_text[:50]}...' with {len(retrieved_docs)} context chunks.")

            try:
                ai_response = llm.invoke(messages)
                response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
                logger.info(f"Successfully generated LLM response for '{query_text[:50]}...'")
                return response_content
            except Exception as e:
                st.error(f"Error generating LLM response: {e}")
                logger.error(f"LLM invocation error for '{query_text[:50]}...': {e}", exc_info=True)
                # Check for common API errors
                if "quota" in str(e).lower():
                    return "Sorry, cannot generate response due to API quota limits."
                elif "api key" in str(e).lower():
                     return "Sorry, there seems to be an issue with the LLM API key configuration."
                return "Sorry, an error occurred while generating the response from the AI model."

        # --- Main Application UI (Authenticated Access) ---

        st.title("📄 Proforma Invoice Query Assistant")
        st.markdown(f"Ask questions about the proforma invoices. You are logged in as **{username}**.")

        # --- Load Resources & Display Status ---
        vector_store = None # Initialize
        initialization_ok = False
        with st.status("Initializing resources...", expanded=True) as status_container:
            st.write("1. Connecting to S3...")
            s3_client = get_s3_client()
            st.write("✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed")

            st.write("2. Loading Local Embedding Model...")
            embeddings = get_embeddings_model()
            st.write("✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed")

            st.write("3. Initializing LLM...")
            gemini_model = get_gemini_model()
            st.write("✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed")

            if s3_client and embeddings and gemini_model:
                st.write("4. Loading Knowledge Base Index from S3...")
                vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
                st.write("✅ Knowledge Base Index Loaded" if vector_store else "❌ Knowledge Base Index Failed")

                if vector_store:
                    status_container.update(label="Initialization Complete! Ready to query.", state="complete", expanded=False)
                    initialization_ok = True
                    logger.info("All resources initialized successfully.")
                else:
                     status_container.update(label="Initialization Failed (Index Load Error)", state="error", expanded=True)
                     st.error("Failed to load the knowledge base index. Querying is disabled. Check logs for details.")
                     logger.error("Initialization failed: Could not load FAISS index.")
            else:
                failed_components = []
                if not s3_client: failed_components.append("S3 Client")
                if not embeddings: failed_components.append("Embeddings Model")
                if not gemini_model: failed_components.append("LLM")
                status_container.update(label="Initialization Failed (Core Component Error)", state="error", expanded=True)
                st.error(f"Core components failed: {', '.join(failed_components)}. Querying is disabled. Check logs.")
                logger.error(f"Initialization failed on core components: {', '.join(failed_components)}")

        # --- Query Interface ---
        st.markdown("---")
        query_disabled = not initialization_ok
        query_placeholder = "e.g., What is the total amount for invoice [filename]?" if initialization_ok else "Application initialization failed. Querying disabled."

        query_text = st.text_input(
            "Enter your query:",
            placeholder=query_placeholder,
            key="query_input",
            disabled=query_disabled,
            # on_change=clear_results # Optional: clear results when input changes
        )

        # --- Advanced Settings (Consider Sidebar or Expander) ---
        with st.sidebar.expander("Advanced Settings"):
            k_results = st.slider("Number of context snippets (k):", min_value=1, max_value=20, value=10, step=1, disabled=query_disabled)
            use_mmr_search = st.checkbox("Use MMR for diverse results", value=False, disabled=query_disabled)

        # --- Execute Query and Display Results ---
        if query_text and initialization_ok:
            logger.info(f"User '{username}' submitted query: '{query_text[:100]}...'")
            # 1. Query FAISS index
            with st.spinner(f"Searching knowledge base..."):
                retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)

            # 2. Generate LLM response
            with st.spinner("🧠 Thinking... Generating response..."):
                response = generate_llm_response(gemini_model, query_text, retrieved_docs)

            # 3. Display response
            st.markdown("### Response:")
            st.markdown(response) # Use markdown for potential formatting from LLM
            st.markdown("---")

            # 4. Optional: Display retrieved context
            if retrieved_docs:
                with st.expander("🔍 Show Retrieved Context Snippets"):
                    st.markdown(f"Retrieved {len(retrieved_docs)} snippets ({'MMR' if use_mmr_search else 'Similarity'} search):")
                    for i, doc in enumerate(retrieved_docs):
                        source_info = "Unknown Source"
                        page_info = ""
                        content = "N/A"
                        score_info = "" # Add score if available (e.g., from similarity_search_with_score)

                        if hasattr(doc, 'page_content'):
                            content = doc.page_content
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                             source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                             page_num = doc.metadata.get('page') # Page numbers are often 0-indexed in extraction
                             if page_num is not None:
                                 try:
                                      # Display page number + 1 for user-friendliness
                                      page_info = f", Page: {int(page_num) + 1}"
                                 except (ValueError, TypeError):
                                      page_info = f", Page: {page_num}" # Display raw if not int

                        st.text_area(
                            label=f"**Snippet {i + 1}** ({source_info}{page_info}{score_info})",
                            value=content,
                            height=150,
                            key=f"snippet_{i}",
                            disabled=True # Make text area read-only
                        )
            elif query_text: # Only show this if a query was actually entered
                 st.info("No specific context snippets were retrieved from the knowledge base for this query.")

# --- End of Authenticated Section ---
# If authenticator failed to initialize, this point might be reached without login form
elif not authenticator:
     st.error("Application failed to initialize the authentication service. Cannot proceed.")
     logger.critical("Authenticator object is None, login cannot be displayed.")

# Add a final log message for the end of a script run
logger.info("Streamlit script execution completed.")