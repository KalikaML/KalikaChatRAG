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
import googlemaps
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"

try:
    secrets = toml.load(SECRETS_FILE_PATH)
    # Invoice Query Settings
    S3_BUCKET = "kalika-rag"
    S3_PROFORMA_INDEX_PATH = "faiss_indexes/proforma_faiss_index"
    MODEL_DIRECTORY = "BAAI/BAAI-bge-base-en-v1.5"
    AWS_ACCESS_KEY = secrets.get("access_key_id")
    AWS_SECRET_KEY = secrets.get("secret_access_key")
    GEMINI_MODEL = "gemini-1.5-pro"
    GEMINI_API_KEY = secrets.get("gemini_api_key")
    # Place Data Extractor Settings
    MAPS_API_KEY = secrets.get("Maps_API_KEY")
    # Authentication Credentials
    CREDENTIALS = secrets.get("credentials", {}).get("usernames", {
        "user1": {
            "name": "User",
            "email": "user@example.com",
            "password": hashlib.sha256("user@123".encode()).hexdigest()
        }
    })

    if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, GEMINI_API_KEY, MAPS_API_KEY]):
        st.error("Missing required keys in secrets.toml: Ensure 'access_key_id', 'secret_access_key', 'gemini_api_key', and 'Maps_API_KEY' are defined.")
        st.stop()

except FileNotFoundError:
    st.error(f"Secrets file not found at {SECRETS_FILE_PATH}. Create a `.streamlit/secrets.toml` with required keys.")
    st.stop()
except toml.TomlDecodeError:
    st.error(f"Invalid TOML syntax in {SECRETS_FILE_PATH}. Check for formatting errors.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret key in {SECRETS_FILE_PATH}: {e}.")
    st.stop()

# --- Authentication Functions ---
def verify_password(username, password):
    if username not in CREDENTIALS:
        logging.warning(f"Login attempt for non-existent user: {username}")
        return False
    stored_hashed_password = CREDENTIALS[username]["password"]
    input_password_hash = hashlib.sha256(password.encode()).hexdigest()
    return input_password_hash == stored_hashed_password

def get_user_info(username):
    return CREDENTIALS[username]["name"] if username in CREDENTIALS else None

# --- Initialize S3 Client ---
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
        st.error(f"Failed to connect to S3: {e}")
        return None

# --- Initialize Embeddings Model ---
@st.cache_resource
def get_embeddings_model():
    model_path = MODEL_DIRECTORY
    if not os.path.isdir(model_path):
        st.error(f"Model directory '{model_path}' not found. Ensure '{MODEL_DIRECTORY}' exists.")
        logging.error(f"Model directory {model_path} not found.")
        return None
    try:
        cache_dir = os.path.abspath('.')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu', "local_files_only": True},
            encode_kwargs={'normalize_embeddings': True}
        )
        _ = embeddings.embed_query("Test query")
        logging.info(f"Embeddings model '{model_path}' loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        logging.error(f"Failed to load embeddings: {e}", exc_info=True)
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
        st.error(f"Failed to initialize Gemini model: {e}")
        logging.error(f"Failed to initialize Gemini: {e}")
        return None

# --- Initialize Google Maps Client ---
@st.cache_resource
def get_gmaps_client():
    try:
        gmaps = googlemaps.Client(key=MAPS_API_KEY)
        logging.info("Google Maps API client initialized.")
        return gmaps
    except Exception as e:
        st.error(f"Failed to initialize Google Maps client: {e}")
        logging.error(f"Failed to initialize Google Maps client: {e}")
        return None

# --- FAISS Index Loading ---
@st.cache_resource(ttl=3600)
def download_and_load_faiss_index(_s3_client, _embeddings, bucket, prefix):
    if not _embeddings or not _s3_client:
        st.error("Cannot load FAISS index: Missing embeddings or S3 client.")
        return None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_index_path = os.path.join(temp_dir, "index.faiss")
            local_pkl_path = os.path.join(temp_dir, "index.pkl")
            logging.info(f"Downloading index from s3://{bucket}/{prefix}")
            _s3_client.download_file(bucket, f"{prefix}.faiss", local_index_path)
            _s3_client.download_file(bucket, f"{prefix}.pkl", local_pkl_path)
            vector_store = FAISS.load_local(
                folder_path=temp_dir,
                embeddings=_embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info("FAISS index loaded successfully.")
            return vector_store
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        logging.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None

# --- Invoice Query Functions ---
def query_faiss_index(vector_store, query_text, k=10, use_mmr=False):
    if not vector_store:
        logging.warning("query_faiss_index called but vector_store is None.")
        return []
    try:
        search_kwargs = {'k': k}
        search_type = 'similarity' if not use_mmr else 'mmr'
        logging.info(f"Performing {search_type} search with k={k} for query: '{query_text}'")
        results = vector_store.max_marginal_relevance_search(query_text, k=k, fetch_k=k*4) if use_mmr else vector_store.similarity_search(query_text, k=k)
        logging.info(f"Retrieved {len(results)} chunks.")
        return results
    except Exception as e:
        st.error(f"Error querying FAISS index: {str(e)}")
        logging.error(f"Error querying FAISS: {e}", exc_info=True)
        return []

def generate_follow_up_questions(llm, query_text, response_text, retrieved_docs):
    if not llm:
        return []
    context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]]) if retrieved_docs else ""
    try:
        follow_up_prompt = f"""Based on the user query and response, generate 3 specific, relevant follow-up questions about invoices.
        Previous Query: {query_text}
        Response: {response_text}
        {f'Context: {context[:500]}...' if context else ''}
        Provide three questions in a list format, one per line, related to invoices."""
        messages = [
            SystemMessage(content="You generate relevant follow-up questions about invoices."),
            HumanMessage(content=follow_up_prompt)
        ]
        ai_response = llm.invoke(messages)
        questions = [line.strip("0123456789.-*• ") for line in ai_response.content.strip().split('\n') if line.strip() and '?' in line]
        return questions[:3]
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}", exc_info=True)
        return []

def generate_llm_response(llm, query_text, retrieved_docs):
    if not llm:
        return "LLM model is not available."
    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        system_prompt = f"""You answer questions about Proforma Invoices based only on the provided context.
        Context:
        ---
        {context}
        ---
        If the answer isn't in the context, state what is available and what cannot be answered."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text)
        ]
        logging.info(f"Generating response with {len(retrieved_docs)} context chunks.")
    else:
        messages = [
            SystemMessage(content="No relevant context documents found."),
            HumanMessage(content=f"{query_text}\n\nPlease state that you cannot answer based on the available knowledge base.")
        ]
    try:
        return llm.invoke(messages).content
    except Exception as e:
        st.error(f"Error generating LLM response: {e}")
        logging.error(f"LLM error: {e}", exc_info=True)
        return "Error generating response."

# --- Place Data Extractor Functions ---
PLACE_DETAIL_FIELDS = [
    'place_id', 'name', 'formatted_address', 'geometry/location', 'rating',
    'user_ratings_total', 'website', 'formatted_phone_number', 'business_status',
    'opening_hours'
]

def find_place_ids_by_text_search(gmaps, query, max_results=5):
    place_ids = []
    try:
        results = gmaps.places(query=query)
        if results.get('status') == 'OK':
            place_ids = [result['place_id'] for result in results.get('results', [])[:max_results]]
            st.write(f"  Found {len(place_ids)} place(s) for '{query}'.")
        elif results.get('status') == 'ZERO_RESULTS':
            st.warning(f"No results for query: '{query}'")
        else:
            st.error(f"Google Maps API error for '{query}': {results.get('status')}")
    except Exception as e:
        st.error(f"Error during text search for '{query}': {e}")
    return place_ids

def extract_emails_from_website(website_url):
    emails = set()
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()
        if 'html' not in response.headers.get('content-type', '').lower():
            return []
        soup = BeautifulSoup(response.content, 'html.parser')
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}"
        emails.update(re.findall(email_pattern, soup.get_text()))
        for a_tag in soup.find_all('a', href=True):
            if a_tag['href'].lower().startswith('mailto:'):
                email_match = re.search(email_pattern, a_tag['href'])
                if email_match:
                    emails.add(email_match.group(0))
    except Exception as e:
        st.warning(f"Error extracting emails from {website_url}: {e}")
    return list(emails)

def extract_social_links_from_website(website_url):
    social_links = {"facebook": None, "instagram": None, "twitter": None}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()
        if 'html' not in response.headers.get('content-type', '').lower():
            return social_links
        soup = BeautifulSoup(response.content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].lower()
            href_original = a_tag['href']
            if not social_links["facebook"] and "facebook.com/" in href and "sharer" not in href:
                social_links["facebook"] = href_original
            elif not social_links["instagram"] and "instagram.com/" in href and "p/" not in href:
                social_links["instagram"] = href_original
            elif not social_links["twitter"] and ("twitter.com/" in href or "x.com/" in href) and "intent" not in href:
                social_links["twitter"] = href_original
            if all(social_links.values()):
                break
    except Exception as e:
        st.warning(f"Error extracting social links from {website_url}: {e}")
    return social_links

def get_place_details(gmaps, place_id):
    try:
        place_details = gmaps.place(place_id, fields=PLACE_DETAIL_FIELDS)
        if place_details.get('status') == 'OK':
            st.write(f"    Fetched details for '{place_details.get('result', {}).get('name', 'N/A')}'")
            return place_details.get('result')
        st.error(f"Google Maps API error for Place ID {place_id}: {place_details.get('status')}")
    except Exception as e:
        st.error(f"Error fetching details for Place ID {place_id}: {e}")
    return None

def populate_emails_and_social_links(place_details):
    website = place_details.get('website')
    if website:
        if not website.startswith(('http://', 'https://')):
            website = 'http://' + website
        place_details['emails'] = extract_emails_from_website(website)
        place_details['social_links'] = extract_social_links_from_website(website)
    else:
        place_details['emails'] = []
        place_details['social_links'] = {"facebook": None, "instagram": None, "twitter": None}

def extract_data(gmaps, queries, max_results_per_query):
    extracted_data = []
    total_places_processed = 0
    st.info(f"Extracting data for {len(queries)} queries...")
    for query_index, query in enumerate(queries):
        if not query:
            continue
        st.markdown(f"--- \n**Query {query_index+1}/{len(queries)}: '{query}'**")
        place_ids = find_place_ids_by_text_search(gmaps, query, max_results_per_query)
        if not place_ids:
            st.write(f"No places found for '{query}'.")
            continue
        for place_index, place_id in enumerate(place_ids):
            st.write(f"  Place {place_index+1}/{len(place_ids)} (ID: {place_id})")
            place_details = get_place_details(gmaps, place_id)
            if place_details:
                populate_emails_and_social_links(place_details)
                extracted_data.append(place_details)
                total_places_processed += 1
            time.sleep(0.1)
    st.success(f"Processed {total_places_processed} place(s).")
    return extracted_data

def create_dataframe(data):
    rows = []
    for row_data in data:
        lat = row_data.get('geometry', {}).get('location', {}).get('lat', 'N/A')
        lng = row_data.get('geometry', {}).get('location', {}).get('lng', 'N/A')
        social_links = row_data.get('social_links', {})
        emails = row_data.get('emails', [])
        opening_hours_text = row_data.get('opening_hours', {}).get('weekday_text', ['N/A'])
        rows.append({
            'Place ID': row_data.get('place_id', 'N/A'),
            'Name': row_data.get('name', 'N/A'),
            'Address': row_data.get('formatted_address', 'N/A'),
            'Latitude': lat,
            'Longitude': lng,
            'Rating': row_data.get('rating', 'N/A'),
            'Total Ratings': row_data.get('user_ratings_total', 'N/A'),
            'Business Status': row_data.get('business_status', 'N/A'),
            'Types': 'N/A',
            'Opening Hours': '; '.join(opening_hours_text),
            'Website': row_data.get('website', 'N/A'),
            'Phone': row_data.get('formatted_phone_number', 'N/A'),
            'Emails': ', '.join(emails) if emails else 'N/A',
            'Facebook': social_links.get('facebook', 'N/A'),
            'Instagram': social_links.get('instagram', 'N/A'),
            'Twitter/X': social_links.get('twitter', 'N/A')
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# --- Login Page ---
def login_page():
    st.title("📄 Kalika Business Assistant - Login")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Login to Access the System")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.name = get_user_info(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# --- Invoice Query Page ---
def invoice_query_page():
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
            st.session_state.current_chat_id = "chat_1"
            st.session_state.chat_counter = 1
            st.session_state.chat_sessions["chat_1"] = {
                'query_history': [], 'response_history': [], 'follow_up_questions': []
            }
        if st.button("New Chat"):
            st.session_state.chat_counter += 1
            new_chat_id = f"chat_{st.session_state.chat_counter}"
            st.session_state.chat_sessions[new_chat_id] = {
                'query_history': [], 'response_history': [], 'follow_up_questions': []
            }
            st.session_state.current_chat_id = new_chat_id
            st.rerun()
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()

    st.title("📄 Proforma Invoice Query Assistant")
    st.markdown("Ask questions about proforma invoices processed from email attachments.")

    s3_client = get_s3_client()
    embeddings = get_embeddings_model()
    gemini_model = get_gemini_model()
    s3_status = "✅ S3 Client Initialized" if s3_client else "❌ S3 Client Failed"
    embeddings_status = "✅ Embeddings Model Loaded" if embeddings else "❌ Embeddings Model Failed"
    gemini_status = "✅ Gemini LLM Initialized" if gemini_model else "❌ Gemini LLM Failed"

    with st.status("Initializing resources...", expanded=False) as status:
        st.write(s3_status)
        st.write(embeddings_status)
        st.write(gemini_status)
        if not all([s3_client, embeddings, gemini_model]):
            st.error("Core components failed to initialize.")
            status.update(label="Initialization Failed!", state="error")
            st.stop()
        st.write("Loading Knowledge Base Index...")
        vector_store = download_and_load_faiss_index(s3_client, embeddings, S3_BUCKET, S3_PROFORMA_INDEX_PATH)
        if vector_store:
            st.write("✅ Knowledge Base Index Loaded")
            status.update(label="Initialization Complete!", state="complete", expanded=False)
        else:
            st.error("Failed to load knowledge base index.")
            status.update(label="Initialization Failed!", state="error")
            st.stop()

    st.markdown("---")
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat_id]
    for i in range(len(current_chat.get('query_history', []))):
        st.markdown(f"**Question:**\n> {current_chat['query_history'][i]}")
        st.markdown(f"**Answer:**\n{current_chat['response_history'][i]}")
        st.markdown("---")

    query_text = st.text_input(
        "Enter your query:",
        placeholder="e.g., What is the total amount for invoice [filename]?"
    )
    k_results = 15
    use_mmr_search = False

    if query_text and vector_store:
        with st.spinner(f"Searching knowledge base (k={k_results}, MMR={use_mmr_search})..."):
            retrieved_docs = query_faiss_index(vector_store, query_text, k=k_results, use_mmr=use_mmr_search)
        with st.spinner("Synthesizing answer..."):
            response = generate_llm_response(gemini_model, query_text, retrieved_docs)
        current_chat['query_history'] = current_chat.get('query_history', []) + [query_text]
        current_chat['response_history'] = current_chat.get('response_history', []) + [response]
        with st.spinner("Generating follow-up questions..."):
            follow_up_questions = generate_follow_up_questions(gemini_model, query_text, response, retrieved_docs)
            current_chat['follow_up_questions'] = follow_up_questions

        st.markdown("### Response:")
        st.markdown(response)
        if follow_up_questions:
            st.markdown("### You might want to ask:")
            cols = st.columns(len(follow_up_questions))
            for i, question in enumerate(follow_up_questions):
                if cols[i].button(question, key=f"follow_up_{i}_{st.session_state.current_chat_id}"):
                    st.session_state.follow_up_clicked = question
                    st.rerun()
        st.markdown("---")

# --- Place Data Extractor Page ---
def place_data_page():
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.name}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()
        st.header("Controls")
        default_queries = "Restaurant near Eiffel Tower\nCoffee shop near Times Square NYC\nHardware store, Pune, India"
        queries_input = st.text_area("Search Queries (one per line):", value=default_queries, height=150)
        max_results_per_query = st.slider("Max Results per Query", 1, 10, 3)
        extract_button = st.button("🚀 Extract Data", type="primary")
        st.markdown("---")
        st.info("Email/social link scraping depends on website structure.")

    st.title("🗺️ Google Maps Place Data Extractor")
    st.markdown("Search for places, fetch details, and scrape website data.")

    gmaps = get_gmaps_client()
    status = "✅ Google Maps Client Initialized" if gmaps else "❌ Google Maps Client Failed"
    with st.status("Initializing...", expanded=False) as status_container:
        st.write(status)
        if not gmaps:
            status_container.update(label="Initialization Failed!", state="error")
            st.stop()
        status_container.update(label="Initialization Complete!", state="complete", expanded=False)

    results_placeholder = st.container()
    if extract_button:
        queries = [q.strip() for q in queries_input.splitlines() if q.strip()]
        if not queries:
            st.warning("Enter at least one search query.")
            return
        results_placeholder.empty()
        with results_placeholder:
            with st.spinner("Processing..."):
                with st.status("Extracting Data...", expanded=True) as status:
                    st.write("Initializing...")
                    extracted_data = extract_data(gmaps, queries, max_results_per_query)
                    if extracted_data:
                        status.update(label="Creating DataFrame...")
                        df = create_dataframe(extracted_data)
                        if not df.empty:
                            status.update(label="Extraction Complete!", state="complete", expanded=False)
                            st.subheader("📊 Extracted Data")
                            st.dataframe(df)
                            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="💾 Download data as CSV",
                                data=csv_data,
                                file_name='Maps_extracted_data.csv',
                                mime='text/csv',
                                key='download-csv'
                            )
                        else:
                            status.update(label="Failed to create DataFrame.", state="warning")
                    else:
                        status.update(label="No data retrieved.", state="warning")

# --- Main Entry Point ---
def main():
    st.set_page_config(layout="wide")
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
    if not st.session_state.authenticated:
        login_page()
    else:
        mode = st.sidebar.selectbox("Select Mode", ["Proforma Invoice Query", "Place Data Extractor"])
        if mode == "Proforma Invoice Query":
            invoice_query_page()
        else:
            place_data_page()

if __name__ == "__main__":
    main()