import streamlit as st
import googlemaps
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import toml
import time # Import time for potential delays

# --- Configuration and Secrets ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml"
API_KEY = None # Initialize API_KEY

st.set_page_config(layout="wide") # Use more screen space

try:
    # More standard key name for Google Maps
    secrets = toml.load(SECRETS_FILE_PATH)
    API_KEY = secrets.get("Maps_API_KEY")
    if not API_KEY:
        st.error(f"Error: 'Maps_API_KEY' not found or is empty in {SECRETS_FILE_PATH}")
        st.stop() # Halt execution if key is missing or empty in the file
except FileNotFoundError:
    st.error(f"Error: Secrets file not found at {SECRETS_FILE_PATH}")
    st.info("Please create a `.streamlit` folder in your project directory and add a `secrets.toml` file with your API key like this:\n\n```toml\nMaps_API_KEY = \"YOUR_API_KEY_HERE\"\n```")
    st.stop() # Halt execution
except toml.TomlDecodeError:
    st.error(f"Error: Could not decode TOML file at {SECRETS_FILE_PATH}. Check for syntax errors.")
    st.stop() # Halt execution
except Exception as e:
    st.error(f"An unexpected error occurred loading secrets: {e}")
    st.stop() # Halt execution

# Initialize Google Maps client only if API_KEY is confirmed available
gmaps = googlemaps.Client(key=API_KEY)
st.sidebar.success("Google Maps API Client Initialized.") # Indicate success

# --- Constants ---
# Define the specific fields you want from the Places API Details request
# *** THIS IS THE CORRECTED LIST - BASED ON THE ERROR MESSAGE ***
PLACE_DETAIL_FIELDS = [
    'place_id', 'name', 'formatted_address', 'geometry/location', 'rating',
    'user_ratings_total', 'website', 'formatted_phone_number', 'business_status',
    'opening_hours' # Request the parent object, REMOVED 'types' and 'opening_hours/weekday_text'
]

# --- Helper Functions ---

def find_place_ids_by_text_search(query, max_results=5):
    """
    Finds multiple place IDs using a text query.
    Returns a list of place IDs. Handles API errors more explicitly.
    """
    place_ids = []
    try:
        # Using st.status for progress updates within functions if desired
        # status_update = st.empty()
        # status_update.write(f"Searching for places matching: '{query}'...") # Show progress
        st.write(f"Searching for places matching: '{query}'...") # Simpler write for now
        results = gmaps.places(query=query) # Text Search request

        if results:
            status = results.get('status')
            if status == 'OK':
                place_ids = [result['place_id'] for result in results.get('results', [])[:max_results]]
                st.write(f"  Found {len(place_ids)} potential place(s).")
            elif status == 'ZERO_RESULTS':
                st.warning(f"No results found for query: '{query}'")
            else:
                st.error(f"Google Maps API error for query '{query}': {status}. Message: {results.get('error_message', 'N/A')}")
        else:
            st.error(f"Received empty response from Google Maps API for query: '{query}'")

    except googlemaps.exceptions.ApiError as e:
        st.error(f"Google Maps API Error during text search for '{query}': {e}")
    except googlemaps.exceptions.HTTPError as e:
        st.error(f"HTTP Error during text search for '{query}': {e}")
    except googlemaps.exceptions.Timeout as e:
        st.error(f"Timeout during text search for '{query}': {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during text search for '{query}': {e}")

    return place_ids

def extract_emails_from_website(website_url):
    """
    Extracts email addresses from a website. Includes timeout and User-Agent.
    """
    emails = set() # Use a set to avoid duplicates
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            # st.info(f"Skipping email extraction for {website_url}: Content type is not HTML ({content_type})") # Can be verbose
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}"
        found_in_text = re.findall(email_pattern, soup.get_text())
        emails.update(found_in_text)

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.lower().startswith('mailto:'):
                email_match = re.search(email_pattern, href)
                if email_match:
                    emails.add(email_match.group(0))

        # st.write(f"  Extracted {len(emails)} emails from {website_url}") # Can be verbose

    except requests.exceptions.Timeout:
        st.warning(f"Timeout fetching website {website_url} for email extraction.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch website {website_url} for email extraction: {e}")
    except Exception as e:
        st.warning(f"Error extracting emails from {website_url}: {e}")

    return list(emails)

def extract_social_links_from_website(website_url):
    """
    Extracts social media links (Facebook, Instagram, Twitter/X) from a website.
    Includes timeout and User-Agent. Looks for common patterns.
    """
    social_links = {"facebook": None, "instagram": None, "twitter": None}
    found_count = 0
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
             # st.info(f"Skipping social link extraction for {website_url}: Content type is not HTML ({content_type})")
             return social_links

        soup = BeautifulSoup(response.content, 'html.parser')

        for a_tag in soup.find_all('a', href=True):
            href_original = a_tag['href'] # Keep original case for storage
            href_lower = href_original.lower()

            if social_links["facebook"] is None and ("facebook.com/" in href_lower) and ("sharer" not in href_lower) and ("plugins" not in href_lower):
                social_links["facebook"] = href_original
                found_count += 1
            elif social_links["instagram"] is None and ("instagram.com/" in href_lower) and ("p/" not in href_lower):
                social_links["instagram"] = href_original
                found_count += 1
            elif social_links["twitter"] is None and (("twitter.com/" in href_lower) or ("x.com/" in href_lower)) and ("intent" not in href_lower) and ("status" not in href_lower):
                 social_links["twitter"] = href_original
                 found_count += 1

            if all(social_links.values()): break # Stop early if all found

        # st.write(f"  Found {found_count} social links on {website_url}") # Can be verbose

    except requests.exceptions.Timeout:
        st.warning(f"Timeout fetching website {website_url} for social link extraction.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch website {website_url} for social link extraction: {e}")
    except Exception as e:
        st.warning(f"Error extracting social links from {website_url}: {e}")

    return social_links

def get_place_details(place_id):
    """
    Retrieves details for a specific place using its place ID and predefined fields.
    Handles API errors more explicitly. Includes DEBUG print.
    """
    place_details_result = None
    try:
        st.write(f"  Fetching details for Place ID: {place_id}...")

        # --- <<< TEMPORARY DEBUGGING LINE >>> ---
        # This will print the fields list to your terminal AND the sidebar
        print(f"DEBUG: Requesting fields for Place ID {place_id}: {PLACE_DETAIL_FIELDS}")
        st.sidebar.caption(f"Requesting fields for {place_id}:")
        st.sidebar.json(PLACE_DETAIL_FIELDS) # Display list in sidebar for easy checking
        # --- <<< END DEBUGGING LINE >>> ---

        # Use the CORRECTED predefined fields list
        place_details = gmaps.place(place_id, fields=PLACE_DETAIL_FIELDS)

        if place_details:
            status = place_details.get('status')
            if status == 'OK':
                place_details_result = place_details.get('result')
                st.write(f"    Successfully fetched details for '{place_details_result.get('name', 'N/A')}'")
            else:
                # Log non-OK statuses
                st.error(f"Google Maps API error fetching details for Place ID {place_id}: {status}. Message: {place_details.get('error_message', 'N/A')}")
        else:
            st.error(f"Received empty response from Google Maps API for details request (Place ID: {place_id})")

    # Catch potential exceptions during the API call
    except googlemaps.exceptions.ApiError as e:
        # This catches errors reported by the API (like invalid key, bad request format *if not caught by status check*)
        st.error(f"Google Maps API Error fetching details for Place ID {place_id}: {e}")
    except googlemaps.exceptions.HTTPError as e:
        st.error(f"HTTP Error fetching details for Place ID {place_id}: {e}")
    except googlemaps.exceptions.Timeout:
        st.error(f"Timeout fetching details for Place ID {place_id}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        st.error(f"An unexpected Python error occurred fetching details for Place ID {place_id}: {e}") # Clarified error source

    time.sleep(0.1) # Small delay
    return place_details_result

def populate_emails_and_social_links(place_details):
    """
    Populates email addresses and social media links by scraping the website if available.
    Modifies the place_details dictionary in-place.
    """
    website = place_details.get('website')
    emails = []
    social_links = {"facebook": None, "instagram": None, "twitter": None}

    if website:
        if not website.startswith(('http://', 'https://')):
            website = 'http://' + website
        # st.write(f"  Attempting to scrape website: {website}") # Can be verbose
        try:
            emails = extract_emails_from_website(website)
            social_links = extract_social_links_from_website(website)
            # st.write(f"    Scraping finished for {website}") # Can be verbose
        except Exception as e:
            st.warning(f"    Unexpected error during scraping process for {website}: {e}")
    # else:
        # st.write(f"  No website found for '{place_details.get('name', 'N/A')}', skipping scraping.") # Can be verbose

    place_details['emails'] = emails
    place_details['social_links'] = social_links

def extract_data(queries, max_results_per_query):
    """
    Extracts data for a list of queries. Retrieves multiple results per query.
    """
    extracted_data = []
    total_places_processed = 0
    st.info(f"Starting data extraction for {len(queries)} queries...")

    for query_index, query in enumerate(queries):
        if not query: continue
        st.markdown(f"--- \n**Processing Query {query_index+1}/{len(queries)}: '{query}'**")

        place_ids = find_place_ids_by_text_search(query, max_results=max_results_per_query)

        if not place_ids:
            st.write(f"No place IDs found for query '{query}', moving to next query.")
            continue

        st.write(f"Processing {len(place_ids)} place(s) found for query '{query}':")
        for place_index, place_id in enumerate(place_ids):
            st.write(f"  Processing Place {place_index+1}/{len(place_ids)} (ID: {place_id})")
            place_details = get_place_details(place_id) # Fetch details using the ID

            if place_details:
                populate_emails_and_social_links(place_details) # Scrape website if available
                extracted_data.append(place_details)
                total_places_processed += 1
            else:
                # Error messages are now inside get_place_details
                st.warning(f"Skipping Place ID {place_id} due to errors fetching details (see messages above).")
            # st.write("---") # Separator between places - removed for less clutter

        # st.write(f"Finished processing query: '{query}'") # Redundant with header

    st.success(f"Data extraction complete. Processed details for {total_places_processed} place(s).")
    return extracted_data

def create_dataframe(data):
    """
    Creates a Pandas DataFrame from the extracted data list (list of dictionaries).
    Handles missing data gracefully using the CORRECTED field access.
    """
    rows = []
    if not data:
        st.warning("No data was successfully extracted to create a DataFrame.")
        return pd.DataFrame()

    for row_data in data:
        lat = row_data.get('geometry', {}).get('location', {}).get('lat')
        lng = row_data.get('geometry', {}).get('location', {}).get('lng')
        social_links = row_data.get('social_links', {})
        emails = row_data.get('emails', [])
        # *** Access opening hours text safely from the 'opening_hours' object ***
        opening_hours_text = row_data.get('opening_hours', {}).get('weekday_text', ['N/A'])

        rows.append({
            'Place ID': row_data.get('place_id', 'N/A'),
            'Name': row_data.get('name', 'N/A'),
            'Address': row_data.get('formatted_address', 'N/A'),
            'Latitude': lat if lat is not None else 'N/A',
            'Longitude': lng if lng is not None else 'N/A',
            'Rating': row_data.get('rating', 'N/A'),
            'Total Ratings': row_data.get('user_ratings_total', 'N/A'),
            'Business Status': row_data.get('business_status', 'N/A'),
            'Types': 'N/A', # Set to N/A as 'types' field was removed from request
            'Opening Hours': '; '.join(opening_hours_text), # Use the safely extracted text
            'Website': row_data.get('website', 'N/A'),
            'Phone': row_data.get('formatted_phone_number', 'N/A'),
            'Emails': ', '.join(emails) if emails else 'N/A',
            'Facebook': social_links.get('facebook', 'N/A'),
            'Instagram': social_links.get('instagram', 'N/A'),
            'Twitter/X': social_links.get('twitter', 'N/A')
        })
    return pd.DataFrame(rows)

# --- Streamlit App Main Function ---
def main():
    st.title("🗺️ Google Maps Place Data Extractor")
    st.markdown("Enter search queries below. The app searches Google Maps, fetches details, and attempts to scrape website data.")

    st.sidebar.header("Controls")
    default_queries = "Restaurant near Eiffel Tower\nCoffee shop near Times Square NYC\nHardware store, Pune, India"
    queries_input = st.sidebar.text_area("Enter Search Queries (one per line):",
                                 value=default_queries, height=150)
    max_results_per_query = st.sidebar.slider("Max Results per Query", min_value=1, max_value=10, value=3,
                                              help="How many places to fetch details for per query.")
    extract_button = st.sidebar.button("🚀 Extract Data", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.info("Note: Email/social link scraping depends on website structure and may not always work.")
    st.sidebar.markdown("##### Debug Info:") # Section for debug output

    results_placeholder = st.container() # Use a container for results area

    if extract_button:
        queries = [q.strip() for q in queries_input.splitlines() if q.strip()]

        if not queries:
            st.warning("Please enter at least one search query in the sidebar.")
            return

        # Clear previous results if any
        results_placeholder.empty()

        with results_placeholder: # Write subsequent output into the container
            with st.spinner("Processing... Please wait."): # Use spinner for long operations
                # Use st.status for detailed progress tracking
                with st.status("Extracting Data...", expanded=True) as status:
                    st.write("Initializing...")
                    if gmaps is None:
                         status.update(label="Error: Google Maps client not available.", state="error", expanded=True)
                         return

                    extracted_data = extract_data(queries, max_results_per_query)

                    if extracted_data:
                        status.update(label="Creating DataFrame...", state="running")
                        df = create_dataframe(extracted_data)

                        if not df.empty:
                             status.update(label="Extraction Complete!", state="complete", expanded=False)
                             st.subheader("📊 Extracted Data")
                             st.dataframe(df) # Display DataFrame within the container

                             csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                             st.download_button(
                                label="💾 Download data as CSV",
                                data=csv_data,
                                file_name='Maps_extracted_data.csv',
                                mime='text/csv',
                                key='download-csv'
                             )
                        else:
                             status.update(label="Extraction finished, but failed to create DataFrame.", state="warning", expanded=True)
                             st.warning("Could not create a table. Check logs above for errors.")

                    else:
                        status.update(label="Extraction finished. No data retrieved.", state="warning", expanded=True)
                        st.warning("No data extracted. Check queries, API key/permissions (Places API enabled?), and billing status. See detailed logs above.")

if __name__ == "__main__":
    main()
