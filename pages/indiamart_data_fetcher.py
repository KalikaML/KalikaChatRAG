import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime
import time # For potential delays

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache for 1 hour to avoid refetching category details too often
def fetch_page(url, headers):
    """Fetches content from a URL."""
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return None

def parse_headers(header_string):
    """Parses a multiline header string into a dictionary."""
    headers = {}
    if not header_string:
        return headers
    lines = header_string.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            headers[key.strip()] = value.strip()
    return headers

def connect_db(host, database, user, password):
    """Connects to the MySQL database."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        st.success("Successfully connected to MySQL database.")
        return connection
    except Error as e:
        st.error(f"Error while connecting to MySQL: {e}")
        return None

def check_duplicate(cursor, table_name, supplier_url):
    """Checks if a supplier URL already exists in the table."""
    try:
        query = f"SELECT id FROM {table_name} WHERE supplier_url = %s"
        cursor.execute(query, (supplier_url,))
        result = cursor.fetchone()
        return result is not None
    except Error as e:
        st.warning(f"Could not check for duplicate URL {supplier_url}: {e}")
        return False # Assume not duplicate if check fails

def insert_data_db(connection, cursor, table_name, data_dict):
    """Inserts a dictionary of data into the specified table."""
    if check_duplicate(cursor, table_name, data_dict.get('supplier_url', '')):
        st.write(f"Skipping duplicate: {data_dict.get('company_name')} ({data_dict.get('supplier_url')})")
        return False

    # Prepare columns and placeholders dynamically
    columns = list(data_dict.keys())
    placeholders = ', '.join(['%s'] * len(columns))
    sql_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    values = [str(data_dict.get(col, '')) for col in columns] # Ensure all values are strings

    try:
        cursor.execute(sql_query, values)
        connection.commit()
        return True
    except Error as e:
        st.error(f"Database Insertion Error for {data_dict.get('company_name')}: {e}")
        st.error(f"Query attempt: {cursor.statement}") # Show the query that failed
        # st.error(f"Values: {values}") # Uncomment carefully - might contain sensitive data
        return False

# --- Main Scraping Logic ---

def scrape_indiamart(start_url, headers, db_connection=None, db_cursor=None, db_table_name=None, save_to_db=False):
    """Scrapes Indiamart based on the starting category URL."""
    BASE_URL = "https://dir.indiamart.com"
    scraped_data = []
    s_count = 0
    total_sub_cats = 0
    processed_sub_cats = 0

    st.info(f"Fetching main category page: {start_url}")
    response = fetch_page(start_url, headers)
    if not response:
        return pd.DataFrame(scraped_data) # Return empty DataFrame on error

    soup = BeautifulSoup(response.text, features='lxml')

    # --- 1. Get Industry Name ---
    industry_h1 = soup.find('div', {"class": "mid"})
    industry = industry_h1.find('h1').text if industry_h1 and industry_h1.find('h1') else "Unknown Industry"
    st.write(f"**Industry:** {industry}")

    # --- 2. Get Main Categories on the Page ---
    category_list = soup.findAll('li', {"class": "q_cb"})
    if not category_list:
        st.warning("No main categories (q_cb list items) found on the page.")
        # Try finding sub-categories directly if top-level isn't structured as expected
        category_list = soup.findAll('li', {"class": "box"}) # Assuming 'box' might contain sub-cats
        if not category_list:
             st.error("Could not find any category links to process. Check the starting URL or page structure.")
             return pd.DataFrame(scraped_data)

    st.write(f"Found {len(category_list)} main category sections.")
    progress_bar_main = st.progress(0)

    for cat_idx, category_section in enumerate(category_list): # e.g., 'Apparel Fabrics', 'Clothing Accessories' sections

        # --- 3. Get Sub-Categories within each Section ---
        # Handle cases where the link might be directly on the 'li' or within an 'a' tag
        cat_name_tag = category_section.find('a')
        cat_name_main = cat_name_tag.text.strip() if cat_name_tag else "Unknown Category Section"

        sub_cat_links = category_section.findAll('a', {"class": "slink"}) # Links like 'Denim Fabric', 'Cotton Fabric'

        if not sub_cat_links:
             st.write(f"No 'slink' sub-category links found under '{cat_name_main}'. Skipping this section.")
             continue

        st.write(f"Processing Section: **{cat_name_main}** ({len(sub_cat_links)} sub-categories)")
        total_sub_cats += len(sub_cat_links)
        status_placeholder = st.empty()
        sub_progress_bar = st.progress(0)

        for sub_cat_idx, sub_cat_link_tag in enumerate(sub_cat_links):
            sub_cat_name = sub_cat_link_tag.text.strip()
            sub_cat_href = sub_cat_link_tag['href']
            sub_cat_full_url = BASE_URL + sub_cat_href if not sub_cat_href.startswith('http') else sub_cat_href

            status_placeholder.info(f"Processing Sub-Category: **{sub_cat_name}** ({sub_cat_idx + 1}/{len(sub_cat_links)})")

            response_sub = fetch_page(sub_cat_full_url, headers)
            if not response_sub:
                st.warning(f"Skipping sub-category {sub_cat_name} due to fetch error.")
                processed_sub_cats += 1
                continue

            soup_sub = BeautifulSoup(response_sub.text, features='lxml')

            # --- 4. Extract MCAT ID ---
            mcatid_ul = soup_sub.find('ul', {"class": "wlm"})
            if not mcatid_ul or 'data-click' not in mcatid_ul.attrs:
                st.warning(f"Could not find MCAT ID for sub-category: {sub_cat_name}. Skipping.")
                processed_sub_cats += 1
                continue

            try:
                mcatid = mcatid_ul['data-click'].split('|')[1]
            except (IndexError, AttributeError):
                 st.warning(f"Could not parse MCAT ID from data-click attribute for: {sub_cat_name}. Skipping.")
                 processed_sub_cats += 1
                 continue

            st.write(f"  Found MCAT ID: {mcatid} for {sub_cat_name}")

            # --- 5. Paginate through Suppliers ---
            page_num = 0
            while True:
                page_start = page_num * 28 + 1
                page_end = page_start + 27
                fetch_start = page_num * 28

                params = [
                    ['mcatId', mcatid],
                    # ['glid', '104881740'], # Glid might be specific, maybe remove or make configurable?
                    ['prod_serv', 'P'],
                    ['mcatName', sub_cat_name], # Use the actual name
                    ['srt', page_start],
                    ['end', page_end],
                    ['ims_flag', ''],
                    ['cityID', ''],
                    ['prc_cnt_flg', '1'],
                    ['fcilp', '0'],
                    ['spec', ''],
                    ['pr', '0'],
                    ['pg', page_num + 1],
                    ['frsc', fetch_start],
                    ['video', ''],
                ]

                next_url = f"{BASE_URL}/impcat/next"
                st.write(f"    Fetching supplier page {page_num + 1} for {sub_cat_name}...")

                try:
                    # Use POST if GET doesn't work, sometimes APIs change
                    # response_suppliers = requests.get(next_url, headers=headers, params=params, timeout=30)
                    response_suppliers = requests.post(next_url, headers=headers, data=params, timeout=30) # Try POST
                    response_suppliers.raise_for_status()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching supplier page {page_num + 1} for {sub_cat_name}: {e}")
                    break # Stop pagination for this sub-category on error

                try:
                    json_data = json.loads(response_suppliers.text)
                    if 'content' not in json_data or not json_data['content']:
                         st.write(f"    No more content found for {sub_cat_name} on page {page_num + 1}. Moving to next sub-category.")
                         break # No more suppliers
                    soup_suppliers = BeautifulSoup(json_data['content'], features='lxml')
                except json.JSONDecodeError:
                     st.error(f"    Failed to decode JSON response for supplier page {page_num + 1}. Content: {response_suppliers.text[:500]}...")
                     break
                except Exception as e:
                    st.error(f"    Error processing supplier page {page_num+1} content: {e}")
                    break


                # --- 6. Extract Supplier Links ---
                suppliers_link_tags = soup_suppliers.findAll("a", {"class": "fs18 ptitle"}) # Check class name

                if not suppliers_link_tags:
                    st.write(f"    No supplier links found on page {page_num + 1} for {sub_cat_name} (or end reached).")
                    if page_num == 0: # Check if the *first* page had no links
                         st.warning(f"    Warning: No suppliers found for the entire sub-category: {sub_cat_name}")
                    break # No more suppliers found on this page

                st.write(f"    Found {len(suppliers_link_tags)} potential suppliers on page {page_num + 1}.")

                # --- 7. Scrape Individual Supplier Pages ---
                for supplier_link_tag in suppliers_link_tags:
                    supplier_page_url = supplier_link_tag.get('href')
                    if not supplier_page_url:
                        st.warning("Found a supplier link tag with no href.")
                        continue

                    # Ensure URL is absolute
                    if supplier_page_url.startswith('//'):
                        supplier_page_url = 'https:' + supplier_page_url
                    elif not supplier_page_url.startswith('http'):
                         st.warning(f"Skipping relative (?) supplier URL: {supplier_page_url}")
                         continue


                    # Introduce a small delay to be polite to the server
                    time.sleep(0.5) # Sleep for 500ms between supplier requests

                    response_supplier = fetch_page(supplier_page_url, headers)
                    if not response_supplier:
                        st.warning(f"Could not fetch supplier page: {supplier_page_url}. Skipping.")
                        continue

                    soup_supplier = BeautifulSoup(response_supplier.text, features='lxml')

                    # --- 8. Extract Supplier Details ---
                    company_name = ''
                    owner_name = ''
                    address = ''
                    website = ''
                    phone = ''
                    about = {}
                    company_desc_list = []
                    product_details = {}
                    product_desc = ''
                    product_info = {}

                    # Contact Details Box
                    seller_contact_details = soup_supplier.find('div',{"class": "fs13 color1 pml10"}) # Check class
                    if seller_contact_details:
                        # Company Name (Robust finding)
                        comp_name_tag = seller_contact_details.find('a', {"class": "pcmN bo"}) or \
                                        seller_contact_details.find('a', {"class": "cpN bo"}) # Variations?
                        if comp_name_tag:
                            company_name = comp_name_tag.string.strip() if comp_name_tag.string else comp_name_tag.get_text(strip=True)


                        # Owner Name
                        owner_tag = seller_contact_details.find('div', {"id":"supp_nm"})
                        if owner_tag:
                            owner_name = owner_tag.string.strip() if owner_tag.string else owner_tag.get_text(strip=True)


                        # Address
                        address_tag = seller_contact_details.find('span', {"class": "color1 dcell verT fs13"}) # Check class
                        if address_tag:
                           address = address_tag.get_text(separator=" ", strip=True)


                        # Website
                        website_tag = seller_contact_details.find('div', {"class": "mt5"}) # Contains website link
                        if website_tag:
                           link_tag = website_tag.find('a', {"class": "color1 utd"}) # Check class
                           if link_tag and link_tag.string:
                               website = link_tag.string.strip()
                           elif link_tag:
                                website = link_tag.get('href') # Fallback to href if no text


                        # Phone (handle potential variations)
                        phone_span = seller_contact_details.find('span', {"class": ["duet", "tel"]}) # Allow multiple classes
                        if phone_span:
                            phone = phone_span.get_text(strip=True)


                    # About Section (Key-Value Pairs)
                    about_section = soup_supplier.find('div', {"id": "compDetail"}) # Look for a container
                    if about_section:
                         about_items = about_section.find_all('div', {"class": "lh21 pdinb wid3 mb20 verT"}) # Check class
                         for item in about_items:
                            spans = item.find_all('span')
                            if len(spans) >= 2:
                                key = spans[0].string.strip() if spans[0].string else "Key"
                                value = spans[1].string.strip() if spans[1].string else spans[1].get_text(strip=True)
                                about[key] = value

                    # Company Description (About Us Text)
                    about_us_div = soup_supplier.find('div', {"id": "aboutUs"})
                    if about_us_div:
                        paragraphs = about_us_div.find_all('p')
                        company_desc_list = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

                    # Product Specification Table
                    prod_spec_div = soup_supplier.find('div', {"id" : "pdpD"}) # Product Detail Page Details?
                    if prod_spec_div:
                        prod_spec_table = prod_spec_div.find('div', {"class": "dtlsec1"}) # Check class
                        if prod_spec_table:
                           rows = prod_spec_table.find_all('tr')
                           for row in rows:
                                cells = row.find_all('td')
                                if len(cells) == 2:
                                    key = cells[0].get_text(strip=True).replace(':', '')
                                    value = cells[1].get_text(strip=True)
                                    product_details[key] = value


                    # Product Description & Info
                    desc_div = soup_supplier.find('div', {"class": "pdest1"}) # Check class
                    if desc_div:
                        # Get main description text (excluding potential tables inside)
                        product_desc = desc_div.find(text=True, recursive=False).strip() # Try to get only immediate text
                        if not product_desc: # Fallback
                             product_desc = desc_div.get_text(separator=' ', strip=True) # Less precise

                        # Product Info Table (often nested)
                        info_table = desc_div.find('div', {"class": "dtlsec1"}) # Check class
                        if info_table:
                            rows = info_table.find_all('tr')
                            for row in rows:
                                cells = row.find_all('td')
                                if len(cells) == 2:
                                    key = cells[0].get_text(strip=True).replace(':', '')
                                    value = cells[1].get_text(strip=True)
                                    product_info[key] = value


                    # --- 9. Store Extracted Data ---
                    supplier_data = {
                        'industry': industry,
                        'sub_cat': sub_cat_name,
                        'supplier_url': supplier_page_url,
                        'company_name': company_name,
                        'owner': owner_name,
                        'address': address,
                        'website': website,
                        'phone': phone,
                        'about': json.dumps(about), # Store JSON string
                        'product_detail': json.dumps(product_details), # Store JSON string
                        'product_desc': "\n".join(company_desc_list), # Combine description paragraphs
                        'product_info': json.dumps(product_info) # Store JSON string
                        # Add 'data-item' if you can find it on the page for unique ID
                    }
                    scraped_data.append(supplier_data)
                    s_count += 1
                    st.write(f"    -> Scraped: {company_name} ({s_count} total)")

                    # --- 10. Insert into DB if enabled ---
                    if save_to_db and db_connection and db_cursor and db_table_name:
                         insert_data_db(db_connection, db_cursor, db_table_name, supplier_data)


                # End of supplier loop for this page
                page_num += 1
                # Add a slightly longer delay between pages
                time.sleep(1)

            # End of pagination loop (while True) for this sub-category
            processed_sub_cats += 1
            if total_sub_cats > 0:
                 sub_progress_bar.progress(processed_sub_cats / total_sub_cats)

        # End loop for sub-categories in this section
        status_placeholder.empty() # Clear status message
        sub_progress_bar.empty()
        progress_bar_main.progress((cat_idx + 1) / len(category_list))

    # --- 11. Return Data ---
    st.success(f"Scraping finished for industry: {industry}")
    return pd.DataFrame(scraped_data)

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("📦 Indiamart Supplier Scraper")

st.markdown("""
**Disclaimer:** This tool is for educational purposes only. Web scraping can be against the terms of service of websites.
Ensure you have permission and scrape responsibly. Excessive requests can overload servers.
Indiamart's structure might change, breaking this script.
""")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Configuration")

start_url = st.sidebar.text_input(
    "Indiamart Category URL",
    "https://dir.indiamart.com/industry/apparel-garments.html", # Default value
    help="Enter the URL of the main Indiamart category page you want to scrape (e.g., .../industry/apparel-garments.html)"
)

st.sidebar.subheader("Request Headers")
st.sidebar.info("""
**IMPORTANT:** You MUST provide headers, especially a valid 'Cookie' from a logged-in Indiamart session.
1. Log in to Indiamart in your browser.
2. Go to a supplier listing page.
3. Open Developer Tools (F12), go to the 'Network' tab.
4. Find a request to `dir.indiamart.com` (e.g., refresh the page or load more results).
5. Right-click the request -> Copy -> Copy as cURL.
6. Paste the cURL command into [curlconverter.com](https://curlconverter.com/) or similar tool.
7. Copy the generated 'headers' dictionary content (Python format) or paste the raw headers below.
""")
headers_str = st.sidebar.text_area(
    "Paste Headers Here (key: value format, one per line)",
    """
authority: dir.indiamart.com
accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
accept-language: en-US,en;q=0.9
cache-control: no-cache
pragma: no-cache
sec-ch-ua: "Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
sec-fetch-dest: document
sec-fetch-mode: navigate
sec-fetch-site: none
sec-fetch-user: ?1
upgrade-insecure-requests: 1
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36
cookie: YOUR_LOGIN_COOKIE_HERE=...; other_cookies=...
    """,
    height=300,
    help="Ensure the 'cookie' line contains your actual login cookie."
)

st.sidebar.subheader("Output Options")
save_option = st.sidebar.radio(
    "How to save the data?",
    ('Save to CSV File Only', 'Save to MySQL Database Only', 'Save to Both CSV and MySQL'),
    index=0
)

# --- MySQL Database Inputs (Conditional) ---
db_conn = None
db_cursor = None
db_table = "apparel" # Default table name from original script

save_db = "MySQL" in save_option
if save_db:
    st.sidebar.subheader("MySQL Database Credentials")
    db_host = st.sidebar.text_input("DB Host", "localhost")
    db_user = st.sidebar.text_input("DB User", "root")
    db_pass = st.sidebar.text_input("DB Password", type="password")
    db_name = st.sidebar.text_input("DB Database Name", "indiamart")
    db_table = st.sidebar.text_input("DB Table Name", "apparel", help="Ensure this table exists with the correct schema.")
    # Check schema button?
    # st.sidebar.code("""
    # CREATE TABLE IF NOT EXISTS apparel (
    #     id int NOT NULL AUTO_INCREMENT,
    #     industry VARCHAR(100),
    #     sub_cat VARCHAR(255),
    #     supplier_url VARCHAR(500) UNIQUE, # Added UNIQUE constraint
    #     company_name VARCHAR(255),        # Increased length
    #     owner VARCHAR(255),              # Increased length
    #     address VARCHAR(500),            # Increased length
    #     website VARCHAR(255),            # Increased length
    #     phone VARCHAR(50),               # Increased length
    #     about TEXT,
    #     product_detail TEXT,
    #     product_desc TEXT,
    #     product_info TEXT,
    #     PRIMARY KEY (id)
    # );
    # """, language="sql")


# --- Main Area ---
if st.button("🚀 Start Scraping", type="primary"):
    headers_dict = parse_headers(headers_str)

    # --- Input Validation ---
    valid_input = True
    if not start_url:
        st.error("Please enter the Indiamart Category URL.")
        valid_input = False
    if not headers_dict or 'cookie' not in headers_dict or not headers_dict['cookie']:
        st.error("Headers are required, especially the 'cookie'. Please provide valid headers.")
        valid_input = False
    if 'user-agent' not in headers_dict:
         st.warning("Consider adding a 'User-Agent' to your headers.")
         # Optionally add a default one if missing, but user-provided is better
         # headers_dict['user-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...'


    if save_db:
        if not all([db_host, db_user, db_name]): # Password can be empty for some setups
            st.error("Please fill in all required MySQL database credentials.")
            valid_input = False
        else:
            # Try connecting before starting the long scrape
            db_conn = connect_db(db_host, db_name, db_user, db_pass)
            if db_conn:
                db_cursor = db_conn.cursor()
            else:
                valid_input = False # Don't proceed if DB connection failed

    if valid_input:
        start_time = datetime.now()
        st.info(f"Scraping started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")
        results_placeholder = st.empty()
        results_placeholder.info("Scraping in progress... Please wait.")
        download_placeholder = st.empty()

        with st.spinner('Scraping Indiamart... This might take a while!'):
            try:
                df_results = scrape_indiamart(
                    start_url,
                    headers_dict,
                    db_connection=db_conn,
                    db_cursor=db_cursor,
                    db_table_name=db_table,
                    save_to_db=save_db
                )

                end_time = datetime.now()
                duration = end_time - start_time
                st.success(f"Scraping completed!")
                st.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.info(f"Duration: {duration}")
                st.info(f"Total Suppliers Found: {len(df_results)}")

                results_placeholder.dataframe(df_results)

                save_csv = "CSV" in save_option
                if not df_results.empty and save_csv:
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    download_placeholder.download_button(
                        label="⬇️ Download Data as CSV",
                        data=csv_data,
                        file_name=f"indiamart_suppliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                    )
                elif df_results.empty:
                     st.warning("No data was scraped.")

            except Exception as e:
                st.error(f"An unexpected error occurred during scraping: {e}")
                import traceback
                st.error("Traceback:")
                st.code(traceback.format_exc())

            finally:
                # --- Close DB Connection ---
                if db_cursor:
                    db_cursor.close()
                    st.write("Database cursor closed.")
                if db_conn and db_conn.is_connected():
                    db_conn.close()
                    st.write("Database connection closed.")

    else:
        st.warning("Please correct the inputs before starting.")

st.markdown("---")
st.markdown("App finished loading.")