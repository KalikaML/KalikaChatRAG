import streamlit as st
import psycopg2
from psycopg2 import OperationalError
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---- CONFIG ----
MODEL_DIRECTORY = "bge-base-en-v1.5"
LOCAL_FAISS_INDEX_DIR = "C:/rag_with_api/faiss_index"
GEMINI_MODEL = "gemini-1.5-pro"
GEMINI_API_KEY = st.secrets["gemini_api_key"]

# ---- POSTGRESQL CONNECTION ----
def get_pg_conn():
    pg = st.secrets["postgres"]
    try:
        conn = psycopg2.connect(
            host=pg["host"],
            port=pg["port"],
            dbname=pg["dbname"],
            user=pg["user"],
            password=pg["password"],
            client_encoding=["UTF8"]
        )
        return conn
    except OperationalError as e:
        st.error(f"Database connection failed: {e}")
        print("Database connection failed:", e)
        return None

# ---- DATABASE CONNECTION TEST ----
def test_db_connection():
    st.info("Testing database connection...")
    conn = get_pg_conn()
    if conn:
        st.success("✅ Database connection successful!")
        print("✅ Database connection successful!")
        conn.close()
    else:
        st.error("❌ Database connection failed! Check credentials and server.")
        print("❌ Database connection failed! Check credentials and server.")

# ---- SAVE CHAT ----
def save_chat(username, user_query, assistant_response):
    try:
        conn = get_pg_conn()
        if not conn:
            st.warning("No database connection. Chat not saved.")
            print("No database connection. Chat not saved.")
            return
        cur = conn.cursor()
        print(f"Inserting: {username}, {user_query}, {assistant_response[:40]}")
        cur.execute(
            "INSERT INTO chat_history (username, user_query, assistant_response) VALUES (%s, %s, %s)",
            (username, user_query, assistant_response)
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Inserted!")
        st.info("Chat saved to database.")
    except Exception as e:
        st.warning(f"Could not save chat to database: {e}")
        print("DB Error:", e)

# ---- LOGIN FUNCTIONALITY ----
def login():
    users = st.secrets["users"]
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if not st.session_state["logged_in"]:
        st.markdown("""
            <style>
            .login-card {
                background: #232526;
                color: #fff;
                padding: 2.5rem 2.5rem 2rem 2.5rem;
                border-radius: 18px;
                box-shadow: 0 6px 32px #0008;
                min-width:340px; max-width:370px;
                margin:auto;
            }
            .login-title {
                text-align:center;
                color:#90caf9;
                margin-bottom:2rem;
                font-weight:700;
                font-size:1.6rem;
            }
            </style>
            <div style='display:flex; justify-content:center; align-items:center; height:70vh;'>
                <div class='login-card'>
                    <div class='login-title'>🔒 Login</div>
        """, unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_btn = st.form_submit_button("Login")
            if login_btn:
                if username in users and password == users[username]:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.stop()

# ---- Embeddings Model ----
@st.cache_resource
def get_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_DIRECTORY,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

# ---- Gemini LLM ----
@st.cache_resource
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

# ---- FAISS Index Loader ----
@st.cache_resource(ttl=3600)
def load_local_faiss_index(_embeddings, folder_path):
    return FAISS.load_local(
        folder_path=folder_path,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True
    )

# ---- Similarity Search with k=50 ----
def get_similar_docs(vector_store, query):
    return vector_store.similarity_search(query, k=50)

# ---- Response Generation ----
def generate_response(llm, query, docs):
    if not docs:
        return "No relevant documents found"
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    messages = [
        SystemMessage(content=f"Answer using this context:\n{context[:20000]}"),
        HumanMessage(content=query)
    ]
    return llm.invoke(messages).content

# ---- Follow-up Question Generation ----
def generate_followup_questions(llm, query, answer):
    prompt = (
        f"Given the user's question: \"{query}\" and the answer: \"{answer}\", "
        "generate 3 concise, relevant follow-up questions the user might ask next about proforma invoices. "
        "Return only the questions as a numbered list."
    )
    messages = [HumanMessage(content=prompt)]
    try:
        followup_text = llm.invoke(messages).content
        questions = []
        for line in followup_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                q = line.split('.', 1)[-1].strip() if '.' in line else line
                if q:
                    questions.append(q)
            elif line:
                questions.append(line)
        questions = [q for q in questions if q]
        return questions[:3] if questions else []
    except Exception as e:
        return []

# ---- Chat Bubble UI (Dark, Modern, Gemini/Claude Style) ----
def chat_message(message, is_user=False):
    if is_user:
        st.markdown(
            f"""
            <div style='display:flex;justify-content:flex-start;'>
                <div style='background:linear-gradient(90deg,#232526 0%,#414345 100%);color:#fff;padding:14px 20px;border-radius:18px 18px 18px 5px;margin-bottom:10px;max-width:70%;box-shadow:0 2px 8px #1976d230;font-size:1.08rem;'>
                    <b style='color:#90caf9;'>You:</b> {message}
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display:flex;justify-content:flex-end;'>
                <div style='background:linear-gradient(90deg,#1976d2 0%,#0d47a1 100%);color:#fff;padding:14px 20px;border-radius:18px 18px 5px 18px;margin-bottom:10px;max-width:70%;box-shadow:0 2px 8px #1976d280;font-size:1.08rem;'>
                    <b style='color:#bbdefb;'>Assistant:</b> {message}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

# ---- Main App ----
def main():
    st.set_page_config(page_title="Invoice Assistant", page_icon="📄", layout="wide")

    # ---- TEST DATABASE CONNECTION ----
    test_db_connection()

    # Custom CSS for sticky input, dark modern UI
    st.markdown("""
        <style>
        .block-container {padding-top: 1.5rem;}
        body, [data-testid="stAppViewContainer"] {
            background: #181A1B !important;
            color: #fff !important;
        }
        .main-center {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 2.5rem;
        }
        .chatbox-container {
            background: #232526;
            border-radius: 20px;
            box-shadow: 0 6px 32px #1976d220;
            padding: 2.5rem 2rem 1.5rem 2rem;
            max-width: 650px;
            min-width: 320px;
            width: 100%;
            margin: auto;
            margin-bottom: 100px; /* space for sticky input */
        }
        .sticky-input-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100vw;
            background: #232526;
            box-shadow: 0 -2px 16px #1976d220;
            padding-top: 18px;
            padding-bottom: 18px;
            z-index: 100;
        }
        .stTextInput > div > div > input {
            background: #181A1B !important;
            color: #fff !important;
            border-radius: 10px;
            border: 1.5px solid #90caf9;
            font-size:1.1rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%);
            color: #fff;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            margin-right: 0.5rem;
            font-size:1.08rem;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #64b5f6 0%, #1976d2 100%);
            color: #fff;
        }
        .stForm {
            background: transparent !important;
            border-radius: 0;
            padding: 0;
            margin-bottom: 0;
            box-shadow: none;
        }
        .followup-row {
            display: flex;
            justify-content: flex-end;
            gap: 0.7rem;
            margin-top: 0.7rem;
            margin-bottom: 1.5rem;
        }
        .followup-btn {
            background: linear-gradient(90deg,#232526 0%,#414345 100%);
            color: #90caf9;
            border: none;
            border-radius: 999px;
            padding: 0.4rem 1.1rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 1px 4px #1976d220;
            transition: background 0.2s;
        }
        .followup-btn:hover {
            background: linear-gradient(90deg,#414345 0%,#232526 100%);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #90caf9 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    login()
    username = st.session_state["username"]

    st.sidebar.title("Invoice Assistant")
    st.sidebar.info("Ask anything about proforma invoices. Powered by RAG + Gemini Pro.")

    st.markdown("<div class='main-center'>", unsafe_allow_html=True)
    st.markdown("<div class='chatbox-container'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#90caf9;margin-bottom:1.7rem;'>📄 Proforma Invoice Chat Assistant</h2>", unsafe_allow_html=True)

    # Initialize components
    embeddings = get_embeddings_model()
    llm = get_gemini_model()
    vector_store = load_local_faiss_index(embeddings, LOCAL_FAISS_INDEX_DIR)

    # Session state for chat history and follow-up
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "followup" not in st.session_state:
        st.session_state["followup"] = []

    # Display chat history (user left, assistant right)
    for idx, msg in enumerate(st.session_state["chat_history"]):
        chat_message(msg["content"], is_user=(msg["role"] == "user"))
        # Show follow-up after assistant response only
        if msg["role"] == "assistant" and idx == len(st.session_state["chat_history"]) - 1 and st.session_state["followup"]:
            # Follow-up buttons, right aligned
            st.markdown("<div class='followup-row'>", unsafe_allow_html=True)
            for i, q in enumerate(st.session_state["followup"]):
                if st.button(q, key=f"followup_{i}", help="Ask this follow-up question"):
                    st.session_state["chat_history"].append({"role": "user", "content": q})
                    with st.spinner("Searching documents and generating response..."):
                        docs = get_similar_docs(vector_store, q)
                        response = generate_response(llm, q, docs)
                        st.session_state["chat_history"].append({"role": "assistant", "content": response})
                        followups = generate_followup_questions(llm, q, response)
                        st.session_state["followup"] = followups
                        # ---- SAVE CHAT TO POSTGRES ----
                        save_chat(username, q, response)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close chatbox-container
    st.markdown("</div>", unsafe_allow_html=True)  # close main-center

    # Sticky input bar at the bottom
    sticky_input = st.empty()
    with sticky_input.container():
        st.markdown("<div class='sticky-input-bar'>", unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your question about proforma invoices...", key="input")
            submitted = st.form_submit_button("Send")
            if submitted and user_input:
                st.session_state["chat_history"].append({"role": "user", "content": user_input})
                with st.spinner("Searching documents and generating response..."):
                    docs = get_similar_docs(vector_store, user_input)
                    response = generate_response(llm, user_input, docs)
                    st.session_state["chat_history"].append({"role": "assistant", "content": response})
                    followups = generate_followup_questions(llm, user_input, response)
                    st.session_state["followup"] = followups
                    # ---- SAVE CHAT TO POSTGRES ----
                    save_chat(username, user_input, response)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #90caf9; margin-top:2rem;'>", unsafe_allow_html=True)
    st.caption(
        "<div style='color:#90caf9; text-align:center;'>© 2024 Invoice Assistant | Powered by RAG + Gemini Pro</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
