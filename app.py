import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from faiss_manager import get_faiss_index, fetch_faiss_index_from_s3

# Access secrets from Streamlit Cloud
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

# Fetch FAISS index from S3 on startup and initialize locally if needed
if fetch_faiss_index_from_s3():
    print("Successfully loaded FAISS index from S3.")
else:
    print("Failed to load FAISS index from S3.")

# Function to process user queries and generate structured output
def query_proforma_rag(query):
    """Query the RAG model using FAISS index and generate structured output."""
    vector_store = get_faiss_index()
    if not vector_store:
        return "FAISS index not available. Please wait for initialization or the next scheduled update.", []

    # Initialize retriever and LLM
    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    )

    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Run query through the chain
    response = chain.run(query)

    # Retrieve relevant documents for structuring output
    sources = retriever.get_relevant_documents(query)

    # Generate structured output
    structured_output = []
    for source in sources:
        structured_output.append({
            "text": source.page_content,
            "source": source.metadata.get("source", "Unknown")
        })

    return response, structured_output

# Streamlit UI Setup
st.title("Proforma Invoice Query System")

# Sidebar Section: Instructions
st.sidebar.header("Instructions")
st.sidebar.write("Enter your query below to fetch relevant information from proforma invoices.")

# Main Section: Query Input and Response Display
st.header("Query Proforma Invoices")

# Initialize chat history in session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages (user and bot interactions)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "bot":
        st.chat_message("bot").write(message["content"])

# User input for chat interaction
if user_query := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get bot response and structured output using query_proforma_rag function
    bot_response, structured_output = query_proforma_rag(user_query)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Display bot response immediately in the chat interface without context
    st.chat_message("bot").write(bot_response)

    # Display structured output using Streamlit widgets
    st.header("Structured Output")
    for item in structured_output:
        with st.expander(f"Source: {item['source']}"):
            st.write(item['text'])
