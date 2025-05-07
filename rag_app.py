import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---- CONFIG ----
MODEL_DIRECTORY = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_FAISS_INDEX_DIR = "C:/rag_with_api"
GEMINI_MODEL = "gemini-1.5-pro"
GEMINI_API_KEY = st.secrets["gemini_api_key"]


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


# ---- Main App ----
def main():
    st.set_page_config(page_title="Invoice Assistant", page_icon="📄", layout="wide")
    st.header("Proforma Invoice Q&A (k=50 Semantic Search)")

    # Initialize components
    embeddings = get_embeddings_model()
    llm = get_gemini_model()
    vector_store = load_local_faiss_index(embeddings, LOCAL_FAISS_INDEX_DIR)

    # Query interface
    query = st.text_input("Ask about proforma invoices:")
    if query:
        with st.spinner("Searching documents..."):
            docs = get_similar_docs(vector_store, query)
            response = generate_response(llm, query, docs)
            st.subheader("Answer:")
            st.write(response)


if __name__ == "__main__":
    main()
