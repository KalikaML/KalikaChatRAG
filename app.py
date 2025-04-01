
import streamlit as st
import boto3
import os
import json
from typing import BinaryIO
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# AWS Configuration
s3 = boto3.client('s3',
                  aws_access_key_id=st.secrets['AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=st.secrets['AWS_SECRET_ACCESS_KEY'])
bucket_name = st.secrets['S3_BUCKET']


@st.cache_resource
def load_faiss_from_s3(folder: str):
    """Load FAISS index directly from S3 without local download"""
    index_file = f"{folder}/index.faiss"
    config_file = f"{folder}/index.pkl"

    # Load index files from S3
    index_obj = s3.get_object(Bucket=bucket_name, Key=index_file)
    config_obj = s3.get_object(Bucket=bucket_name, Key=config_file)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create FAISS instance from bytes
    return FAISS.deserialize_from_bytes(
        embeddings=embeddings,
        serialized=(
            index_obj['Body'].read(),
            config_obj['Body'].read()
        )
    )


def get_new_file_count(folder: str):
    """Count unprocessed files in S3 folder"""
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/raw/")
    return len(response.get('Contents', []))


# Streamlit UI
st.title("Proforma/PO Chat Assistant")
doc_type = st.selectbox("Select Document Type", ["Proforma", "Purchase Order"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display new file counts
proforma_count = get_new_file_count("proforma_invoice")
po_count = get_new_file_count("PO_Dump")
st.sidebar.markdown(f"**New Proforma Files:** {proforma_count}")
st.sidebar.markdown(f"**New PO Files:** {po_count}")

# Load appropriate FAISS index
folder_map = {
    "Proforma": "proforma_invoice",
    "Purchase Order": "PO_Dump"
}
selected_folder = folder_map[doc_type]
vector_db = load_faiss_from_s3(selected_folder)

# RAG Pipeline
prompt_template = """Use the following context to answer the question:
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Proforma/PO documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = qa_chain.invoke({"query": prompt})

    with st.chat_message("assistant"):
        st.markdown(response['result'])
    st.session_state.messages.append({"role": "assistant", "content": response['result']})
