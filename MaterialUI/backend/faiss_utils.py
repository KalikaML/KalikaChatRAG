import os
import boto3
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging 

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
FAISS_INDEX_S3_KEY = os.getenv("FAISS_INDEX_S3_KEY")  # now configurable

def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3.list_buckets()
        logging.info("S3 client initialized successfully.")
        return s3
    except Exception as e:
        logging.error(f"Error initializing S3 client: {str(e)}")
        logging.error(f"Failed to connect to S3. Check AWS credentials and permissions. Error: {e}")
        return None

def check_s3_connection():
    try:
        s3 = get_s3_client()
        s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)
        return True, "S3 connection successful"
    except Exception as e:
        return False, str(e)

def download_faiss_index(local_path="faiss_index.idx", s3_key=None):
    s3_key = FAISS_INDEX_S3_KEY
    try:
        s3 = get_s3_client()
        s3.download_file(S3_BUCKET, s3_key, local_path)
        return True, f"Downloaded FAISS index to {local_path}"
    except Exception as e:
        return False, str(e)

def load_faiss_index(local_path="faiss_index.idx"):
    try:
        index = faiss.read_index(local_path)
        return True, index
    except Exception as e:
        return False, str(e)

def load_embedding_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("BAAI-bge-base-en-v1.5", local_files_only=True)
        model = AutoModel.from_pretrained("BAAI-bge-base-en-v1.5", local_files_only=True)
        model.eval()
        return True, (tokenizer, model)
    except Exception as e:
        return False, str(e)

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy()
