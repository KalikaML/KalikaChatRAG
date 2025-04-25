import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from faiss_utils import (
    check_s3_connection,
    download_faiss_index,
    load_faiss_index,
    load_embedding_model,
    embed_text,
)
load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

faiss_index = None
embedding_tokenizer = None
embedding_model = None

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class QueryRequest(BaseModel):
    query: str
    source: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    if req.username == "admin" and req.password == "admin@123":
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/check-s3")
def api_check_s3():
    success, message = check_s3_connection()
    if not success:
        raise HTTPException(status_code=500, detail=message)
    return {"status": "success", "message": message}

@app.get("/download-faiss")
def api_download_faiss():
    success, message = download_faiss_index()
    if not success:
        raise HTTPException(status_code=500, detail=message)
    return {"status": "success", "message": message}

@app.get("/load-faiss")
def api_load_faiss():
    global faiss_index, embedding_tokenizer, embedding_model
    success, result = load_faiss_index()
    if not success:
        raise HTTPException(status_code=500, detail=result)
    faiss_index = result
    success, model_result = load_embedding_model()
    if not success:
        raise HTTPException(status_code=500, detail=model_result)
    embedding_tokenizer, embedding_model = model_result
    return {"status": "success", "message": "FAISS index and embedding model loaded"}

@app.post("/query")
def api_query(request: QueryRequest):
    global faiss_index, embedding_tokenizer, embedding_model
    if request.source == "kb":
        return {"response": "Knowledge base functionality is not implemented yet. Please select FAISS option."}
    if request.source == "faiss":
        if faiss_index is None or embedding_tokenizer is None or embedding_model is None:
            raise HTTPException(status_code=400, detail="FAISS index or embedding model not loaded")
        query_embedding = embed_text(request.query, embedding_tokenizer, embedding_model)
        k = 15
        print(faiss_index, query_embedding)
        distances, indices = faiss_index.search(query_embedding, k)
        retrieved_texts = [f"Proforma invoice data for index {idx}" for idx in indices[0]]
        prompt = f"User query: {request.query}\nRelevant info:\n" + "\n".join(retrieved_texts) + "\nAnswer:"
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return {"response": response.text}
    raise HTTPException(status_code=400, detail="Invalid source selected")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)