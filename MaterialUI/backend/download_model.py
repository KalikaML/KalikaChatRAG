from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("BAAI/BAAI/bge-base-en-v1.5", cache_dir="./BAAI-bge-base-en-v1.5")
tokenizer = AutoTokenizer.from_pretrained("BAAI/BAAI/bge-base-en-v1.5", cache_dir="./BAAI-bge-base-en-v1.5")
