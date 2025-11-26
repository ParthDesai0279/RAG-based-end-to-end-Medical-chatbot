# store_index.py
import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, split_text, download_hugging_face_embeddings, build_pinecone_vectorstore

# config
index_name = os.getenv("PINECONE_INDEX", "medical-chatbot")
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("PINECONE_API_KEY missing in environment")

# initialize pinecone client (minimal)
pc = Pinecone(api_key=api_key)

# create index if doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # matches sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# load & chunk
print("Loading PDFs...")
documents = load_pdf("data/")    # folder with PDFs
print(f"Loaded {len(documents)} documents")

print("Splitting into chunks...")
chunks = split_text(documents, chunk_size=1000, chunk_overlap=200)
print(f"Created {len(chunks)} chunks")

# embeddings
print("Preparing embeddings...")
embeddings = download_hugging_face_embeddings()

# upsert to pinecone
print("Building/upserting vectorstore...")
vectorstore = build_pinecone_vectorstore(chunks, embeddings, index_name)
print("Indexing complete.")
