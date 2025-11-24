from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pinecone import Pinecone, ServerlessSpec
import os

# Pinecone Setup
# -----------------------------
os.environ["PINECONE_API_KEY"] = "pcsk_2SrqWF_TwCsRJJ8ALB37vTgan1GhMuXv2gmJPwW3mYDGDivkvGCcvAr7JfH9Xm369rWWgm"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


# Load PDF & Chunk

def load_pdf(data_dir):
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

extracted_data = load_pdf("data/")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(extracted_data)


# Embeddings & Vector Store
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)