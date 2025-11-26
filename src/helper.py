# src/helper.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use the new huggingface embeddings package
from langchain_huggingface import HuggingFaceEmbeddings

# Pinecone vectorstore import (used by store_index; app loads existing index)
from langchain_pinecone import PineconeVectorStore

def load_pdf(data_dir: str):
    """
    Returns list[Document] loaded from all PDFs in data_dir.
    """
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def split_text(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits documents into chunks (returns list[Document]).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns an embeddings object (langchain_huggingface.HuggingFaceEmbeddings).
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def build_pinecone_vectorstore(text_chunks, embeddings, index_name: str):
    """
    Builds / upserts text_chunks into Pinecone index and returns PineconeVectorStore.
    (text_chunks: iterable of Document objects)
    """
    # Create vectorstore from texts (upsert)
    texts = [chunk.page_content for chunk in text_chunks]
    # If you want metadata, pass list of dicts as 'metadatas' param
    vectorstore = PineconeVectorStore.from_texts(
        texts,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore


def get_vectorstore_for_existing_index(index_name: str, embeddings):
    """
    Return a PineconeVectorStore instance that points to an existing index.
    """
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)
