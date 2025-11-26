# app.py
from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()

import os
from src.helper import download_hugging_face_embeddings, get_vectorstore_for_existing_index
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment")

# embeddings + vectorstore (existing index)
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# Prompt: context + question -> answer
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}

Helpful answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Local LLM (CTransformers) — adjust model path to yours
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8}
)

# Build retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# Build runnable RAG pipeline:
# It will call retriever to get 'context', fill the prompt, call llm, and parse string output.
rag_pipeline = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form.get("msg", "")
    if not user_msg:
        return "No message received", 400

    # invoke pipeline. pass a mapping: the runnable uses 'question' as the passthrough
    out = rag_pipeline.invoke({"question": user_msg})
    # StrOutputParser returns a string
    if isinstance(out, dict) and "result" in out:
        answer = out["result"]
    else:
        answer = out if isinstance(out, str) else str(out)

    return answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
