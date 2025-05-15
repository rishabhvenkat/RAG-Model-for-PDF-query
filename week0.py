# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from week0gpt import get_answer

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load and index PDF
@st.cache_resource
def load_pdf_and_create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Streamlit UI
st.set_page_config(page_title="PDF Chat with GPT-4o", layout="centered")
st.title("Chat with PDF using GPT-4o")
st.markdown("Ask questions about `10_OE_AM_Chassis.pdf`")

db = load_pdf_and_create_vectorstore("10_OE_AM_Chassis.pdf")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        relevant_docs = db.similarity_search(query, k=4)
        answer = get_answer(query, relevant_docs)
        st.subheader("Answer:")
        st.write(answer)

        with st.expander("Source Chunks"):
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f"**Chunk {i}:** {doc.page_content}")

