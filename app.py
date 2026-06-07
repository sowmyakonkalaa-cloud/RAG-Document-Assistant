import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.write("Upload a PDF and ask questions about it!")

groq_api_key = st.text_input("Enter your Groq API key:", type="password")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

    with st.spinner("Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    st.success(f"PDF loaded! {len(chunks)} chunks created.")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Finding answer..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])

            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                api_key=groq_api_key
            )

            prompt = f"""Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""

            response = llm.invoke(prompt)

        st.write("**Answer:**")
        st.write(response.content)
