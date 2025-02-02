import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the RAG templates
chat_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

summary_template = """
Summarize the following document in a concise and clear manner. Focus on the key points and provide a short summary.
Context: {context}
Summary:
"""

# Define paths
PDF_DIR = "data/"
CHROMA_DB_DIR = "chroma_db/"

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Initialize models
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = Chroma(collection_name="pdf_docs", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
model = OllamaLLM(model="deepseek-r1:1.5b")

def upload_pdf(file):
    """Save uploaded PDF to local directory."""
    file_path = os.path.join(PDF_DIR, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    """Load PDF content."""
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()   
    return documents

def split_text(documents):
    """Split text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    """Store document embeddings in ChromaDB."""
    vector_store.add_documents(documents)
    vector_store.persist()  # Automatically saves embeddings

def retrieve_docs(query):
    """Retrieve relevant documents based on query."""
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    """Generate answers based on retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(chat_template)    
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def summarize_document(documents):
    """Generate a summary of the document."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(summary_template)
    chain = prompt | model
    return chain.invoke({"context": context})

# Streamlit UI
st.title("📄 PDF Chat & Summarizer with ChromaDB")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    file_path = upload_pdf(uploaded_file)
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    # Tabs for Chat & Summary
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "📄 Summarize PDF"])

    with tab1:
        question = st.chat_input("Ask something about the document...")

        if question:
            st.chat_message("user").write(question)
            related_documents = retrieve_docs(question)
            answer = answer_question(question, related_documents)
            st.chat_message("assistant").write(answer)

    with tab2:
        if st.button("Generate Summary"):
            summary = summarize_document(chunked_documents)
            st.write("### Summary:")
            st.write(summary)
