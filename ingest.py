import os
import tempfile
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def ingest_docs():
    """Ingest all PDFs from the docs/ directory into ChromaDB."""
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Ingested {len(chunks)} chunks from {len(documents)} pages into ChromaDB")
    return len(chunks)

def ingest_uploaded_file(uploaded_file) -> int:
    """Ingest a Streamlit UploadedFile object into ChromaDB. Returns chunk count."""
    # Save to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    # Tag source as original filename
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    # Load existing store and add (don't recreate from scratch)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    vectorstore.add_documents(chunks)

    return len(chunks)

if __name__ == "__main__":
    ingest_docs()
