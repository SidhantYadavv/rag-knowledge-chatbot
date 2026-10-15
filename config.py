import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
DOCS_DIR = "docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
