import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCS_DIR, get_vectorstore

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def ingest_pdf(vectorstore: Chroma, path: str, source_name: str) -> int:
    """Index one PDF under `source_name`, replacing any chunks previously indexed under that name.

    Chunk IDs are deterministic, so re-ingesting the same file never creates duplicates.
    """
    documents = PyPDFLoader(path).load()
    for doc in documents:
        doc.metadata["source"] = source_name
    chunks = splitter.split_documents(documents)

    existing = vectorstore.get(where={"source": source_name}, include=[])["ids"]
    if existing:
        vectorstore.delete(ids=existing)
    if chunks:
        ids = [f"{source_name}:{i}" for i in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)


def ingest_uploaded_file(vectorstore: Chroma, uploaded_file) -> int:
    """Ingest a Streamlit UploadedFile. Returns chunk count."""
    # PyPDFLoader needs a path on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return ingest_pdf(vectorstore, tmp_path, uploaded_file.name)
    finally:
        os.unlink(tmp_path)


def list_indexed_sources(vectorstore: Chroma) -> list[str]:
    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metadatas if m and "source" in m})


def ingest_docs():
    """Ingest all PDFs from the docs/ directory into ChromaDB."""
    pdfs = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}/")
        return 0
    vectorstore = get_vectorstore()
    total = 0
    for name in pdfs:
        count = ingest_pdf(vectorstore, os.path.join(DOCS_DIR, name), name)
        print(f"  {name}: {count} chunks")
        total += count
    print(f"✅ Ingested {total} chunks from {len(pdfs)} file(s) into ChromaDB")
    return total


if __name__ == "__main__":
    ingest_docs()
