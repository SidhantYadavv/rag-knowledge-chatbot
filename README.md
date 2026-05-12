#  RAG Knowledge Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with your own documents. Upload any PDF and ask it questions — the AI retrieves the most relevant passages and generates accurate, grounded answers.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | Google Gemini 1.5 Flash + Embedding-001 |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| UI | Streamlit |
| Language | Python 3.11+ |

## Architecture

```
PDF Documents → Chunking → Embeddings → ChromaDB
                                             ↓
User Question → Embed Question → Retrieve Top-K Chunks → Gemini LLM → Answer
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-chatbot.git
cd rag-knowledge-chatbot
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com)

### 5. Add your PDFs
Drop any PDF files into the `docs/` folder.

### 6. Ingest your documents
```bash
python ingest.py
```

### 7. Run the chatbot
```bash
streamlit run app.py
```

## 📁 Project Structure

```
rag-knowledge-chatbot/
├── app.py              # Streamlit chat UI
├── rag_pipeline.py     # RAG chain (retrieval + generation)
├── ingest.py           # PDF loading, chunking & embedding
├── docs/               # Drop your PDFs here
├── requirements.txt
└── .env                # API key (never committed)
```
