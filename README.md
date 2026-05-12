# RAG Knowledge Chatbot: Production-Style Document Intelligence

A Retrieval-Augmented Generation (RAG) pipeline designed to transform static PDF documents into interactive knowledge bases. This system leverages Google Gemini 1.5 Flash and LangChain to provide accurate, grounded, and context-aware responses to user queries based on private document sets.

---

## Key Features

- **Semantic Document Retrieval**: Utilizes advanced vector embeddings to understand the intent and context of questions, moving beyond simple keyword matching.
- **Grounded AI Responses**: Minimizes hallucinations by anchoring the Large Language Model's (LLM) answers strictly to retrieved passages from uploaded documents.
- **Automated PDF Ingestion**: Support for multi-page PDF documents with automated text extraction, chunking, and metadata preservation.
- **Interactive Chat Interface**: A responsive Streamlit-based web application for an intuitive user experience.
- **Scalable Vector Store**: Built on ChromaDB, allowing for fast and efficient retrieval even as the document library expands.

---

## Technical Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM** | Google Gemini 1.5 Flash | Core reasoning and response generation |
| **Embeddings** | Google Embedding-001 | Conversion of text into high-dimensional vectors |
| **Orchestration** | LangChain | Management of data flow between retrieval and generation |
| **Vector Database** | ChromaDB | High-performance storage and retrieval of embeddings |
| **Frontend** | Streamlit | Deployment of the interactive web application |
| **Language** | Python 3.11+ | Implementation language for the entire pipeline |

---

## Technical Architecture

The pipeline follows a Retrieval-Augmented Generation pattern, divided into two distinct phases:

### 1. Data Ingestion (Offline Phase)
1. **Document Loading**: Extraction of text from source PDF files.
2. **Text Chunking**: Segmenting text into manageable units using recursive character splitting.
3. **Vectorization**: Converting text chunks into numerical vectors via Google's embedding models.
4. **Persistence**: Storing vectors and original text in the ChromaDB vector database.

### 2. Retrieval and Generation (Online Phase)
1. **Query Processing**: Converting the user's question into a vector embedding.
2. **Similarity Search**: Identifying the most relevant text chunks within ChromaDB.
3. **Context Construction**: Aggregating retrieved passages into a coherent context for the LLM.
4. **Response Generation**: Prompting Gemini 1.5 Flash to answer the question using only the provided context.

---

## Getting Started

### Prerequisites
- Python 3.11 or higher
- A Google Cloud Project with the Generative AI API enabled

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/SidhantYadavv/rag-knowledge-chatbot.git
cd rag-knowledge-chatbot

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Google API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
API keys can be obtained from the Google AI Studio.

### 3. Populating the Knowledge Base
Place PDF files in the `docs/` folder and execute the ingestion script:
```bash
python ingest.py
```

### 4. Launching the Application
```bash
streamlit run app.py
```

---

## Project Structure

- `app.py`: The entry point for the Streamlit application; manages UI state and user interaction.
- `rag_pipeline.py`: Implements the LangChain retrieval chain logic, including prompt templates and LLM configuration.
- `ingest.py`: Handles PDF processing, text splitting, and vector database population.
- `docs/`: Storage directory for source PDF documents.
- `chroma_db/`: Local directory for the persisted vector database.

---

## Implementation Best Practices
- **Chunk Overlap**: Preserves context across split boundaries to ensure no information is lost at the edges of chunks.
- **Environment Management**: Secure handling of API keys via environment variables.
- **Modular Architecture**: Clean separation between data ingestion and inference logic for improved maintainability.

---

Developed by Sidhant Yadav