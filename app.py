import html
import os
import streamlit as st
from config import CHAT_MODEL, get_vectorstore
from ingest import ingest_uploaded_file, list_indexed_sources
from rag_pipeline import build_qa_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root / Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161f 0%, #0d0f14 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 24px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}
.sidebar-logo .icon {
    font-size: 2rem;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    padding: 6px 10px;
    line-height: 1;
}
.sidebar-logo .text h2 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-logo .text p {
    margin: 0;
    font-size: 0.72rem;
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
}

/* ── Section labels ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #475569 !important;
    text-transform: uppercase;
    margin-bottom: 8px;
    padding-left: 2px;
}

/* ── File badge ── */
.doc-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 6px;
    font-size: 0.78rem;
}
.doc-badge .dot { color: #6366f1; font-size: 0.5rem; }

/* ── Main header ── */
.hero {
    text-align: center;
    padding: 48px 24px 32px 24px;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.hero p {
    font-size: 1rem;
    color: #64748b;
    margin: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    max-width: 780px;
    margin: 0 auto 12px auto;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 12px 18px;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Source tag ── */
.source-tag {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.2);
    color: #34d399 !important;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    margin-top: 8px;
    margin-right: 4px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 40px;
    color: #475569;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: 0.9rem; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background-color: rgba(255,255,255,0.1) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #fff !important;
}

/* ── Bottom Chat Container ── */
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"], div[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Input bar ── */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.5) !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > section {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px dashed rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] > section * {
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Hide streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Resources ─────────────────────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to a `.env` file in the project root and restart the app.")
    st.stop()


@st.cache_resource
def get_store():
    return get_vectorstore()


@st.cache_resource
def get_chain():
    return build_qa_chain(get_store())


vectorstore = get_store()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">🧠</div>
        <div class="text">
            <h2>Knowledge AI</h2>
            <p>RAG-powered document chat</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload PDFs ──
    st.markdown('<div class="sidebar-label">📁 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # The uploader keeps its files across reruns, so only index each upload once
    new_files = [f for f in uploaded_files or [] if f.file_id not in st.session_state.processed_uploads]
    if new_files:
        with st.spinner(f"Indexing {len(new_files)} file(s)..."):
            for f in new_files:
                try:
                    chunks = ingest_uploaded_file(vectorstore, f)
                except Exception as e:
                    st.error(f"Failed to index {f.name}: {e}")
                    continue
                st.session_state.processed_uploads.add(f.file_id)
                if chunks == 0:
                    st.warning(f"No extractable text in {f.name} (scanned PDF?)")
        st.success(f"✅ {len(new_files)} doc(s) processed")

    # ── Indexed docs list (read from the vector store, so it survives refreshes) ──
    indexed_docs = list_indexed_sources(vectorstore)
    if indexed_docs:
        st.markdown('<div class="sidebar-label" style="margin-top:20px;">📄 Indexed Documents</div>', unsafe_allow_html=True)
        for doc in indexed_docs:
            name = doc[:28] + "…" if len(doc) > 30 else doc
            st.markdown(f'<div class="doc-badge"><span class="dot">●</span>{html.escape(name)}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Clear chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── About ──
    st.markdown('<div class="sidebar-label" style="margin-top:16px;">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#475569; line-height:1.6;">
    Upload any PDF and ask questions about it.
    Powered by <span style="color:#a78bfa;">{html.escape(CHAT_MODEL)}</span> +
    <span style="color:#60a5fa;">ChromaDB</span> for
    grounded, citation-backed answers.
    </div>
    """, unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Knowledge Chatbot</h1>
    <p>Ask anything about your documents — grounded answers with source citations</p>
</div>
""", unsafe_allow_html=True)

# ── Suggested starter chips ───────────────────────────────────────────────────
SUGGESTIONS = [
    "Summarize the document",
    "What are the key findings?",
    "List the main topics covered",
    "What conclusions are drawn?",
]

if not st.session_state.messages:
    cols = st.columns(len(SUGGESTIONS))
    for i, (col, suggestion) in enumerate(zip(cols, SUGGESTIONS)):
        with col:
            if st.button(suggestion, key=f"chip_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
AVATARS = {"user": "🧑", "assistant": "🤖"}


def render_message(msg):
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            tags = "".join(f'<span class="source-tag">📄 {html.escape(src)}</span>' for src in msg["sources"])
            st.markdown(tags, unsafe_allow_html=True)


if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💬</div>
        <p>Upload a PDF in the sidebar, then ask me anything about it.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    render_message(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents…"):
    msg = {"role": "user", "content": question}
    st.session_state.messages.append(msg)
    render_message(msg)


def answer(question: str) -> dict:
    if not indexed_docs:
        return {
            "content": "⚠️ No documents indexed yet. Please upload a PDF using the sidebar first.",
            "sources": [],
        }
    try:
        result = get_chain().invoke(question)
    except Exception as e:
        return {"content": f"❌ An error occurred: {e}", "sources": []}
    # Unique source names in retrieval-rank order
    sources = list(dict.fromkeys(
        os.path.basename(doc.metadata.get("source", "Unknown")) for doc in result["source_documents"]
    ))
    return {"content": result["answer"], "sources": sources}


# Answer the latest question, whether it came from the input box or a suggestion chip
messages = st.session_state.messages
if messages and messages[-1]["role"] == "user":
    with st.spinner("Thinking…"):
        reply = {"role": "assistant", **answer(messages[-1]["content"])}
    messages.append(reply)
    render_message(reply)
