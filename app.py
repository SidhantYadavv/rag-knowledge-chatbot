import streamlit as st
import os
from rag_pipeline import build_qa_chain
from ingest import ingest_uploaded_file

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

/* ── Suggestion chips ── */
.chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 32px;
}
.chip {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 9999px;
    padding: 6px 16px;
    font-size: 0.82rem;
    color: #a78bfa;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover {
    background: rgba(99,102,241,0.25);
    border-color: rgba(99,102,241,0.5);
}

/* ── Chat messages ── */
.msg-container { max-width: 780px; margin: 0 auto; }

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
}

.msg-assistant {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
}
.msg-assistant .avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    padding-top: 4px;
}
.msg-assistant .bubble {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #e2e8f0;
    backdrop-filter: blur(10px);
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

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chain_ready" not in st.session_state:
    st.session_state.chain_ready = False

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

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_docs]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)..."):
                for f in new_files:
                    chunks = ingest_uploaded_file(f)
                    st.session_state.indexed_docs.append(f.name)
                # Rebuild chain with fresh vectorstore
                st.session_state.qa_chain = build_qa_chain()
                st.session_state.chain_ready = True
            st.success(f"✅ {len(new_files)} doc(s) indexed!")

    # ── Indexed docs list ──
    if st.session_state.indexed_docs:
        st.markdown('<div class="sidebar-label" style="margin-top:20px;">📄 Indexed Documents</div>', unsafe_allow_html=True)
        for doc in st.session_state.indexed_docs:
            name = doc[:28] + "…" if len(doc) > 30 else doc
            st.markdown(f'<div class="doc-badge"><span class="dot">●</span>{name}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Clear chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── About ──
    st.markdown('<div class="sidebar-label" style="margin-top:16px;">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#475569; line-height:1.6;">
    Upload any PDF and ask questions about it.
    Powered by <span style="color:#a78bfa;">Gemini 1.5 Flash</span> +
    <span style="color:#60a5fa;">ChromaDB</span> for
    grounded, citation-backed answers.
    </div>
    """, unsafe_allow_html=True)

# ── Load chain on startup (pre-indexed docs) ──────────────────────────────────
@st.cache_resource
def load_default_chain():
    """Load chain from existing chroma_db if it already has data."""
    try:
        chain = build_qa_chain()
        # Quick check: try to get collection count
        return chain
    except Exception:
        return None

if not st.session_state.chain_ready:
    chain = load_default_chain()
    if chain is not None:
        st.session_state.qa_chain = chain
        st.session_state.chain_ready = True

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
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <p>Upload a PDF in the sidebar, then ask me anything about it.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    for src in msg["sources"]:
                        src_short = os.path.basename(src)
                        sources_html += f'<span class="source-tag">📄 {src_short}</span>'
                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="avatar">🤖</div>
                    <div class="bubble">
                        {msg["content"]}
                        <div>{sources_html}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})

    if not st.session_state.chain_ready or st.session_state.qa_chain is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ No documents indexed yet. Please upload a PDF using the sidebar first.",
            "sources": [],
        })
        st.rerun()

    with st.spinner("Thinking…"):
        try:
            result = st.session_state.qa_chain({"query": question})
            answer = result["result"]
            sources = list(set(
                doc.metadata.get("source", "Unknown")
                for doc in result.get("source_documents", [])
            ))
        except Exception as e:
            answer = f"❌ An error occurred: {str(e)}"
            sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()
