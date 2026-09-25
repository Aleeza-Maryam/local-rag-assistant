"""Streamlit UI for the Local Knowledge Base RAG Assistant."""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path so `src` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.ingest import ingest_pdf
from src.retriever import Retriever
from src.generator import generate_answer
from src.utils import get_vector_db_path


# ---------- Page config ----------
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }

    /* Header styling */
    .app-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #2a2d35;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #fafafa;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #8b8f99;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Section headers */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Source card */
    .source-card {
        background-color: #16181d;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }
    .source-card .source-title {
        color: #93c5fd;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .source-card .source-meta {
        color: #6b7280;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .source-card .source-text {
        color: #d1d5db;
        line-height: 1.5;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #fafafa;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        border: 1px solid #2a2d35;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #93c5fd;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0c10;
        border-right: 1px solid #1f2228;
    }

    /* Divider */
    hr {
        border-color: #1f2228;
        margin: 1rem 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #8b8f99;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 8px;
        border: 1px solid #2a2d35;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    with st.spinner("Initializing embedding model..."):
        st.session_state.retriever = Retriever()


# ---------- Header ----------
st.markdown("""
<div class="app-header">
    <h1>Local Knowledge Base RAG Assistant</h1>
    <p>Upload PDFs, ask questions, and receive answers with exact citations from your documents.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "groq"],
        index=0,
        help="Gemini uses Google's models. Groq uses Llama for fast inference.",
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of document chunks to retrieve for each query.",
    )

    st.divider()

    st.markdown('<div class="section-label">Document Ingestion</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("Ingest Documents", use_container_width=True):
        progress = st.progress(0, text="Processing documents...")
        total_chunks = 0
        file_count = len(uploaded_files)

        for i, uploaded in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                chunks = ingest_pdf(tmp_path)
                for c in chunks:
                    c["source"] = uploaded.name
                added = st.session_state.retriever.add_chunks(chunks)
                total_chunks += added
                st.success(f"{uploaded.name} — {added} chunks indexed")
            except Exception as e:
                st.error(f"{uploaded.name} — failed: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            progress.progress((i + 1) / file_count, text=f"Processed {i+1}/{file_count}")

        progress.empty()
        st.info(f"Indexed {total_chunks} new chunks. Total: {st.session_state.retriever.count()}")

    st.divider()

    st.markdown('<div class="section-label">Statistics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", st.session_state.retriever.count())
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.divider()

    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Reset DB", use_container_width=True):
            st.session_state.retriever.reset()
            st.session_state.messages = []
            st.success("Vector database reset.")
            st.rerun()

    st.divider()
    st.caption(f"Database: `{get_vector_db_path()}`")


# ---------- Chat display ----------
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #6b7280;">
        <div style="font-size: 1rem; margin-bottom: 0.5rem;">No conversation yet</div>
        <div style="font-size: 0.875rem;">Upload a PDF in the sidebar, then ask a question below.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander(f"View {len(msg['sources'])} source(s)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{src['source']}</div>
                            <div class="source-meta">Page {src['page']} &nbsp;|&nbsp; Chunk {src.get('chunk_index', i-1)} &nbsp;|&nbsp; Rank {i}</div>
                            <div class="source-text">{src['text'][:400]}{'...' if len(src['text']) > 400 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ---------- Chat input ----------
query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if st.session_state.retriever.count() == 0:
            answer = "No documents in the knowledge base. Please upload PDF files first."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("Searching documents..."):
                hits = st.session_state.retriever.search(query, top_k=top_k)

            with st.spinner(f"Generating answer with {provider}..."):
                try:
                    answer = generate_answer(query, hits, provider=provider)
                except Exception as e:
                    answer = f"Error from {provider}: {e}"
                    st.error(answer)

            if not answer.startswith("Error"):
                st.markdown(answer)

                if hits:
                    with st.expander(f"View {len(hits)} source(s)"):
                        for i, src in enumerate(hits, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">{src['source']}</div>
                                <div class="source-meta">Page {src['page']} &nbsp;|&nbsp; Chunk {src.get('chunk_index', i-1)} &nbsp;|&nbsp; Rank {i}</div>
                                <div class="source-text">{src['text'][:400]}{'...' if len(src['text']) > 400 else ''}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": hits,
                })