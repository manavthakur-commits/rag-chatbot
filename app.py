"""Streamlit UI for the RAG chatbot: upload files, then ask questions about them."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()

st.set_page_config(
    page_title="Northwind Knowledge Assistant", page_icon="📄", layout="wide"
)
st.markdown(
    """
    <style>
    :root {
        --brand-ink: #172554;
        --brand-blue: #2563eb;
        --brand-violet: #7c3aed;
        --brand-muted: #64748b;
    }

    .main .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .brand-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.25rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 24px;
        background:
            radial-gradient(ellipse at 92% 12%, rgba(196, 181, 253, 0.42), transparent 34%),
            linear-gradient(120deg, #eff6ff 0%, #f8fafc 52%, #f5f3ff 100%);
        animation: brand-enter 650ms ease-out both;
    }

    .brand-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.8rem;
        color: var(--brand-blue);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .brand-mark {
        display: inline-grid;
        width: 1.7rem;
        height: 1.7rem;
        place-items: center;
        border-radius: 0.55rem;
        background: linear-gradient(135deg, var(--brand-blue), var(--brand-violet));
        color: white;
        font-size: 1rem;
        letter-spacing: 0;
    }

    .brand-hero h1 {
        margin: 0;
        color: var(--brand-ink);
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.1;
    }

    .brand-hero p {
        max-width: 660px;
        margin: 0.8rem 0 0;
        color: var(--brand-muted);
        font-size: 1.05rem;
        line-height: 1.65;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        animation: message-enter 350ms ease-out both;
    }

    .stButton > button {
        border-radius: 12px;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.15);
    }

    @keyframes brand-enter {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes message-enter {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    <header class="brand-hero">
        <div class="brand-kicker"><span class="brand-mark">▤</span> Northwind · Document intelligence</div>
        <h1>Northwind Knowledge Assistant</h1>
        <p class="brand-subtitle">Answers grounded in your documents, with citations.</p>
        <p>Use the API key and document upload panels below to build your knowledge base,
        then ask a question. Explore the cited source snippets to see where each
        answer came from.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(
        api_key=st.session_state.get("openai_api_key")
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

api_key_configured = bool(
    st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
)
api_column, documents_column = st.columns(2, gap="large")

with api_column:
    st.header("OpenAI API key")
    st.caption(
        "Enter your key for embeddings and answers. It stays in this browser "
        "session and is not written to a file."
    )
    with st.form("api_key_form"):
        entered_api_key = st.text_input(
            "API key",
            type="password",
            placeholder="sk-...",
            help="Your key is used only for this app session.",
        )
        apply_api_key = st.form_submit_button("Apply API key")

    if apply_api_key:
        if not entered_api_key.strip():
            st.error("Enter a valid API key.")
        else:
            st.session_state.openai_api_key = entered_api_key.strip()
            st.session_state.pipeline = RAGPipeline(
                api_key=st.session_state.openai_api_key
            )
            st.session_state.indexed = False
            st.success("API key applied for this session. Rebuild your knowledge base.")
    elif api_key_configured:
        st.success("API key configured")
    else:
        st.info("Add an API key to build a knowledge base and ask questions.")

with documents_column:
    st.header("Add your documents")
    st.caption("Upload one or more documents to build your knowledge base.")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or CSV", type=["pdf", "txt", "csv"], accept_multiple_files=True
    )
    if st.button(
        "Build knowledge base",
        disabled=not uploaded_files or not api_key_configured,
    ):
        with st.spinner("Loading, chunking, and embedding..."):
            tmp_dir = Path(tempfile.mkdtemp())
            paths = []
            for uploaded in uploaded_files:
                dest = tmp_dir / uploaded.name
                dest.write_bytes(uploaded.getbuffer())
                paths.append(str(dest))
            n_chunks = st.session_state.pipeline.ingest(paths)
            st.session_state.indexed = True
        st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed:
        st.success("Knowledge base ready ✅")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['chunk_id']}**")
            preview = source["text"][:300]
            if len(source["text"]) > 300:
                preview += "..."
            st.caption(preview)


st.header("Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.pipeline.answer(question)
        st.markdown(result.answer)
        render_sources(result.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
