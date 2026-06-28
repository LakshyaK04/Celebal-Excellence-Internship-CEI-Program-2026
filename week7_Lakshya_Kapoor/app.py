/"""
app.py — Streamlit UI for the RAG Document Question Answering System

Provides an interactive chatbot interface where users can:
1. Upload PDF or TXT documents, or load the HuggingFace vectara/open_ragbench dataset
2. Process documents through the RAG pipeline (ingest → chunk → embed → store)
3. Ask questions and receive answers grounded in the document content
"""

import os
os.environ["HF_HOME"] = "d:/CODING FILES/Celebal Internship/week7/.hf_cache"
import streamlit as st
from vectorstore import (
    load_uploaded_files,
    load_huggingface_dataset,
    chunk_documents,
    get_embedding_model,
    create_vector_store,
)
from chatbot import create_qa_chain, ask_question


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Question Answering System (RAG)")
st.markdown(
    "Upload your documents or load the HuggingFace dataset, "
    "then ask questions and get answers grounded in the content."
)

# ---------------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "sample_queries" not in st.session_state:
    st.session_state.sample_queries = {}


# ---------------------------------------------------------------------------
# Sidebar — Configuration & Document Loading
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")


    groq_api_key = st.text_input(
        "Groq API Key",
        value="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",
        type="password",
        help="Get a free API key from https://console.groq.com",
    )

    st.divider()


    st.header("📂 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Upload Files", "HuggingFace Dataset"],
        help="Upload your own PDFs/TXT files or load the vectara/open_ragbench dataset",
    )

    uploaded_files = None
    num_papers = 10

    if data_source == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

    else:
        num_papers = st.slider(
            "Number of papers to load",
            min_value=5,
            max_value=1000,
            value=10,
            help="Number of arXiv papers to load from the dataset",
        )

    st.divider()

    process_btn = st.button("▶️ Process Documents", use_container_width=True)

    if st.session_state.documents_processed:
        st.success(f"✅ {st.session_state.chunk_count} chunks indexed and ready!")


# ---------------------------------------------------------------------------
# Document Processing Logic
# ---------------------------------------------------------------------------

if process_btn:
    if not groq_api_key:
        st.error("❌ Please enter your Groq API key in the sidebar.")
        st.stop()

    if data_source == "Upload Files" and not uploaded_files:
        st.error("❌ Please upload at least one file.")
        st.stop()

    with st.spinner("Processing documents..."):
        progress = st.progress(0, text="Stage 1/4: Loading documents...")

        if data_source == "Upload Files":
            documents = load_uploaded_files(uploaded_files)
            st.session_state.sample_queries = {}
        else:
            documents, sample_queries = load_huggingface_dataset(num_papers)
            st.session_state.sample_queries = sample_queries

        if not documents:
            st.error("❌ No text could be extracted from the provided documents.")
            st.stop()

        progress.progress(25, text="Stage 2/4: Chunking text...")
        chunks = chunk_documents(documents)

        progress.progress(50, text="Stage 3/4: Creating embeddings...")
        embedding_model = get_embedding_model()

        progress.progress(75, text="Stage 4/4: Building vector store...")
        vector_store = create_vector_store(chunks, embedding_model)

        qa_chain = create_qa_chain(vector_store, groq_api_key)

        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.documents_processed = True
        st.session_state.chunk_count = len(chunks)
        st.session_state.chat_history = []

        progress.progress(100, text="Done!")
        st.success(
            f"✅ Processed {len(documents)} documents into {len(chunks)} chunks. "
            "You can now ask questions!"
        )


# ---------------------------------------------------------------------------
# Sample Queries (when using HuggingFace dataset)
# ---------------------------------------------------------------------------

if st.session_state.sample_queries and st.session_state.documents_processed:
    with st.expander("💡 Sample Questions from Dataset"):
        for query, answer in list(st.session_state.sample_queries.items())[:5]:
            st.markdown(f"**Q:** {query}")
            st.markdown(f"**Expected A:** {answer[:200]}...")
            st.divider()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Source Documents"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i + 1}** — *{source['source']}*")
                    st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                    st.divider()


if question := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.documents_processed:
        st.error("❌ Please process documents first using the sidebar.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            result = ask_question(st.session_state.qa_chain, question)

        st.markdown(result["answer"])

        sources = []
        if result["source_documents"]:
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    source_name = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {i + 1}** — *{source_name}*")
                    content = doc.page_content
                    st.text(content[:300] + "..." if len(content) > 300 else content)
                    st.divider()
                    sources.append({
                        "source": source_name,
                        "content": content,
                    })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
        })
