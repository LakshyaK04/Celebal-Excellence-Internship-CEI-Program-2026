"""
vectorstore.py — Stages 1-4 of the RAG Pipeline

Stage 1: Document Ingestion — Load PDFs, TXT files, or HuggingFace dataset
Stage 2: Text Chunking — Split text into smaller chunks for better retrieval
Stage 3: Embedding Creation — Convert chunks into vector representations
Stage 4: Vector Database — Store embeddings in FAISS for similarity search
"""

import os
import json
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Stage 1: Document Ingestion
# ---------------------------------------------------------------------------

def load_uploaded_files(uploaded_files):
    """
    Load documents from uploaded files (PDF or TXT).
    
    Accepts Streamlit UploadedFile objects, saves them temporarily,
    and extracts text content using appropriate loaders.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        List of LangChain Document objects with text and metadata
    """
    documents = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(docs)

            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(docs)

            else:
                print(f"Unsupported file type: {suffix}")
        finally:
            os.unlink(tmp_path)

    return documents


def load_huggingface_dataset(num_papers=10):
    """
    Load documents from the HuggingFace vectara/open_ragbench dataset.
    
    Downloads arXiv research papers from the dataset, extracts text
    from their sections, and converts them into LangChain Documents.
    
    Args:
        num_papers: Number of papers to load (default 10)
        
    Returns:
        Tuple of (documents, sample_queries) where:
          - documents: List of LangChain Document objects
          - sample_queries: Dict of sample question-answer pairs for testing
    """
    from huggingface_hub import snapshot_download

    cache_dir = "d:/CODING FILES/Celebal Internship/week7/.hf_cache"
    os.environ["HF_HOME"] = cache_dir

    local_dir = snapshot_download(
        repo_id="vectara/open_ragbench",
        repo_type="dataset",
        allow_patterns="pdf/arxiv/*",
        cache_dir=cache_dir
    )

    pdf_arxiv_dir = os.path.join(local_dir, "pdf", "arxiv")
    corpus_dir = os.path.join(pdf_arxiv_dir, "corpus")

    documents = []


    if os.path.exists(corpus_dir):
        corpus_files = [f for f in os.listdir(corpus_dir) if f.endswith(".json")]
        count = 0
        for filename in corpus_files:
            if count >= num_papers:
                break

            filepath = os.path.join(corpus_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                continue

            paper_id = item.get("id", filename.replace(".json", ""))
            title = item.get("title", "Untitled")
            abstract = item.get("abstract", "")
            sections = item.get("sections", [])


            if abstract:
                documents.append(Document(
                    page_content=f"Title: {title}\n\nAbstract: {abstract}",
                    metadata={
                        "source": f"HuggingFace: {paper_id}",
                        "title": title,
                        "paper_id": paper_id,
                        "section": "abstract",
                    }
                ))


            for i, section in enumerate(sections):
                text = section.get("text", "") if isinstance(section, dict) else str(section)
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"HuggingFace: {paper_id}",
                            "title": title,
                            "paper_id": paper_id,
                            "section": f"section_{i}",
                        }
                    ))

            count += 1


    sample_queries = {}
    try:
        queries_path = os.path.join(pdf_arxiv_dir, "queries.json")
        answers_path = os.path.join(pdf_arxiv_dir, "answers.json")

        if os.path.exists(queries_path) and os.path.exists(answers_path):
            with open(queries_path, "r", encoding="utf-8") as f:
                queries_data = json.load(f)
            with open(answers_path, "r", encoding="utf-8") as f:
                answers_data = json.load(f)


            count = 0
            for q_id, q_info in queries_data.items():
                if count >= 5:
                    break
                query_text = q_info.get("query") if isinstance(q_info, dict) else str(q_info)
                answer_text = answers_data.get(q_id, "No answer available")
                if query_text:
                    sample_queries[query_text] = answer_text
                    count += 1
    except Exception:
        pass

    return documents, sample_queries



# ---------------------------------------------------------------------------
# Stage 2: Text Chunking
# ---------------------------------------------------------------------------

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better retrieval accuracy.
    
    Uses RecursiveCharacterTextSplitter which splits on paragraphs,
    sentences, and words in that order, preserving natural boundaries.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked Document objects with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


# ---------------------------------------------------------------------------
# Stage 3: Embedding Creation
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Initialize the embedding model for converting text into vectors.
    
    Uses HuggingFace's all-MiniLM-L6-v2 model which runs locally
    (no API key needed) and produces 384-dimensional embeddings.
    
    Returns:
        HuggingFaceEmbeddings model instance
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


# ---------------------------------------------------------------------------
# Stage 4: Vector Database
# ---------------------------------------------------------------------------

def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector store from document chunks.
    
    Embeds all chunks using the embedding model and indexes them
    in a FAISS vector store for efficient similarity search.
    
    Args:
        chunks: List of chunked Document objects
        embedding_model: HuggingFaceEmbeddings model instance
        
    Returns:
        FAISS vector store instance
    """
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store
