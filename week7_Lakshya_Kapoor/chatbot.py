"""
chatbot.py — Stages 5-7 of the RAG Pipeline

Stage 5: Query Processing — Convert user question into an embedding
Stage 6: Context Retrieval — Retrieve most relevant chunks from vector database
Stage 7: Answer Generation — Generate answer using LLM with retrieved context
"""

from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from documents.

Use ONLY the following context to answer the question. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# Stages 5-7: Query Processing, Context Retrieval, Answer Generation
# ---------------------------------------------------------------------------

def create_qa_chain(vector_store, groq_api_key):
    """
    Create a RetrievalQA chain that handles stages 5-7 of the pipeline.
    
    Stage 5 (Query Processing): The chain automatically converts the user's
    question into an embedding using the same model used for document chunks.
    
    Stage 6 (Context Retrieval): The chain retrieves the top-k most relevant
    chunks from the FAISS vector store using similarity search.
    
    Stage 7 (Answer Generation): The retrieved chunks are passed as context
    to the Groq LLM, which generates a grounded answer.
    
    Args:
        vector_store: FAISS vector store with embedded document chunks
        groq_api_key: API key for Groq LLM service
        
    Returns:
        RetrievalQA chain instance
    """
    # Initialize the language model (Groq provides fast, free inference)
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    # Create a retriever from the vector store
    # k=5 means we retrieve the 5 most relevant chunks for each query
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    return qa_chain


def ask_question(qa_chain, question):
    """
    Run the full RAG pipeline: query → retrieve → generate.
    
    Takes a user question, retrieves relevant document chunks,
    and generates a grounded answer using the LLM.
    
    Args:
        qa_chain: RetrievalQA chain instance
        question: User's question string
        
    Returns:
        Dictionary with:
          - "answer": Generated answer text
          - "source_documents": List of retrieved Document chunks used as context
    """
    result = qa_chain.invoke({"query": question})

    return {
        "answer": result.get("result", "No answer generated."),
        "source_documents": result.get("source_documents", []),
    }
