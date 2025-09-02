import os
import fitz 
import docx
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import requests
import gradio as gr


# 1. DOCUMENT LOADING
def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text


# 2. CHUNKING
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[Document]:
    """Chunk large text into LangChain Document objects."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.create_documents([text])


# 3. STORE EMBEDDINGS
def store_in_chroma(documents: List[Document], persist_directory="chroma_db"):
    """Store document embeddings in Chroma vector DB."""
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


# 4. HYBRID SEARCH (BM25 + VECTOR)
def bm25_search(query: str, documents: List[Document], top_k=5):
    """Perform BM25 keyword search."""
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], scores[i]) for i in top_indices]

def vector_search(query: str, vectordb: Chroma, top_k=5):
    """Perform vector similarity search."""
    results = vectordb.similarity_search_with_score(query, k=top_k)
    return results

def hybrid_search(query: str, vectordb: Chroma, documents: List[Document], top_k=5):
    """Combine BM25 + vector search results."""
    bm25_results = bm25_search(query, documents, top_k=top_k)
    vector_results = vector_search(query, vectordb, top_k=top_k)
    
    combined = bm25_results + vector_results
    seen = set()
    unique_results = []
    for doc, score in combined:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append((doc, score))
    return unique_results[:top_k*2]  # keep a bit extra for reranking


# 5. RERANKING WITH CROSS-ENCODER
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: List[Tuple[Document, float]], top_k=5):
    """Re-rank retrieved documents using a cross-encoder."""
    pairs = [(query, d.page_content) for d, _ in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for (d, _), _ in scored_docs[:top_k]]


# 6. QUERY LLM
from typing import List
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"

def ask_llm(query: str, context_docs: List) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    # Combine retrieved documents into one context string
    context_text = "\n\n".join([d.page_content for d in context_docs])
    prompt = f"Answer the following question using the context only:\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    # Make the API request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract the response text
    return data["choices"][0]["message"]["content"]


# 7. FULL PIPELINE
def rag_pipeline(file_path, query):
    # Detect file type
    if file_path.endswith(".pdf"):
        text = load_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = load_docx(file_path)
    else:
        return "Unsupported file format."
    
    # Chunk & store
    docs = chunk_text(text)
    vectordb = store_in_chroma(docs)
    
    # Hybrid search + rerank
    hybrid_results = hybrid_search(query, vectordb, docs)
    reranked_docs = rerank(query, hybrid_results)
    
    # Query LLM
    answer = ask_llm(query, reranked_docs)
    return answer


# 8. GRADIO UI
def gradio_interface(file, query):
    return rag_pipeline(file.name, query)

with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Advanced RAG System with Hybrid Search + Reranking")
    file_input = gr.File(label="Upload PDF or DOCX")
    query_input = gr.Textbox(label="Ask a question")
    output = gr.Textbox(label="Answer")
    submit = gr.Button("Submit")
    submit.click(gradio_interface, inputs=[file_input, query_input], outputs=output)

demo.launch()
