# 🧠 RAG Pipeline with Hybrid Search, Reranking, and LLMs  
> End-to-End Retrieval-Augmented Generation (RAG) System with ChromaDB, BM25, and Groq’s LLaMA Models and Langchain  
> Currently deployed on Hugging Face Spaces 🚀 | Planned migration to AWS/GCP ☁️

---

## 📌 Overview  
This project is a **complete Retrieval-Augmented Generation (RAG)** pipeline that combines **vector search**, **keyword-based search (BM25)**, and **reranking** from **Langchain** to retrieve the most relevant context from documents before querying a **Large Language Model (LLM)** for precise answers.  

We built this with **Langchain**:  
- **PDF & DOCX ingestion**  
- **Chunking and vector embedding storage (ChromaDB)**  
- **Hybrid retrieval (BM25 + vector search)**  
- **Reranking of candidate results**  
- **Query answering via Groq LLaMA-3 API**  
- **Simple Gradio UI for interaction**  

💡 Currently deployed on **Hugging Face Spaces**, with plans to scale to **AWS or GCP** for production workloads.

---

## 🔥 Key Features
- 📂 **Document Parsing**: Extracts and preprocesses text from both `.pdf` and `.docx` files.  
- 🔍 **Hybrid Search**: Combines dense vector embeddings with sparse keyword-based retrieval (BM25).  
- 🏆 **Reranking**: Selects the top-k most relevant document chunks for higher accuracy.  
- 🤖 **LLM Querying**: Uses Groq API with `llama-3.1-8b-instant` for final answer generation.  
- 🎛️ **Gradio UI**: User-friendly interface for uploading documents and asking questions.  
- ☁️ **Scalable Design**: Architecture ready for deployment on AWS Lambda, ECS Fargate, or GCP Vertex AI.

---

## 🛠️ Tech Stack

| Layer                | Technology                                                                 |
|----------------------|---------------------------------------------------------------------------|
| **LLM Inference**    | Groq API (LLaMA-3.1-8B-Instant)                                           |
| **Vector DB**        | [ChromaDB](https://www.trychroma.com/)                                   |
| **Sparse Retrieval** | [rank-bm25](https://pypi.org/project/rank-bm25/)                         |
| **Document Parsing** | PyMuPDF (`fitz`), python-docx                                             |
| **UI**               | [Gradio](https://gradio.app/)                                            |
| **Deployment**       | [Hugging Face Spaces (current)](https://huggingface.co/spaces/SKB3002/Advanced-RAG-with-Hybrid-Search-and-Reranking), AWS/GCP (planned)                         |
| **Language**         | Python 3.11                                                              |

---

## ⚙️ Architecture

flowchart TD
    A[Upload PDF/DOCX] --> B[Text Extraction & Chunking]
    B --> C[ChromaDB - Vector Store]
    B --> D[BM25 Keyword Index]
    C --> E[Vector Retrieval]
    D --> E
    E --> F[Reranking]
    F --> G[LLM Query - Groq API]
    G --> H[Answer Display in Gradio]

---

## 🗺️ Roadmap
PDF & DOCX parsing
Hybrid Search (BM25 + ChromaDB)
LLaMA Querying via Groq API
Gradio UI Deployment on Hugging Face Spaces
### Future Plannings
CI/CD with GitHub Actions
AWS Lambda/ECS or GCP Vertex AI Deployment

---

## 📜 License
This project is licensed under the MIT License.

---

## 👤 Author
**Suyash Bhatkar**
AI Engineer in Progress 🚀 | Machine Learning & Data Science Enthusiast
 
