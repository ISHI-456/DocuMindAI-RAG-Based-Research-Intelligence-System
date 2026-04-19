<div align="center">

# 🚀 DocuMind AI

### 🧠 RAG-Based Research Intelligence System

<p align="center">
  <b>Ask questions across multiple research papers and get accurate, cited answers instantly.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-Qwen%2032B-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

</div>

---

## ✨ Overview

**DocuMind AI** is a Retrieval-Augmented Generation (RAG) system that lets you interact with multiple AI research papers through a conversational interface.

- 🔧 Complete RAG pipeline built from scratch — ingestion → chunking → embedding → vector search
- 🤖 Groq LLM (Qwen 32B) integrated on top for intelligent answer generation
- 📌 Accurate, context-grounded responses with source citations

---

## 🔥 Features

| Feature | Description |
|---|---|
| 📚 Multi-paper QA | Ask questions spanning multiple research papers |
| 🔎 Semantic Search | Embedding-based similarity retrieval |
| 🧠 Full RAG Pipeline | End-to-end pipeline built from scratch |
| 🤖 LLM-Powered | Groq (Qwen 32B) for response generation |
| 📌 Source Citations | Pinpoints paper name + page number |
| ⚡ Fast Retrieval | Pre-built ChromaDB vector store |
| 🔗 Cross-paper Reasoning | Synthesizes insights across multiple documents |

---

## 📄 Papers Indexed

- 📝 Attention Is All You Need *(2017)*
- 📝 BERT *(2018)*
- 📝 Dense Passage Retrieval *(2020)*
- 📝 FAISS *(2017)*
- 📝 RAG Survey *(2024)*
- 📝 LLaMA 2 *(2023)*
- 📝 Self-RAG *(2023)*

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Groq (Qwen 32B) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| UI | Streamlit |
| PDF Parsing | PyPDFLoader |

---

## 🏗️ Architecture

```
RAG Pipeline (Built First)
────────────────────────────────────────────
PDF Download → Chunking → Embeddings → ChromaDB

Query Pipeline (LLM Integration)
────────────────────────────────────────────
User Query → Embed → Retrieve → Groq LLM → Answer + Citations
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/documind-ai
cd documind-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create a `.env` file in the root directory and add:

```env
GROQ_API_KEY=your_key_here
```

### 5. Run Ingestion *(One-Time Setup)*

```bash
python ingest.py
```

This step will:
- ✅ Download research papers from ArXiv
- ✅ Split documents into chunks (500 tokens, 50 overlap)
- ✅ Generate embeddings using `sentence-transformers`
- ✅ Store vectors in ChromaDB

> ⏱️ **Estimated time:** ~3–5 minutes

### 6. Launch the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
DocuMind-AI/
├── app.py              # Streamlit UI + query pipeline
├── ingest.py           # PDF ingestion + embedding pipeline
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
├── data/               # Downloaded research papers
└── vector_store/       # Persisted ChromaDB embeddings
```

---

## ⚡ How It Works

### 🔧 1. RAG Pipeline *(Built First)*
- Downloads PDFs from ArXiv
- Splits text into chunks (500 tokens, 50 overlap)
- Generates embeddings using `sentence-transformers`
- Stores embeddings in ChromaDB

### 🔎 2. Retrieval
- Converts user query into an embedding vector
- Performs cosine similarity search against the vector store
- Filters results using a relevance threshold (0.3)

### 🤖 3. LLM Integration
- Retrieved context is passed to Groq LLM
- Generates grounded, accurate answers
- Displays citations with paper name + page number

---

## 🎯 Key Highlight

> This project **first builds a complete RAG pipeline from scratch**, and then integrates **Groq LLM on top** for intelligent, citation-backed answer generation — making it a great reference for understanding RAG systems end-to-end.

---



## ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub — it helps a lot!
