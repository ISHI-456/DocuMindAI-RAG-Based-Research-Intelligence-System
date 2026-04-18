 DocuMind AI — RAG-Based Research Assistant

DocuMind AI is a Retrieval-Augmented Generation (RAG) system that enables users to query multiple AI research papers and receive accurate, context-grounded answers with citations.

It combines semantic search + LLM reasoning to act as an intelligent research assistant.

 Features

🔍 Query across multiple research papers
🧠 Semantic search using embeddings
🤖 Context-aware answers using LLM
📌 Source citations (paper + page number)
⚡ Fast responses using pre-built vector index
🔗 Supports cross-paper reasoning

📄 Papers Indexed
Attention Is All You Need (Vaswani et al., 2017)
BERT (Devlin et al., 2018)
Dense Passage Retrieval (Karpukhin et al., 2020)
FAISS (Johnson et al., 2017)
RAG Survey (Gao et al., 2024)
LLaMA 2 (Touvron et al., 2023)
Self-RAG (Asai et al., 2023)

🛠️ Tech Stack
Component	Technology
Framework	LangChain
LLM	Groq (Qwen 32B)
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector Store	ChromaDB
UI	Streamlit
PDF Parsing	PyPDFLoader

🏗️ Architecture
ingest.py          →   app.py
─────────────────      ──────────────────
Download PDFs          Load vector store
Chunk text             Embed user query
Generate embeddings    Search ChromaDB
Store in ChromaDB      Send to Groq LLM
                       Display answer + sources
                       
⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/documind-ai
cd documind-ai
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
3. Install Dependencies
pip install -r requirements.txt

4. Add Groq API Key

Create a .env file:

GROQ_API_KEY=your_key_here

Get your key: https://console.groq.com

5. Run Ingestion (One-Time Step)
python ingest.py

This step:

Downloads research papers
Splits text into chunks (500 tokens, 50 overlap)
Generates embeddings
Stores them in ChromaDB

⏱️ Takes ~3–5 minutes

6. Run the Application
streamlit run app.py

📂 Project Structure
DocuMind-AI/
│
├── app.py              # Streamlit UI + query pipeline
├── ingest.py           # PDF ingestion + embedding pipeline
├── requirements.txt
├── .env                # API keys (not committed)
├── .gitignore
│
├── data/               # Downloaded PDFs
└── vector_store/       # ChromaDB index

⚡ How It Works
1. Ingestion
Downloads PDFs from ArXiv
Splits into chunks (500 tokens, 50 overlap)
Generates embeddings using sentence-transformers
Stores embeddings in ChromaDB
2. Retrieval
User query → embedding
ChromaDB performs cosine similarity search
Filters results using score threshold (0.3)
3. Generation
Retrieved chunks passed to Groq LLM
Uses a strict prompt to ensure grounded answers
Returns answer with source citations
