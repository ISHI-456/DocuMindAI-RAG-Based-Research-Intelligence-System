 DocuMind AI: RAG-Based Research Intelligence system

 <p align="center"> <b>Ask questions across multiple research papers and get accurate, cited answers instantly.</b> </p> <p align="center"> <img src="https://img.shields.io/badge/Framework-LangChain-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/LLM-Groq%20(Qwen%2032B)-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/VectorDB-ChromaDB-green?style=for-the-badge"> <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge"> </p>

 ✨ Overview
DocuMind AI is a Retrieval-Augmented Generation (RAG) system that allows users to interact with multiple AI research papers like a chatbot.
🔧 A complete RAG pipeline is built from scratch (ingestion → retrieval → vector search)
🤖 Then Groq LLM is integrated on top for answer generation
📌 Provides accurate, context-grounded responses with citations

🔥 Features
📚 Multi-paper question answering
🔎 Semantic search using embeddings
🧠 Full RAG pipeline implementation
🤖 LLM-powered responses (Groq - Qwen 32B)
📌 Source citation (paper + page number)
⚡ Fast retrieval using pre-built vector database
🔗 Cross-paper reasoning

📄 Papers Indexed
Attention Is All You Need (2017)
BERT (2018)
Dense Passage Retrieval (2020)
FAISS (2017)
RAG Survey (2024)
LLaMA 2 (2023)
Self-RAG (2023)

🛠️ Tech Stack
Component	Technology
Framework	LangChain
LLM	Groq (Qwen 32B)
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector Store	ChromaDB
UI	Streamlit
PDF Parsing	PyPDFLoader

🏗️ Architecture
RAG Pipeline (Built First)
────────────────────────────────────
PDF Download → Chunking → Embeddings → ChromaDB

Query Pipeline (LLM Integration)
────────────────────────────────────
User Query → Embed → Retrieve → Groq LLM → Answer + Citations

⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/yourusername/documind-ai
cd documind-ai
2. Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate  

# Mac/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt
4. Add API Key

Create a .env file:

GROQ_API_KEY=your_key_here

5. Run Ingestion (One-Time)
python ingest.py
Downloads research papers
Splits into chunks (500 tokens, 50 overlap)
Generates embeddings
Stores data in ChromaDB

6. Run the App
streamlit run app.py

📂 Project Structure
DocuMind-AI/
├── app.py
├── ingest.py
├── requirements.txt
├── .env
├── .gitignore
├── data/
└── vector_store/

⚡ How It Works
1. RAG Pipeline (Built First)
Downloads PDFs from ArXiv
Splits text into chunks (500 tokens, 50 overlap)
Generates embeddings using sentence-transformers
Stores embeddings in ChromaDB
2. Retrieval
Converts user query into embedding
Performs cosine similarity search
Filters results using threshold (0.3)
3. LLM Integration
Retrieved context is passed to Groq LLM
Generates grounded answers
Displays citations (paper + page number)
   












