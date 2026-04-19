# DocuMind AI: RAG-Based Research Intelligence System
# Dependencies:
# pip install langchain langchain-community langchain-groq pypdf pymupdf
#             sentence-transformers chromadb streamlit python-dotenv



import os
import streamlit as st
import urllib.request
import tempfile
import uuid

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from ingest import PAPERS
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# ENV FIXES (IMPORTANT)
# ─────────────────────────────────────────────
os.environ["TRANSFORMERS_NO_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ─────────────────────────────────────────────
# STREAMLIT UI CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DocuMind AI", layout="wide")

st.title("📄 DocuMind AI")
st.caption("RAG-powered research assistant — ask questions across AI research papers with citations")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR UI
# ─────────────────────────────────────────────
st.sidebar.title("📚 Papers Indexed")

paper_names = [
    "Attention Is All You Need (2017)",
    "BERT (2018)",
    "Dense Passage Retrieval (2020)",
    "FAISS (2017)",
    "RAG Survey (2024)",
    "LLaMA 2 (2023)",
    "Self-RAG (2023)"
]

for paper in paper_names:
    st.sidebar.markdown(f"• {paper}")

st.sidebar.markdown("---")
st.sidebar.info("💡 Ask questions across multiple papers and get cited answers.")

# ─────────────────────────────────────────────
# PDF Loading
# ─────────────────────────────────────────────
def load_all_pdfs():
    all_docs = []

    for url in PAPERS:
        paper_name = url.split("/")[-1]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            urllib.request.urlretrieve(url, tmp_path)
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = paper_name

            all_docs.extend(docs)

        except Exception as e:
            print(f"Failed to load {paper_name}: {e}")

        finally:
            os.remove(tmp_path)

    return all_docs


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────
# Embedding Manager
# ─────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts)


# ─────────────────────────────────────────────
# Vector Store Manager
# ─────────────────────────────────────────────
class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="pdf_documents")

    def add_documents(self, documents, embeddings):
        ids = []
        metadatas = []
        contents = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4()}")
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i

            metadatas.append(metadata)
            contents.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=contents,
            embeddings=embeddings_list
        )


# ─────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────
class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, threshold=0.3):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        for doc_id, metadata, document, distance in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        ):
            score = 1 - distance
            if score >= threshold:
                retrieved_docs.append({
                    "document": document,
                    "metadata": metadata,
                    "score": round(score, 3)
                })

        return retrieved_docs


# ─────────────────────────────────────────────
# Initialize Pipeline
# ─────────────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStoreManager()

    if vector_store.collection.count() == 0:
        docs = load_all_pdfs()
        chunks = split_docs(docs)
        texts = [c.page_content for c in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)

    return RAGRetriever(embedding_manager, vector_store)


# ─────────────────────────────────────────────
# LLM Setup (FIXED)
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()

    return ChatGroq(
        groq_api_key=api_key,
        model="qwen/qwen3-32b",
        temperature=0.1
    )


# ─────────────────────────────────────────────
# Answer Generation
# ─────────────────────────────────────────────
def generate_answer(query, retriever, llm):
    docs = retriever.retrieve(query)

    if not docs:
        return "No relevant information found.", []

    context = "\n\n".join([d["document"] for d in docs])

    prompt = f"""
Answer the question using ONLY the context below.
If not found, say so clearly.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content, docs


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
retriever = initialize_pipeline()
llm = get_llm()

query = st.text_input("🔍 Ask a question about the research papers")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(query, retriever, llm)

        st.subheader("🧠 Answer")
        st.write(answer)

        if sources:
            st.subheader("📄 Sources")

            for doc in sources:
                src = doc["metadata"].get("source", "Unknown")
                page = doc["metadata"].get("page", "N/A")

                st.markdown(f"**📘 {src} | Page {page} | Score: {doc['score']}**")

                with st.expander("View Content"):
                    st.write(doc["document"][:500])

                st.markdown("---")
