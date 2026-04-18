# DocuMind AI: RAG-Based Research Intelligence System
# Dependencies:
# pip install langchain langchain-community langchain-groq pypdf pymupdf
#             sentence-transformers chromadb streamlit python-dotenv

import os
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from dotenv import load_dotenv
from ingest import PAPERS
from langchain_groq import ChatGroq

st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("DocuMind AI")
st.caption("RAG-powered research assistant — ask questions across your uploaded papers")




# ─────────────────────────────────────────────
#  PDF Loading
# ─────────────────────────────────────────────
import urllib.request
import tempfile

def load_all_pdfs():
    all_docs = []

    for url in PAPERS:
        # extract a clean name from the URL for metadata
        paper_name = url.split("/")[-1]
        print(f"Downloading {paper_name}...")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            urllib.request.urlretrieve(url, tmp_path)
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = paper_name

            all_docs.extend(docs)
            print(f"  loaded {len(docs)} pages from {paper_name}")

        except Exception as e:
            print(f"  failed to load {paper_name}: {e}")

        finally:
            os.remove(tmp_path)

    print(f"Total pages loaded: {len(all_docs)}")
    return all_docs


# ─────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ─────────────────────────────────────────────
#  Embedding Manager
# ─────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings


# ─────────────────────────────────────────────
#  Vector Store Manager
# ─────────────────────────────────────────────
class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG vector store for research papers"}
        )
        print(f"Vector store ready. Documents in collection: {self.collection.count()}")

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Mismatch between number of documents and embeddings.")

        ids = []
        metadatas = []
        contents = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4()}")

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
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
# RAG Retriever
# ─────────────────────────────────────────────
class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.3):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            for i, (doc_id, metadata, document, distance) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0]
            )):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "document": document,
                        "metadata": metadata,
                        "similarity_score": round(similarity_score, 4),
                        "rank": i + 1
                    })

        print(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        return retrieved_docs


@st.cache_resource
def initialize_pipeline():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStoreManager()

    # Only ingest if the collection is empty — skips re-ingestion on restarts
    if vector_store.collection.count() == 0:
        print("Collection empty — running ingestion pipeline...")
        docs = load_all_pdfs()
        chunks = split_docs(docs)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
    else:
        print("Collection already populated — skipping ingestion.")

    return RAGRetriever(embedding_manager, vector_store)


# ─────────────────────────────────────────────
# 7. LLM — cached
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return ChatGroq(
        groq_api_key=api_key,
        model="qwen/qwen3-32b",
        temperature=0.1,
        max_tokens=1024
    )


# ─────────────────────────────────────────────
# 8. Answer Generation
# ─────────────────────────────────────────────
def generate_answer(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return "No relevant content found in the documents for this question.", []

    context = "\n\n".join([doc["document"] for doc in results])

    prompt = f"""You are an AI research assistant. Answer the question using ONLY the provided context.
If the answer is not present in the context, say "This information is not found in the provided documents."
Give a clear, concise, and accurate answer.

Context:
{context}

Question:
{query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, results


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
rag_retriever = initialize_pipeline()
llm = get_llm()

query = st.text_input("Ask a question about your research papers")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            answer, sources = generate_answer(query, rag_retriever, llm, top_k=3)

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for doc in sources:
                source_file = os.path.basename(doc["metadata"].get("source", "Unknown file"))
                page = doc["metadata"].get("page", "N/A")
                score = doc["similarity_score"]

                with st.expander(f"{source_file} — Page {page} — Relevance: {score}"):
                    st.caption(doc["document"][:500] + ("..." if len(doc["document"]) > 500 else ""))
