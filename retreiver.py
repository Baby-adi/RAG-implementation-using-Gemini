from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json, os
import numpy as np

def save_store(store_path, docs, embeddings):
    data = []
    for doc in docs:
        vec = embeddings.embed_query(doc.page_content)
        data.append({"text": doc.page_content, "embedding": vec})
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_store(store_path):
    if not os.path.exists(store_path):
        return []
    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class SimpleRetriever:
    def __init__(self, store_path, embeddings):
        self.store_path = store_path
        self.embeddings = embeddings
        self.store = load_store(store_path)

    def add_documents(self, docs):
        save_store(self.store_path, docs, self.embeddings)
        self.store = load_store(self.store_path)

    def get_relevant_documents(self, query, top_k=3):
        if not self.store:
            return []
        query_vec = self.embeddings.embed_query(query)
        scored = [
            (cosine_similarity(query_vec, item["embedding"]), item["text"])
            for item in self.store
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [{"page_content": text} for _, text in scored[:top_k]]


def process_pdf(pdf_path, store_path="store.json"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = SimpleRetriever(store_path, embeddings)
    retriever.add_documents(chunks)
    return retriever
