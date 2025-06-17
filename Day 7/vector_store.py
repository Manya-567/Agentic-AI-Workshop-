# rag/vector_store.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vector_store():
    loader = PyPDFLoader("data/communication_guide.pdf")
    pages = loader.load_and_split()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(pages, embeddings)
    db.save_local("rag_index")

if __name__ == "__main__":
    build_vector_store()
