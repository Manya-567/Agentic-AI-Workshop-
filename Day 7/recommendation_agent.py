from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

class RecommendationAgent:
    def __init__(self):
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("rag_index", embeddings=embedding, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        self.chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            chain_type="stuff"
        )

    def recommend(self, query: str) -> str:
        return self.chain.run(query)
