import os, time, uuid
from typing import Optional, Any
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_emb_fn = embedding_functions.DefaultEmbeddingFunction()

qa_collection = _client.get_or_create_collection(
    name="qa_knowledge_base",
    embedding_function=_emb_fn,
    metadata={"hnsw:space": "cosine"}
)

if qa_collection.count() == 0:
    seed_qa_pairs = [
        {"question": "What is your return policy?", "answer": "Returns within 30 days get 100% refund. After 30 days: store credit."},
        {"question": "What are your support hours?", "answer": "Support is active 9:00 AM to 6:00 PM EST, Monday-Friday."},
        {"question": "What are transfer limits?", "answer": "Credits over $100 require Human Manager approval."}
    ]
    qa_collection.add(
        documents=[item["question"] for item in seed_qa_pairs],
        ids=[f"seed_{i+1}" for i in range(len(seed_qa_pairs))],
        metadatas=[{"answer": item["answer"]} for item in seed_qa_pairs]
    )

def store_qa_pair(question: str, answer: str, thread_id: str) -> str:
    qa_id = f"qa_{uuid.uuid4().hex[:10]}"
    qa_collection.add(
        documents=[question],
        ids=[qa_id],
        metadatas=[{"answer": answer, "thread_id": thread_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}]
    )
    return qa_id

def find_similar_past_qa(query: str, similarity_threshold: float = 0.55) -> Optional[dict[str, Any]]:
    if qa_collection.count() == 0: return None
    res = qa_collection.query(query_texts=[query], n_results=1, include=["documents", "metadatas", "distances"])
    docs, metas, dists = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]
    if docs and dists and dists[0] <= similarity_threshold:
        return {"matched_question": docs[0], "matched_answer": metas[0].get("answer", ""), "distance": dists[0]}
    return None