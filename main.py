import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from .graph import chatbot_graph
from .database import store_qa_pair, find_similar_past_qa


app = FastAPI(title="Enterprise Agent")

adapter_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"
class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str
    modified_amount: float = None

    
@app.get("/", response_class=HTMLResponse)
def get_ui():
    with open(UI_PATH, "r", encoding="utf-8") as f: return f.read()

@app.post("/chat")
def chat(payload: ChatRequest):
    onfig = {"configurable": {"thread_id": payload.thread_id}}
    is_tx = any(w in payload.message.lower() for w in ["credit", "refund", "dollar", "$"])
# Check Semantic Q&A Cache first
    match = None if is_tx else find_similar_past_qa(payload.message)
    if match:
        prompt = f"User asked: '{payload.message}'. Base answer: '{match['matched_answer']}'. Adapt naturally."