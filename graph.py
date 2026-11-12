import os
import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from state import ChatbotState
from tools import ALL_TOOLS

# 1. Load environment variables & initialize LLM
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = SystemMessage(
    content="You are a customer assistant. For credits/refunds, call execute_account_credit."
)

# 2. Node: Agent Reasoning Node
def agent_node(state: ChatbotState) -> dict:
    """Invokes the LLM with full context and intercepts sensitive tool calls."""
    resp = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    sensitive = False
    details = None

    if hasattr(resp, "tool_calls") and resp.tool_calls:
        for tc in resp.tool_calls:
            # Financial guardrail: any credit exceeding $100 requires supervisor approval
            if tc["name"] == "execute_account_credit" and tc.get("args", {}).get("amount", 0) > 100:
                sensitive = True
                details = {"tool": "execute_account_credit", "args": tc["args"]}

    return {
        "messages": [resp],
        "audit_log": [f"agent: turn done (tool_calls={bool(resp.tool_calls)})"],
        "sensitive_action_pending": sensitive,
        "pending_action_details": details
    }

# 3. Node: Human-in-the-Loop Approval Gate Node
def sensitive_action_gate_node(state: ChatbotState) -> dict:
    """Executes only when a human supervisor approves the pending action."""
    args = (state.get("pending_action_details") or {}).get("args", {})
    amount = args.get("amount", 0)
    return {
        "messages": [AIMessage(content=f"Human Supervisor APPROVED credit of ${amount:.2f}.")],
        "audit_log": ["supervisor: transaction settled"],
        "sensitive_action_pending": False,
        "pending_action_details": None
    }

# 4. Conditional Edge Router
def custom_router(state: ChatbotState) -> str:
    """Evaluates state to route to HITL gate, regular tool execution, or termination."""
    if state.get("sensitive_action_pending"):
        return "sensitive_action_gate"
    
    last = state["messages"][-1]
    return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

# 5. Build Graph Structure
builder = StateGraph(ChatbotState)

# Add Nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools=ALL_TOOLS))
builder.add_node("sensitive_action_gate", sensitive_action_gate_node)

# Add Edges
builder.add_edge(START, "agent")

builder.add_conditional_edges("agent", custom_router, {
    "tools": "tools",
    "sensitive_action_gate": "sensitive_action_gate",
    END: END
})

# ReAct feedback cycle: after tools run, return output to agent
builder.add_edge("tools", "agent")

# Termination edge: after supervisor approval, exit the workflow
builder.add_edge("sensitive_action_gate", END)

# 6. Checkpoint Persistence & HITL Compilation
# Resolve project data folder safely whether in root or app/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "app" else CURRENT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

db_conn = sqlite3.connect(os.path.join(DATA_DIR, "chat_checkpoints.db"), check_same_thread=False)

# Compile graph with SQLite checkpointing and human-in-the-loop pause
chatbot_graph = builder.compile(
    checkpointer=SqliteSaver(db_conn),
    interrupt_before=["sensitive_action_gate"]
)
