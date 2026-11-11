from typing import Annotated, TypedDict, Optional, Any
from langgraph.graph.message import add_messages
import operator

class ChatbotState(TypedDict):
    messages: Annotated[list, add_messages]
    audit_log: Annotated[list[str], operator.add]
    sensitive_action_pending: bool
    pending_action_details: Optional[dict[str, Any]]