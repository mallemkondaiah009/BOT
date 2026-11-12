from langchain_core.tools import tool

@tool
def execute_account_credit(user_id: str, amount: float, reason: str) -> str:
    """Issues an account credit or refund to a user. Credits > $100 require approval."""
    if amount > 100.0:
        return f"ALERT: Amount ${amount:.2f} exceeds $100. Placed on HOLD for supervisor approval."
    return f"SUCCESS: Credit of ${amount:.2f} applied to {user_id}."

ALL_TOOLS = [execute_account_credit]