from langchain_core.tools import tool

from app.rag.retriever import retrieve_policy_chunks


@tool
def search_hr_policy(query: str) -> str:
    """Search the company's HR policy documents (leave, WFH, maternity/paternity,
    equipment, etc.) and return the most relevant policy excerpts for the query.
    Use this whenever the user asks about a rule, entitlement, or process defined
    in HR policy."""
    chunks = retrieve_policy_chunks(query, k=3)
    if not chunks:
        return "No relevant policy information found."
    formatted = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )
    return formatted


@tool
def calculate_leave_balance(months_worked: int, leave_days_taken: int) -> str:
    """Calculate an employee's remaining annual leave balance.
    Annual leave accrues at 1.5 days per completed month of service (per policy),
    capped at 18 days/year. Pass the number of completed months worked this year
    and the number of leave days already taken."""
    accrued = min(months_worked * 1.5, 18)
    remaining = round(accrued - leave_days_taken, 1)
    return (
        f"Accrued leave: {accrued} days. "
        f"Taken: {leave_days_taken} days. "
        f"Remaining balance: {remaining} days."
    )


TOOLS = [search_hr_policy, calculate_leave_balance]
