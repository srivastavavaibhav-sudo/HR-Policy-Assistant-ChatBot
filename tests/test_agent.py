from fastapi.testclient import TestClient

from app.main import app
from app.agents.tools import calculate_leave_balance

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_calculate_leave_balance():
    result = calculate_leave_balance.invoke({"months_worked": 6, "leave_days_taken": 3})
    assert "Remaining balance: 6.0 days" in result


def test_calculate_leave_balance_caps_at_18():
    result = calculate_leave_balance.invoke({"months_worked": 24, "leave_days_taken": 0})
    assert "Accrued leave: 18" in result
