"""Order authorization and response contracts using synthetic, temporary data."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import OperationalError

from business.database import Base, create_session_factory, create_sqlite_engine
from business.models import Order, OrderStatus, Product, User, UserRole
from business.repositories import OrderRepository
from business.services import OrderService


@pytest.fixture
def database(tmp_path):
    engine = create_sqlite_engine(tmp_path / "order-service.sqlite3")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    local_time = datetime(2026, 10, 26, 16, 30, tzinfo=timezone(timedelta(hours=8)))
    with factory.begin() as session:
        owner = User(auth_subject="synthetic-owner-auth", display_name="Synthetic owner")
        other = User(auth_subject="synthetic-other-private-auth", display_name="Synthetic private user")
        inactive = User(auth_subject="synthetic-disabled-auth", display_name="Disabled", is_active=False)
        staff = User(auth_subject="synthetic-staff-auth", display_name="Staff", role=UserRole.STAFF)
        product = Product(
            sku="CURRENT-SKU", name="Current catalog name", description="Synthetic product",
            price_cents=9999,
        )
        session.add_all([owner, other, inactive, staff, product])
        session.flush()
        ids = {"owner": owner.id, "other": other.id, "inactive": inactive.id, "staff": staff.id}
        for status in OrderStatus:
            session.add(Order(
                order_no=f"OWN-{status.value}", user_id=owner.id, product_id=product.id,
                product_name_snapshot="Original product name", sku_snapshot="ORIGINAL-SKU",
                unit_price_cents=2999, quantity=2, total_amount_cents=5998,
                status=status, created_at=local_time,
                paid_at=local_time if status in (OrderStatus.PAID, OrderStatus.SHIPPED, OrderStatus.COMPLETED) else None,
                shipped_at=local_time if status in (OrderStatus.SHIPPED, OrderStatus.COMPLETED) else None,
                completed_at=local_time if status == OrderStatus.COMPLETED else None,
            ))
        for user_id, order_no in [(other.id, "OTHER-ORDER"), (inactive.id, "DISABLED-ORDER")]:
            session.add(Order(
                order_no=order_no, user_id=user_id, product_id=product.id,
                product_name_snapshot="Private order snapshot", sku_snapshot="PRIVATE-SKU",
                unit_price_cents=1234, quantity=1, total_amount_cents=1234,
            ))
    yield engine, factory, ids
    engine.dispose()


@pytest.fixture
def service(database):
    return OrderService(database[1])


def test_own_order_success_snapshot_money_and_times(service, database):
    response = service.get_order(current_user_id=database[2]["owner"], order_no="OWN-completed")
    assert response["ok"] is True
    assert response["error"] is None
    data = response["data"]
    assert data["order_no"] == "OWN-completed"
    assert data["status"] == "completed"
    assert data["total_amount_cents"] == 5998
    assert data["currency"] == "CNY"
    assert data["product"]["name"] == "Original product name"
    assert data["product"]["sku"] == "ORIGINAL-SKU"
    assert data["product"]["unit_price_cents"] == 2999
    assert data["product"]["quantity"] == 2
    assert data["product"]["product_id"]
    for field in ("created_at", "paid_at", "shipped_at", "completed_at"):
        assert data[field] == "2026-10-26T08:30:00+00:00"
    assert json.loads(json.dumps(response)) == response


def test_other_order_and_missing_order_have_identical_responses(service, database):
    other = service.get_order(current_user_id=database[2]["owner"], order_no="OTHER-ORDER")
    missing = service.get_order(current_user_id=database[2]["owner"], order_no="DOES-NOT-EXIST")
    assert other == missing == {
        "ok": False, "data": None,
        "error": {"code": "ORDER_NOT_FOUND", "message": "未找到该订单。", "retryable": False},
    }


@pytest.mark.parametrize("identity", ["missing-user", "inactive"])
def test_missing_and_disabled_users_rejected_before_order_query(service, database, monkeypatch, identity):
    query = Mock(side_effect=AssertionError("Order lookup must not execute"))
    monkeypatch.setattr(OrderRepository, "get_by_order_no_for_user", query)
    user_id = database[2].get(identity, identity)
    response = service.get_order(current_user_id=user_id, order_no="DISABLED-ORDER")
    assert response["ok"] is False
    assert response["data"] is None
    assert response["error"]["code"] == "UNAUTHENTICATED"
    assert response["error"]["retryable"] is False
    query.assert_not_called()


@pytest.mark.parametrize("identity", [None, "", " ", 123, True, {}, "x" * 37, " user "])
def test_malformed_identity_rejected_without_database(identity):
    factory = Mock(side_effect=AssertionError("Must not open database"))
    response = OrderService(factory).get_order(current_user_id=identity, order_no="OWN-completed")
    assert response["error"]["code"] == "UNAUTHENTICATED"
    factory.assert_not_called()


@pytest.mark.parametrize("order_no", [
    None, "", " ", 123, True, {}, [], "x" * 65,
    "OWN completed", " OWN-completed", "OWN-completed ", "OWN-completed\n",
    "OWN/completed", "订单001", "_ORDER", "-ORDER", "ORDER;DROP", "' OR 1=1 --",
])
def test_invalid_order_number_rejected_without_database(order_no):
    factory = Mock(side_effect=AssertionError("Must not open database"))
    response = OrderService(factory).get_order(current_user_id="trusted-test-user", order_no=order_no)
    assert response == {
        "ok": False, "data": None,
        "error": {
            "code": "INVALID_ARGUMENT",
            "message": "订单号须为1–64位字母、数字、下划线或连字符，且以字母或数字开头。",
            "retryable": False,
        },
    }
    factory.assert_not_called()


@pytest.mark.parametrize("order_no", ["A", "9", "a" * 64, "Test_001-ABC"])
def test_valid_order_number_boundaries(service, database, order_no):
    response = service.get_order(current_user_id=database[2]["owner"], order_no=order_no)
    assert response["error"]["code"] == "ORDER_NOT_FOUND"


@pytest.mark.parametrize("status,expected", [
    (OrderStatus.PENDING_PAYMENT, []),
    (OrderStatus.PAID, ["return_request"]),
    (OrderStatus.SHIPPED, ["return_request", "exchange_request", "repair_request"]),
    (OrderStatus.COMPLETED, ["return_request", "exchange_request", "repair_request"]),
    (OrderStatus.CANCELLED, []),
])
def test_available_ticket_types(service, database, status, expected):
    response = service.get_order(current_user_id=database[2]["owner"], order_no=f"OWN-{status.value}")
    assert response["ok"] is True
    assert response["data"]["available_ticket_types"] == expected


def test_unpaid_order_returns_null_optional_times(service, database):
    data = service.get_order(
        current_user_id=database[2]["owner"], order_no="OWN-pending_payment"
    )["data"]
    assert data["paid_at"] is data["shipped_at"] is data["completed_at"] is None


def test_response_field_allowlist_and_no_private_data(service, database):
    response = service.get_order(current_user_id=database[2]["owner"], order_no="OWN-completed")
    assert set(response) == {"ok", "data", "error"}
    assert set(response["data"]) == {
        "order_no", "status", "product", "total_amount_cents", "currency",
        "created_at", "paid_at", "shipped_at", "completed_at", "available_ticket_types",
    }
    assert set(response["data"]["product"]) == {"product_id", "name", "sku", "unit_price_cents", "quantity"}
    serialized = json.dumps(response)
    for forbidden in [
        "user_id", "auth_subject", "display_name", "reviewed_by", "_sa_instance_state",
        "Private order snapshot", "PRIVATE-SKU", *database[2].values(),
    ]:
        assert forbidden not in serialized


def test_staff_has_no_ownership_bypass(service, database):
    response = service.get_order(current_user_id=database[2]["staff"], order_no="OTHER-ORDER")
    assert response["error"]["code"] == "ORDER_NOT_FOUND"


def test_repository_filters_both_order_and_user(database):
    _, factory, ids = database
    with factory() as session:
        repository = OrderRepository(session)
        assert repository.get_by_order_no_for_user(order_no="OTHER-ORDER", current_user_id=ids["owner"]) is None
        assert repository.get_by_order_no_for_user(order_no="OTHER-ORDER", current_user_id=ids["other"]) is not None
        assert repository.get_by_order_no_for_user(order_no="MISSING", current_user_id=ids["other"]) is None


def test_repository_uses_bound_parameters_for_injection_payload(database):
    _, factory, ids = database
    with factory() as session:
        repository = OrderRepository(session)
        assert repository.get_by_order_no_for_user(
            order_no="' OR 1=1 --", current_user_id=ids["owner"]
        ) is None
        assert session.scalar(select(func.count()).select_from(Order)) == 7


@pytest.mark.parametrize("method", ["get_user", "get_by_order_no_for_user"])
def test_database_errors_are_safe(service, database, monkeypatch, caplog, method):
    sensitive = "SYNTHETIC_PRIVATE_SQL_OR_PATH"
    failure = Mock(side_effect=OperationalError(sensitive, {"user_id": sensitive}, Exception(sensitive)))
    monkeypatch.setattr(OrderRepository, method, failure)
    response = service.get_order(current_user_id=database[2]["owner"], order_no="OWN-completed")
    assert response == {
        "ok": False, "data": None,
        "error": {"code": "TEMPORARY_UNAVAILABLE", "message": "订单查询暂时不可用，请稍后重试。", "retryable": True},
    }
    assert sensitive not in json.dumps(response)
    assert sensitive not in caplog.text


def test_real_database_error_is_safe(tmp_path):
    engine = create_sqlite_engine(tmp_path / "empty.sqlite3")
    try:
        response = OrderService(create_session_factory(engine)).get_order(
            current_user_id="synthetic-user", order_no="TEST-ORDER"
        )
        assert response["error"]["code"] == "TEMPORARY_UNAVAILABLE"
        assert response["error"]["retryable"] is True
        assert "SELECT" not in json.dumps(response)
        assert str(tmp_path) not in json.dumps(response)
    finally:
        engine.dispose()


def test_connection_creation_error_is_safe():
    factory = Mock(side_effect=OperationalError("private connection detail", {}, Exception("private")))
    response = OrderService(factory).get_order(current_user_id="synthetic-user", order_no="TEST-ORDER")
    assert response["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert "private" not in json.dumps(response)


def test_service_only_reads_and_closes_session(service, database):
    engine, _, ids = database
    statements = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = service.get_order(current_user_id=ids["owner"], order_no="OWN-completed")
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert response["ok"] is True
    assert len(statements) == 2
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
    assert engine.pool.checkedout() == 0
