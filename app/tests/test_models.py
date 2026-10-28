"""Persistence contract tests. Each test owns a fresh temporary SQLite file."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError

from business.database import (
    Base, PROJECT_ROOT, create_session_factory, create_sqlite_engine, get_database_path,
)
from business.models import Order, OrderStatus, Product, Ticket, TicketStatus, TicketType, User, UserRole


@pytest.fixture
def engine(tmp_path):
    engine = create_sqlite_engine(tmp_path / "test.sqlite3")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with create_session_factory(engine)() as session:
        yield session


@pytest.fixture
def order(session):
    user = User(auth_subject="test-customer", display_name="Test customer")
    product = Product(sku="TEST-SKU", name="Test product", description="Synthetic data", price_cents=1000)
    order = Order(
        order_no="TEST-ORDER", user=user, product=product,
        product_name_snapshot=product.name, sku_snapshot=product.sku,
        unit_price_cents=1000, quantity=2, total_amount_cents=2000,
        status=OrderStatus.COMPLETED,
    )
    session.add(order)
    session.commit()
    return order


def make_ticket(order, **changes):
    values = dict(
        ticket_no=f"TEST-{uuid4()}", user_id=order.user_id, order_id=order.id,
        type=TicketType.REPAIR_REQUEST, description="Synthetic repair request",
        idempotency_key=str(uuid4()), request_fingerprint="a" * 64,
    )
    values.update(changes)
    return Ticket(**values)


def assert_integrity_failure(session):
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_four_tables_and_foreign_keys_on_every_connection(engine):
    assert set(inspect(engine).get_table_names()) == {"users", "products", "orders", "tickets"}
    with engine.connect() as first, engine.connect() as second:
        assert first.scalar(text("PRAGMA foreign_keys")) == 1
        assert second.scalar(text("PRAGMA foreign_keys")) == 1


def test_defaults_and_relationships(session, order):
    ticket = make_ticket(order)
    session.add(ticket)
    session.commit()
    assert UUID(order.id)
    assert UUID(ticket.id)
    assert order.user.role is UserRole.CUSTOMER
    assert order.user.is_active
    assert order.product.currency == "CNY"
    assert ticket.status is TicketStatus.PENDING_REVIEW
    assert ticket.user is order.user
    assert ticket.order is order
    assert ticket in order.tickets
    assert ticket in order.user.tickets
    assert order in order.product.orders


@pytest.mark.parametrize("field", ["user_id", "product_id"])
def test_order_foreign_keys(session, order, field):
    setattr(order, field, "missing-id")
    assert_integrity_failure(session)


@pytest.mark.parametrize("field", ["user_id", "order_id", "reviewed_by"])
def test_ticket_foreign_keys(session, order, field):
    session.add(make_ticket(order, **{field: "missing-id"}))
    assert_integrity_failure(session)


def test_ticket_cannot_claim_another_orders_owner(session, order):
    other = User(auth_subject="other-test-customer", display_name="Other")
    session.add(other)
    session.commit()
    session.add(make_ticket(order, user_id=other.id))
    assert_integrity_failure(session)


def test_referenced_order_cannot_be_deleted(session, order):
    session.add(make_ticket(order))
    session.commit()
    with pytest.raises(IntegrityError):
        session.execute(text("DELETE FROM orders WHERE id = :id"), {"id": order.id})
    session.rollback()


def test_order_number_unique(session, order):
    duplicate = Order(
        order_no=order.order_no, user_id=order.user_id, product_id=order.product_id,
        product_name_snapshot="Test", sku_snapshot="TEST", unit_price_cents=1,
        quantity=1, total_amount_cents=1,
    )
    session.add(duplicate)
    assert_integrity_failure(session)


@pytest.mark.parametrize("model", [User, Product])
def test_identity_and_sku_unique(session, order, model):
    if model is User:
        duplicate = User(auth_subject=order.user.auth_subject, display_name="Duplicate")
    else:
        duplicate = Product(sku=order.product.sku, name="Duplicate", description="Test", price_cents=0)
    session.add(duplicate)
    assert_integrity_failure(session)


@pytest.mark.parametrize("changes", [
    {"unit_price_cents": -1}, {"quantity": 0}, {"quantity": -1},
    {"total_amount_cents": -1}, {"total_amount_cents": 999}, {"currency": "USD"},
    {"unit_price_cents": 1.5, "total_amount_cents": 3},
    {"quantity": 1.5, "total_amount_cents": 1500},
    {"unit_price_cents": 0.75, "total_amount_cents": 1.5},
    {"quantity": "not-a-number"},
])
def test_order_amount_quantity_and_currency_constraints(session, order, changes):
    for field, value in changes.items():
        setattr(order, field, value)
    assert_integrity_failure(session)


@pytest.mark.parametrize("changes", [
    {"price_cents": -1}, {"currency": "USD"},
    {"price_cents": 1.5}, {"price_cents": "not-a-number"},
])
def test_product_price_and_currency_constraints(session, order, changes):
    for field, value in changes.items():
        setattr(order.product, field, value)
    assert_integrity_failure(session)


def test_zero_price_allowed(session, order):
    order.product.price_cents = 0
    order.unit_price_cents = 0
    order.total_amount_cents = 0
    session.commit()


@pytest.mark.parametrize("table,column", [
    ("users", "display_name"), ("products", "description"),
    ("orders", "quantity"), ("tickets", "request_fingerprint"),
])
def test_non_null_constraints(session, order, table, column):
    session.add(make_ticket(order))
    session.commit()
    # Identifiers are fixed test constants, never user or model input.
    with pytest.raises(IntegrityError):
        session.execute(text(f"UPDATE {table} SET {column} = NULL"))
    session.rollback()


@pytest.mark.parametrize("table,column", [
    ("users", "role"), ("orders", "status"), ("tickets", "type"), ("tickets", "status"),
])
def test_invalid_enum_rejected_by_database(session, order, table, column):
    session.add(make_ticket(order))
    session.commit()
    with pytest.raises(IntegrityError):
        session.execute(text(f"UPDATE {table} SET {column} = 'invalid'"))
    session.rollback()


def test_ticket_number_unique(session, order):
    first = make_ticket(order, status=TicketStatus.CANCELLED)
    session.add(first)
    session.commit()
    session.add(make_ticket(order, ticket_no=first.ticket_no))
    assert_integrity_failure(session)


def test_idempotency_key_unique_even_after_terminal_state(session, order):
    first = make_ticket(order, status=TicketStatus.CANCELLED)
    session.add(first)
    session.commit()
    session.add(make_ticket(order, idempotency_key=first.idempotency_key))
    assert_integrity_failure(session)


def test_idempotency_key_scoped_to_user(session, order):
    session.add(make_ticket(order, idempotency_key="shared-request-key"))
    other_user = User(auth_subject="second-customer", display_name="Second customer")
    other_order = Order(
        order_no="SECOND-ORDER", user=other_user, product=order.product,
        product_name_snapshot="Test", sku_snapshot="TEST", unit_price_cents=1000,
        quantity=1, total_amount_cents=1000,
    )
    session.add(other_order)
    session.flush()
    session.add(make_ticket(other_order, idempotency_key="shared-request-key"))
    session.commit()


@pytest.mark.parametrize("first_status", [
    TicketStatus.PENDING_REVIEW, TicketStatus.APPROVED, TicketStatus.IN_PROGRESS,
])
@pytest.mark.parametrize("second_status", [
    TicketStatus.PENDING_REVIEW, TicketStatus.APPROVED, TicketStatus.IN_PROGRESS,
])
def test_active_ticket_unique(session, order, first_status, second_status):
    session.add(make_ticket(order, status=first_status))
    session.commit()
    session.add(make_ticket(order, status=second_status))
    assert_integrity_failure(session)


@pytest.mark.parametrize("terminal_status", [
    TicketStatus.RESOLVED, TicketStatus.REJECTED, TicketStatus.CANCELLED,
])
def test_terminal_ticket_allows_new_application(session, order, terminal_status):
    first = make_ticket(order)
    session.add(first)
    session.commit()
    first.status = terminal_status
    first.review_note = "Synthetic rejection explanation"
    first.resolution_note = "Synthetic resolution"
    session.commit()
    second = make_ticket(order)
    session.add(second)
    session.commit()
    assert second.id != first.id
    assert second.status is TicketStatus.PENDING_REVIEW


def test_different_ticket_types_can_coexist(session, order):
    session.add(make_ticket(order, type=TicketType.REPAIR_REQUEST))
    session.add(make_ticket(order, type=TicketType.EXCHANGE_REQUEST))
    session.commit()


def test_same_type_for_different_orders_can_coexist(session, order):
    another_order = Order(
        order_no="ANOTHER-ORDER", user=order.user, product=order.product,
        product_name_snapshot="Test", sku_snapshot="TEST", unit_price_cents=1000,
        quantity=1, total_amount_cents=1000,
    )
    session.add(another_order)
    session.flush()
    session.add(make_ticket(order))
    session.add(make_ticket(another_order))
    session.commit()


def test_reactivating_terminal_ticket_cannot_bypass_active_index(session, order):
    first = make_ticket(order, status=TicketStatus.CANCELLED)
    session.add(first)
    session.add(make_ticket(order))
    session.commit()
    first.status = TicketStatus.PENDING_REVIEW
    assert_integrity_failure(session)


@pytest.mark.parametrize("changes", [
    {"description": " "}, {"description": "x" * 2001}, {"idempotency_key": ""},
    {"request_fingerprint": " "}, {"status": TicketStatus.REJECTED},
    {"status": TicketStatus.RESOLVED},
])
def test_ticket_content_constraints(session, order, changes):
    session.add(make_ticket(order, **changes))
    assert_integrity_failure(session)


def test_time_roundtrip_normalizes_to_utc(session, order):
    local_time = datetime(2026, 10, 26, 16, 30, 0, 123456, tzinfo=timezone(timedelta(hours=8)))
    order.paid_at = local_time
    order.shipped_at = local_time
    order.completed_at = local_time
    ticket = make_ticket(order, reviewed_at=local_time)
    session.add(ticket)
    session.commit()
    session.expire_all()
    expected = local_time.astimezone(timezone.utc)
    assert order.paid_at == order.shipped_at == order.completed_at == ticket.reviewed_at == expected
    assert order.paid_at.tzinfo is timezone.utc
    for value in [
        order.created_at, order.updated_at, order.user.created_at,
        order.product.created_at, order.product.updated_at, ticket.created_at, ticket.updated_at,
    ]:
        assert value.tzinfo is timezone.utc
    stored = session.scalar(text("SELECT paid_at FROM orders WHERE id = :id"), {"id": order.id})
    assert stored == "2026-10-26 08:30:00.123456"


def test_naive_datetime_rejected(session, order):
    order.paid_at = datetime(2026, 10, 26, 8, 30)
    with pytest.raises(StatementError, match="explicit timezone"):
        session.commit()
    session.rollback()


def test_updated_at_changes_on_orm_update(session, order):
    old_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    order.updated_at = old_time
    session.commit()
    order.product_name_snapshot = "Changed snapshot"
    session.commit()
    assert order.updated_at > old_time
    assert order.updated_at.tzinfo is timezone.utc


def test_database_path_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVICEPILOT_DB_PATH", raising=False)
    assert get_database_path() == PROJECT_ROOT / "servicepilot.sqlite3"
    monkeypatch.setenv("SERVICEPILOT_DB_PATH", "local/test.sqlite3")
    assert get_database_path() == PROJECT_ROOT / "local" / "test.sqlite3"
    assert get_database_path(tmp_path / "explicit.db") == tmp_path / "explicit.db"


@pytest.mark.parametrize("path", ["", " ", "database.txt"])
def test_invalid_database_path_rejected(path):
    with pytest.raises(ValueError):
        get_database_path(path)


def test_session_factory_rolls_back_failed_transaction(engine):
    factory = create_session_factory(engine)
    with pytest.raises(RuntimeError):
        with factory.begin() as session:
            session.add(User(auth_subject="rolled-back-user", display_name="Test"))
            session.flush()
            raise RuntimeError("Synthetic failure")
    with factory() as session:
        assert session.scalar(select(User)) is None
