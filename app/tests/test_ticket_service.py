"""Confirmed ticket creation, durable idempotency, rollback and SQLite races."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from business.database import Base, create_session_factory, create_sqlite_engine
from business.models import Order, OrderStatus, Product, Ticket, TicketStatus, TicketType, User
from business.repositories import TicketRepository
from business.services import OrderService, TicketService


@pytest.fixture
def db(tmp_path):
    engine = create_sqlite_engine(tmp_path / "tickets.sqlite3")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        owner = User(auth_subject="synthetic-owner", display_name="Synthetic owner")
        other = User(auth_subject="synthetic-private-other", display_name="Synthetic other")
        disabled = User(auth_subject="synthetic-disabled", display_name="Disabled", is_active=False)
        product = Product(sku="SYNTHETIC-SKU", name="Synthetic product", description="Test only", price_cents=100)
        session.add_all([owner, other, disabled, product])
        session.flush()
        ids = dict(owner=owner.id, other=other.id, disabled=disabled.id)
        for status in OrderStatus:
            session.add(Order(
                order_no=f"OWN-{status.value}", user_id=owner.id, product_id=product.id,
                product_name_snapshot=product.name, sku_snapshot=product.sku,
                unit_price_cents=100, quantity=1, total_amount_cents=100, status=status,
            ))
        for user, number in [(other, "OTHER-ORDER"), (disabled, "DISABLED-ORDER")]:
            session.add(Order(
                order_no=number, user_id=user.id, product_id=product.id,
                product_name_snapshot=product.name, sku_snapshot=product.sku,
                unit_price_cents=100, quantity=1, total_amount_cents=100, status=OrderStatus.COMPLETED,
            ))
    yield SimpleNamespace(engine=engine, factory=factory, ids=ids, service=TicketService(factory))
    engine.dispose()


def request(db, **overrides):
    args = dict(
        current_user_id=db.ids["owner"], order_no="OWN-paid", ticket_type="return_request",
        description="Synthetic issue", idempotency_key="request-001", confirmed=True,
    )
    args.update(overrides)
    return db.service.create_ticket(**args)


def count(db):
    with db.factory() as session:
        return session.scalar(select(func.count()).select_from(Ticket))


def test_create_return_response_and_persisted_values(db):
    result = request(db, description="  Synthetic issue  ")
    assert result["ok"] is True and result["error"] is None
    data = result["data"]
    assert set(data) == {"ticket_no", "order_no", "type", "status", "created_at", "created"}
    assert data["ticket_no"].startswith("SPT-")
    assert data["order_no"] == "OWN-paid"
    assert data["type"] == "return_request"
    assert data["status"] == "pending_review"
    assert data["created"] is True
    assert datetime.fromisoformat(data["created_at"]).tzinfo == timezone.utc
    with db.factory() as session:
        ticket = session.scalar(select(Ticket))
        assert ticket.user_id == db.ids["owner"]
        assert ticket.description == "Synthetic issue"
        assert ticket.status is TicketStatus.PENDING_REVIEW
        assert len(ticket.request_fingerprint) == 64
        assert ticket.reviewed_by is ticket.reviewed_at is None
        assert session.scalar(select(Order).where(Order.order_no == "OWN-paid")).status is OrderStatus.PAID
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("status", list(OrderStatus))
@pytest.mark.parametrize("kind", list(TicketType))
def test_eligibility_matches_order_query_rules(db, status, kind):
    number = f"OWN-{status.value}"
    allowed = OrderService(db.factory).get_order(current_user_id=db.ids["owner"], order_no=number)["data"]["available_ticket_types"]
    result = request(db, order_no=number, ticket_type=kind)
    if kind.value in allowed:
        assert result["ok"]
        assert count(db) == 1
    else:
        assert result["error"]["code"] == "TICKET_TYPE_NOT_ALLOWED"
        assert count(db) == 0


@pytest.mark.parametrize("changes,code", [
    ({"order_no": "OTHER-ORDER"}, "ORDER_NOT_FOUND"),
    ({"order_no": "MISSING-ORDER"}, "ORDER_NOT_FOUND"),
    ({"current_user_id": "missing-user"}, "UNAUTHENTICATED"),
    ({"current_user_id": None}, "UNAUTHENTICATED"),
    ({"current_user_id": " user "}, "UNAUTHENTICATED"),
    ({"current_user_id": 123}, "UNAUTHENTICATED"),
    ({"current_user_id": "x" * 37}, "UNAUTHENTICATED"),
])
def test_ownership_and_identity_failures(db, changes, code):
    result = request(db, **changes)
    assert result["error"]["code"] == code
    assert result["data"] is None
    assert count(db) == 0


def test_disabled_user_rejected(db):
    result = request(db, current_user_id=db.ids["disabled"], order_no="DISABLED-ORDER")
    assert result["error"]["code"] == "UNAUTHENTICATED"
    assert count(db) == 0


@pytest.mark.parametrize("confirmed", [False, None, 1, 0, "true", [], {}])
def test_no_confirmation_means_no_database_access(db, confirmed):
    factory = Mock(side_effect=AssertionError("Unconfirmed request must not access database"))
    db.service = TicketService(factory)
    result = request(db, confirmed=confirmed)
    assert result["error"]["code"] == "CONFIRMATION_REQUIRED"
    factory.assert_not_called()
    assert count(db) == 0


@pytest.mark.parametrize("changes", [
    {"ticket_type": "refund"}, {"ticket_type": ""}, {"ticket_type": None}, {"ticket_type": 1},
    {"description": ""}, {"description": " \n\t"}, {"description": None}, {"description": []},
    {"description": "x" * 2001}, {"description": "a\x00b"}, {"description": "\ud800"},
    {"idempotency_key": ""}, {"idempotency_key": " "}, {"idempotency_key": "a b"},
    {"idempotency_key": "x" * 256}, {"idempotency_key": None}, {"idempotency_key": 1},
    {"order_no": ""}, {"order_no": "' OR 1=1"}, {"order_no": None}, {"order_no": "x" * 65},
])
def test_invalid_arguments_no_writes(db, changes):
    result = request(db, **changes)
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert result["error"]["retryable"] is False
    assert count(db) == 0


def test_description_and_key_length_boundaries(db):
    assert request(db, description="x" * 2000, idempotency_key="k" * 255)["ok"]


def test_same_request_replays_existing_ticket(db):
    first = request(db)
    second = request(db, description="  Synthetic issue\n")
    assert first["data"]["created"] is True
    assert second["data"] == {**first["data"], "created": False}
    assert count(db) == 1


@pytest.mark.parametrize("changes", [
    {"description": "A different issue"}, {"ticket_type": "repair_request"},
    {"order_no": "OWN-completed"},
])
def test_same_key_different_request_conflicts_before_eligibility(db, changes):
    assert request(db)["ok"]
    assert request(db, **changes)["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert count(db) == 1


@pytest.mark.parametrize("status", [TicketStatus.PENDING_REVIEW, TicketStatus.APPROVED, TicketStatus.IN_PROGRESS])
def test_active_ticket_different_key_rejected(db, status):
    request(db)
    with db.factory.begin() as session:
        session.scalar(select(Ticket)).status = status
    result = request(db, idempotency_key="request-002")
    assert result["error"]["code"] == "ACTIVE_TICKET_EXISTS"
    assert count(db) == 1


@pytest.mark.parametrize("status", [TicketStatus.RESOLVED, TicketStatus.REJECTED, TicketStatus.CANCELLED])
def test_terminal_ticket_allows_new_request_but_original_key_still_replays(db, status):
    first = request(db)
    with db.factory.begin() as session:
        ticket = session.scalar(select(Ticket))
        ticket.status = status
        ticket.review_note = "Synthetic review"
        ticket.resolution_note = "Synthetic resolution"
    replay = request(db)
    assert replay["data"]["ticket_no"] == first["data"]["ticket_no"]
    assert replay["data"]["status"] == status.value
    assert replay["data"]["created"] is False
    assert request(db, idempotency_key="new-request")["data"]["created"] is True
    assert count(db) == 2


def test_replay_survives_changed_order_eligibility(db):
    request(db)
    with db.factory.begin() as session:
        session.scalar(select(Order).where(Order.order_no == "OWN-paid")).status = OrderStatus.CANCELLED
    assert request(db)["data"]["created"] is False
    assert request(db, idempotency_key="new-request")["error"]["code"] == "TICKET_TYPE_NOT_ALLOWED"


def test_key_is_scoped_to_user(db):
    assert request(db)["ok"]
    assert request(db, current_user_id=db.ids["other"], order_no="OTHER-ORDER")["ok"]
    assert count(db) == 2


def test_different_types_and_orders_can_coexist(db):
    assert request(db)["ok"]
    assert request(db, order_no="OWN-completed", idempotency_key="second")["ok"]
    assert request(db, order_no="OWN-completed", ticket_type="repair_request", idempotency_key="third")["ok"]
    assert count(db) == 3


@pytest.mark.parametrize("collision", ["same-request", "key-conflict", "active-conflict"])
def test_actual_unique_constraint_collisions_resolved_after_rollback(db, monkeypatch, collision):
    request(db)
    original_key = TicketRepository.get_by_idempotency_key
    original_active = TicketRepository.get_active
    key_calls = []
    active_calls = []

    def hide_key_once(self, **kwargs):
        key_calls.append(True)
        return None if len(key_calls) == 1 else original_key(self, **kwargs)

    def hide_active_once(self, **kwargs):
        active_calls.append(True)
        return None if len(active_calls) == 1 else original_active(self, **kwargs)

    monkeypatch.setattr(TicketRepository, "get_by_idempotency_key", hide_key_once)
    monkeypatch.setattr(TicketRepository, "get_active", hide_active_once)
    rollbacks = []
    event.listen(db.engine, "rollback", lambda connection: rollbacks.append(True))
    changes = {}
    if collision == "key-conflict":
        changes["description"] = "Different issue"
    elif collision == "active-conflict":
        changes["idempotency_key"] = "another-key"
    result = request(db, **changes)
    if collision == "same-request":
        assert result["ok"] and result["data"]["created"] is False
    else:
        assert result["error"]["code"] == (
            "IDEMPOTENCY_CONFLICT" if collision == "key-conflict" else "ACTIVE_TICKET_EXISTS"
        )
    assert rollbacks
    assert count(db) == 1


def test_check_constraint_failure_rolls_back_and_session_can_recover(db, monkeypatch, caplog):
    original_add = TicketRepository.add

    def corrupt(self, ticket):
        ticket.description = ""
        original_add(self, ticket)

    with monkeypatch.context() as patch:
        patch.setattr(TicketRepository, "add", corrupt)
        assert request(db)["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert count(db) == 0
    assert request(db)["ok"]
    assert "CHECK constraint" not in caplog.text


@pytest.mark.parametrize("error_class", [OperationalError, IntegrityError])
def test_commit_failure_not_reported_as_creation_and_rolls_back(db, error_class, caplog):
    class FailingSession(Session):
        pass

    def fail_commit(session):
        raise error_class("SYNTHETIC_PRIVATE_SQL_PATH", {}, Exception("Authorization: SYNTHETIC_TOKEN"))

    event.listen(FailingSession, "before_commit", fail_commit)
    db.service = TicketService(sessionmaker(bind=db.engine, class_=FailingSession))
    result = request(db)
    assert result["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert result["data"] is None
    assert count(db) == 0
    assert "SYNTHETIC_PRIVATE_SQL_PATH" not in caplog.text + json.dumps(result)
    assert "Authorization" not in caplog.text + json.dumps(result)


@pytest.mark.parametrize("method", ["begin_sqlite_write", "get_by_idempotency_key", "get_active", "add"])
def test_database_errors_return_safe_response(db, monkeypatch, caplog, method):
    marker = "SYNTHETIC_SQL_PRIVATE_PATH"
    monkeypatch.setattr(TicketRepository, method, Mock(side_effect=OperationalError(marker, {}, Exception(marker))))
    result = request(db)
    assert result["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert result["error"]["retryable"] is True
    assert marker not in caplog.text + json.dumps(result)
    assert count(db) == 0


@pytest.mark.parametrize("race", ["same-request", "different-key", "different-description"])
def test_concurrent_submissions_are_serialized_and_do_not_duplicate(db, race):
    barrier = Barrier(2)

    def submit(index):
        barrier.wait(timeout=5)
        changes = {}
        if index == 1 and race == "different-key":
            changes["idempotency_key"] = "second-request"
        elif index == 1 and race == "different-description":
            changes["description"] = "Different issue"
        return request(db, **changes)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, [0, 1]))
    assert count(db) == 1
    assert sum(r["ok"] and r["data"]["created"] for r in results) == 1
    if race == "same-request":
        assert all(r["ok"] for r in results)
        assert results[0]["data"]["ticket_no"] == results[1]["data"]["ticket_no"]
    else:
        failure = next(r for r in results if not r["ok"])
        assert failure["error"]["code"] == (
            "ACTIVE_TICKET_EXISTS" if race == "different-key" else "IDEMPOTENCY_CONFLICT"
        )
    assert db.engine.pool.checkedout() == 0


def test_no_user_data_in_response_and_foreign_missing_errors_identical(db):
    result = request(db)
    serialized = json.dumps(result)
    for forbidden in [*db.ids.values(), "auth_subject", "description", "idempotency_key", "request_fingerprint", "reviewed_by"]:
        assert forbidden not in serialized
    assert request(db, order_no="OTHER-ORDER") == request(db, order_no="MISSING-ORDER")
