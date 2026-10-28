"""Human review authorization, atomic records, rollback and real SQLite races."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from threading import Barrier
from unittest.mock import Mock

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from business.models import Order, Product, Ticket, TicketStatus, User, UserRole
from business.repositories import OrderRepository, TicketRepository
from business.services import TicketReviewService
from test_ticket_service import db, request


@pytest.fixture
def review_db(db):
    result = request(db)
    db.ticket_no = result["data"]["ticket_no"]
    with db.factory.begin() as session:
        for number in (1, 2):
            session.add(User(
                id=f"synthetic-reviewer-{number}", auth_subject=f"synthetic-staff-{number}",
                display_name=f"Synthetic reviewer {number}", role=UserRole.STAFF,
            ))
        session.add(User(id="synthetic-disabled-staff", auth_subject="synthetic-disabled-staff",
                         display_name="Disabled staff", role=UserRole.STAFF, is_active=False))
    db.reviews = TicketReviewService(db.factory)
    return db


def review(db, **changes):
    args = dict(current_reviewer_id="synthetic-reviewer-1", ticket_no=db.ticket_no,
                decision="approve", review_note="Synthetic review opinion")
    args.update(changes)
    return db.reviews.review_ticket(**args)


def snapshot(db):
    with db.factory() as session:
        ticket = session.scalar(select(Ticket).where(Ticket.ticket_no == db.ticket_no))
        return {column.name: getattr(ticket, column.name) for column in Ticket.__table__.columns}


@pytest.mark.parametrize("decision,status", [("approve", TicketStatus.APPROVED), ("reject", TicketStatus.REJECTED)])
def test_staff_review_records_and_response(review_db, decision, status):
    db = review_db
    before = snapshot(db)
    result = review(db, decision=decision, review_note="  Synthetic review opinion  ")
    assert result["ok"] is True and result["error"] is None
    assert set(result["data"]) == {"ticket_no", "status", "reviewed_by", "reviewed_at", "review_note"}
    assert result["data"]["status"] == status.value
    assert json.loads(json.dumps(result)) == result
    after = snapshot(db)
    assert after["status"] is status
    assert after["reviewed_by"] == "synthetic-reviewer-1"
    assert after["review_note"] == "Synthetic review opinion"
    assert after["reviewed_at"].tzinfo == timezone.utc
    assert datetime.fromisoformat(result["data"]["reviewed_at"]) == after["reviewed_at"]
    assert after["updated_at"] == after["reviewed_at"]
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"status", "reviewed_by", "review_note", "reviewed_at", "updated_at"}


@pytest.mark.parametrize("identity,code", [
    ("customer", "FORBIDDEN"), ("synthetic-disabled-staff", "UNAUTHENTICATED"),
    ("missing-reviewer", "UNAUTHENTICATED"), (None, "UNAUTHENTICATED"),
    ("", "UNAUTHENTICATED"), (" staff ", "UNAUTHENTICATED"),
    (1, "UNAUTHENTICATED"), ("x" * 37, "UNAUTHENTICATED"),
])
def test_review_identity_and_role(review_db, identity, code):
    db = review_db
    before = snapshot(db)
    if identity == "customer":
        identity = db.ids["owner"]
    result = review(db, current_reviewer_id=identity)
    assert result["error"]["code"] == code and result["data"] is None
    assert snapshot(db) == before


def test_forbidden_cannot_discover_ticket_existence(review_db, monkeypatch):
    monkeypatch.setattr(TicketRepository, "get_by_ticket_no", Mock(side_effect=AssertionError("No lookup")))
    assert review(review_db, current_reviewer_id=review_db.ids["owner"], ticket_no="MISSING")["error"]["code"] == "FORBIDDEN"


def test_missing_ticket(review_db):
    result = review(review_db, ticket_no="SYNTHETIC-MISSING")
    assert result["error"]["code"] == "TICKET_NOT_FOUND"


@pytest.mark.parametrize("status", [s for s in TicketStatus if s is not TicketStatus.PENDING_REVIEW])
def test_only_pending_status_can_be_reviewed(review_db, status):
    db = review_db
    with db.factory.begin() as session:
        ticket = session.scalar(select(Ticket))
        ticket.status = status
        if status is TicketStatus.REJECTED:
            ticket.review_note = "Synthetic existing rejection"
        if status is TicketStatus.RESOLVED:
            ticket.resolution_note = "Synthetic existing resolution"
    before = snapshot(db)
    assert review(db)["error"]["code"] == "INVALID_TICKET_STATUS"
    assert snapshot(db) == before


@pytest.mark.parametrize("changes", [
    {"decision": "refund"}, {"decision": "Approve"}, {"decision": True}, {"decision": None},
    {"review_note": ""}, {"review_note": " \n "}, {"review_note": None}, {"review_note": 1},
    {"review_note": "x" * 2001}, {"review_note": "a\x00b"}, {"review_note": "\ud800"},
    {"ticket_no": ""}, {"ticket_no": "bad number"}, {"ticket_no": "x" * 65},
    {"ticket_no": None}, {"ticket_no": "' OR 1=1 --"},
])
def test_invalid_arguments_do_not_access_database(review_db, changes):
    factory = Mock(side_effect=AssertionError("No database access"))
    review_db.reviews = TicketReviewService(factory)
    assert review(review_db, **changes)["error"]["code"] == "INVALID_ARGUMENT"
    factory.assert_not_called()


@pytest.mark.parametrize("length", [1, 2000])
def test_note_boundaries(review_db, length):
    assert review(review_db, review_note="审" * length)["ok"]


@pytest.mark.parametrize("first", ["approve", "reject"])
def test_repeat_cannot_overwrite_review(review_db, first):
    db = review_db
    assert review(db, decision=first)["ok"]
    before = snapshot(db)
    result = review(db, current_reviewer_id="synthetic-reviewer-2", decision="reject", review_note="Changed")
    assert result["error"]["code"] == "INVALID_TICKET_STATUS"
    assert snapshot(db) == before


def test_pending_with_existing_audit_record_is_not_overwritten(review_db):
    db = review_db
    with db.factory.begin() as session:
        session.scalar(select(Ticket)).review_note = "Synthetic preexisting audit record"
    before = snapshot(db)
    assert review(db)["error"]["code"] == "INVALID_TICKET_STATUS"
    assert snapshot(db) == before


def test_two_reviewers_race_has_one_winner(review_db):
    db = review_db
    barrier = Barrier(2)
    def submit(number):
        barrier.wait(timeout=5)
        return review(db, current_reviewer_id=f"synthetic-reviewer-{number}",
                      decision="approve" if number == 1 else "reject")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (1, 2)))
    assert sum(result["ok"] for result in results) == 1
    loser = next(result for result in results if not result["ok"])
    assert loser["error"]["code"] == "INVALID_TICKET_STATUS"
    winner = next(result for result in results if result["ok"])["data"]
    persisted = snapshot(db)
    assert persisted["reviewed_by"] == winner["reviewed_by"]
    assert persisted["status"].value == winner["status"]


@pytest.mark.parametrize("stage", ["lookup", "after-update", "commit"])
def test_database_exception_rolls_back_and_is_safe(review_db, monkeypatch, caplog, stage):
    db = review_db
    before = snapshot(db)
    def fail(*args, **kwargs):
        raise OperationalError("SYNTHETIC_PRIVATE_SQL_PATH", {}, Exception("PRIVATE_AUTH"))
    with monkeypatch.context() as patch:
        if stage == "lookup":
            patch.setattr(OrderRepository, "get_user", fail)
        elif stage == "after-update":
            original = TicketRepository.review_if_pending
            def update_then_fail(repository, **kwargs):
                original(repository, **kwargs)
                fail()
            patch.setattr(TicketRepository, "review_if_pending", update_then_fail)
        else:
            class FailingCommitSession(Session):
                pass
            event.listen(FailingCommitSession, "before_commit", fail)
            db.reviews = TicketReviewService(sessionmaker(bind=db.engine, class_=FailingCommitSession))
        result = review(db)
    assert result["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert result["error"]["retryable"] is True
    assert "SYNTHETIC_PRIVATE_SQL_PATH" not in json.dumps(result) + caplog.text
    assert "PRIVATE_AUTH" not in json.dumps(result) + caplog.text
    assert snapshot(db) == before


def test_update_is_database_compare_and_set(review_db):
    statements = []
    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)
    event.listen(review_db.engine, "before_cursor_execute", capture)
    try:
        assert review(review_db)["ok"]
    finally:
        event.remove(review_db.engine, "before_cursor_execute", capture)
    updates = [sql for sql in statements if sql.startswith("UPDATE")]
    assert len(updates) == 1
    assert "tickets.status =" in updates[0].split("WHERE")[1]
    assert "tickets.reviewed_by IS NULL" in updates[0]


def test_review_does_not_mutate_orders_products_or_users(review_db):
    db = review_db
    def records():
        with db.factory() as session:
            return {model.__tablename__: [tuple(getattr(row, col.name) for col in model.__table__.columns)
                    for row in session.scalars(select(model).order_by(model.id))]
                    for model in (Order, Product, User)}
    before = records()
    assert review(db)["ok"]
    assert records() == before
