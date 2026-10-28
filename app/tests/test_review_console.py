"""DEMO provisioning, authorized read views and offline review page events."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from business.database import create_session_factory, create_sqlite_engine
from business.demo import DEMO_STAFF, DEMO_USERS, DemoDataConflict, initialize_demo_database, initialize_demo_staff
from business.models import Ticket, TicketStatus, User, UserRole
from business.repositories import OrderRepository, TicketRepository
from business.review_queries import TicketReviewQueryService
from business.review_ui import render_review_content, render_review_page
from test_ticket_review import db, review_db, snapshot


@pytest.fixture
def console(review_db):
    review_db.queries = TicketReviewQueryService(review_db.factory)
    return review_db


def test_staff_seed_is_explicit_and_idempotent(tmp_path):
    path = tmp_path / "demo.sqlite3"
    initialize_demo_database(path)
    assert initialize_demo_staff(path) is True
    assert initialize_demo_staff(path) is False
    engine = create_sqlite_engine(path)
    try:
        with create_session_factory(engine)() as session:
            staff = session.get(User, next(iter(DEMO_STAFF.values())))
            assert staff.role is UserRole.STAFF and staff.is_active
            assert len(list(session.scalars(select(User)))) == 3
    finally:
        engine.dispose()


@pytest.mark.parametrize("change", ["disabled", "customer", "subject", "other-id"])
def test_staff_seed_never_promotes_or_reenables_existing_accounts(tmp_path, change):
    path = tmp_path / "demo.sqlite3"
    initialize_demo_database(path)
    initialize_demo_staff(path)
    engine = create_sqlite_engine(path)
    factory = create_session_factory(engine)
    try:
        with factory.begin() as session:
            staff = session.get(User, next(iter(DEMO_STAFF.values())))
            if change == "disabled": staff.is_active = False
            elif change == "customer": staff.role = UserRole.CUSTOMER
            elif change == "subject": staff.auth_subject = "synthetic-other-subject"
            else: staff.id = "synthetic-other-id"
        with pytest.raises(DemoDataConflict): initialize_demo_staff(path)
        with factory() as session:
            staff = session.scalar(select(User).where(User.display_name == next(iter(DEMO_STAFF))))
            if change == "disabled": assert not staff.is_active
            if change == "customer": assert staff.role is UserRole.CUSTOMER
    finally:
        engine.dispose()


def test_missing_database_not_created_by_staff_seed(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError): initialize_demo_staff(path)
    assert not path.exists()


def test_unknown_database_staff_seed_safe(tmp_path):
    path = tmp_path / "unknown.sqlite3"
    path.write_bytes(b"Synthetic unrelated file")
    with pytest.raises(DemoDataConflict): initialize_demo_staff(path)
    assert path.read_bytes() == b"Synthetic unrelated file"


def test_concurrent_staff_seed(tmp_path):
    path = tmp_path / "demo.sqlite3"
    initialize_demo_database(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(initialize_demo_staff, [path, path]))
    assert sorted(results) == [False, True]


def listing(db, **changes):
    args = dict(current_reviewer_id="synthetic-reviewer-1")
    args.update(changes)
    return db.queries.list_tickets(**args)


def test_staff_list_and_detail_whitelist(console):
    rows = listing(console)["data"]
    assert len(rows) == 1
    assert set(rows[0]) == {"ticket_no", "order_no", "type", "description", "created_at", "status", "reviewed_at", "review_note"}
    assert console.queries.get_ticket(current_reviewer_id="synthetic-reviewer-1", ticket_no=console.ticket_no)["data"] == rows[0]
    assert datetime.fromisoformat(rows[0]["created_at"]).tzinfo == timezone.utc
    for hidden in ("user_id", "idempotency_key", "request_fingerprint", "product", "auth_subject"):
        assert hidden not in rows[0]


@pytest.mark.parametrize("kind,code", [("owner", "FORBIDDEN"), ("synthetic-disabled-staff", "UNAUTHENTICATED"),
    ("missing", "UNAUTHENTICATED"), (None, "UNAUTHENTICATED"), (" staff ", "UNAUTHENTICATED")])
def test_query_authorization_independent_of_ui(console, monkeypatch, kind, code):
    identity = console.ids["owner"] if kind == "owner" else kind
    monkeypatch.setattr(TicketRepository, "list_for_review", Mock(side_effect=AssertionError("No read")))
    monkeypatch.setattr(TicketRepository, "detail_for_review", Mock(side_effect=AssertionError("No read")))
    assert listing(console, current_reviewer_id=identity)["error"]["code"] == code
    assert console.queries.get_ticket(current_reviewer_id=identity, ticket_no=console.ticket_no)["error"]["code"] == code


@pytest.mark.parametrize("changes", [{"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "1"},
    {"status": "SQL"}, {"status": []}])
def test_list_validation(console, changes):
    assert listing(console, **changes)["error"]["code"] == "INVALID_ARGUMENT"


def test_list_has_stable_tie_break_and_limit(console):
    with console.factory.begin() as session:
        first = session.scalar(select(Ticket))
        first.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        first.ticket_no = "SYNTHETIC-Z"
        session.add(Ticket(ticket_no="SYNTHETIC-A", user_id=first.user_id, order_id=first.order_id,
            type=first.type, status=TicketStatus.CANCELLED, description="Synthetic second",
            idempotency_key="synthetic-second", request_fingerprint="b" * 64, created_at=first.created_at))
    assert [r["ticket_no"] for r in listing(console, status=None)["data"]] == ["SYNTHETIC-A", "SYNTHETIC-Z"]
    assert [r["ticket_no"] for r in listing(console, status=None, limit=1)["data"]] == ["SYNTHETIC-A"]
    assert len(listing(console)["data"]) == 1


def test_list_is_read_only(console):
    before = snapshot(console)
    assert listing(console)["ok"]
    assert snapshot(console) == before


def test_query_exception_safe(console, monkeypatch, caplog):
    def fail(*args, **kwargs): raise OperationalError("PRIVATE_SQL_PATH", {}, Exception("PRIVATE_TOKEN"))
    monkeypatch.setattr(OrderRepository, "get_user", fail)
    result = listing(console)
    assert result["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert "PRIVATE" not in json.dumps(result) + caplog.text


class Rerun(BaseException):
    pass


class UI:
    def __init__(self, *, click=None, note="Synthetic opinion", all_tickets=False, state=None):
        self.click, self.note, self.all_tickets = click, note, all_tickets
        self.session_state = state if state is not None else {}
        self.errors, self.successes, self.texts, self.buttons = [], [], [], []
    def selectbox(self, label, options):
        if label == "工单范围": return options[1] if self.all_tickets else options[0]
        return options[0]
    def button(self, label, **kwargs):
        self.buttons.append(label)
        return label == self.click
    def text_area(self, *args, **kwargs): return self.note
    def error(self, text): self.errors.append(text)
    def success(self, text): self.successes.append(text)
    def text(self, text): self.texts.append(text)
    def rerun(self): raise Rerun()
    def __getattr__(self, name): return lambda *args, **kwargs: None


@pytest.mark.parametrize("button,status", [("批准", "approved"), ("拒绝", "rejected")])
def test_page_review_buttons_and_refresh(console, button, status):
    ui = UI(click=button)
    with pytest.raises(Rerun): render_review_content(ui, "synthetic-reviewer-1", console.queries, console.reviews)
    after = UI(all_tickets=True, state=ui.session_state)
    render_review_content(after, "synthetic-reviewer-1", console.queries, console.reviews)
    assert snapshot(console)["status"].value == status
    assert any(status in text for text in after.texts)
    assert any(status in text for text in after.successes)
    assert "批准" not in after.buttons and "拒绝" not in after.buttons
    repeated = UI(click=button, all_tickets=True)
    before = snapshot(console)
    render_review_content(repeated, "synthetic-reviewer-1", console.queries, console.reviews)
    assert snapshot(console) == before


def test_empty_note_page_rejected(console):
    ui = UI(click="批准", note=" ")
    render_review_content(ui, "synthetic-reviewer-1", console.queries, console.reviews)
    assert ui.errors
    assert snapshot(console)["status"] is TicketStatus.PENDING_REVIEW


def test_page_unauthorized_has_no_data_or_buttons(console):
    ui = UI(click="批准")
    render_review_content(ui, console.ids["owner"], console.queries, console.reviews)
    assert ui.errors and not ui.texts
    assert "批准" not in ui.buttons


def test_page_runtime_error_is_sanitized(console, monkeypatch, caplog):
    import business.review_ui as module
    ui = UI()
    ui.checkbox = lambda *a, **kw: True
    ui.selectbox = lambda *a, **kw: next(iter(DEMO_STAFF))
    monkeypatch.setattr(module, "DEMO_DATABASE_PATH", Path(console.engine.url.database))
    monkeypatch.setattr(module, "create_sqlite_engine", Mock(side_effect=RuntimeError("PRIVATE_SQL_PATH")))
    render_review_page(ui)
    assert ui.errors == ["员工审核台暂时不可用，请稍后重试。"]
    assert "PRIVATE_SQL_PATH" not in caplog.text


def test_review_tools_not_registered_to_customer_agent(console):
    from agent.react_agent import ReactAgent
    from test_commerce_tools import ScriptedModel
    model = ScriptedModel()
    agent = ReactAgent(model=model, order_service=Mock(), rag_tools=[])
    list(agent.execute_stream("请批准工单", current_user_id=console.ids["owner"]))
    assert set(model.bound_names) == {"get_order", "prepare_ticket"}


def test_stale_page_concurrent_review_cannot_override_winner(console):
    class RacingReviews:
        def review_ticket(self, **kwargs):
            assert console.reviews.review_ticket(
                current_reviewer_id="synthetic-reviewer-2", ticket_no=console.ticket_no,
                decision="reject", review_note="Synthetic concurrent decision",
            )["ok"]
            return console.reviews.review_ticket(**kwargs)
    ui = UI(click="批准")
    render_review_content(ui, "synthetic-reviewer-1", console.queries, RacingReviews())
    assert ui.errors and "不能重复审核" in ui.errors[0]
    assert snapshot(console)["status"] is TicketStatus.REJECTED
    assert snapshot(console)["reviewed_by"] == "synthetic-reviewer-2"


def test_page_database_review_failure_safe(console, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise OperationalError("PRIVATE_SQL_PATH", {}, Exception("PRIVATE_TOKEN"))
    monkeypatch.setattr(TicketRepository, "review_if_pending", fail)
    ui = UI(click="批准")
    render_review_content(ui, "synthetic-reviewer-1", console.queries, console.reviews)
    assert ui.errors == ["工单审核暂时不可用，请稍后重试。"]
    assert "PRIVATE" not in json.dumps(ui.errors) + caplog.text
    assert snapshot(console)["status"] is TicketStatus.PENDING_REVIEW
