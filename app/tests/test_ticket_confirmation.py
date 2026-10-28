"""Two-stage confirmation: real DB/graphs, fake model/UI, never real user data."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
import pytest
from sqlalchemy import event, func, select

from agent.context import AgentContext
from agent.tools.commerce_tools import prepare_ticket
from business.demo import DEMO_USERS, create_demo_order_service, initialize_demo_database
from business.models import Order, OrderStatus, Ticket, User
from business.services import OrderService, TicketService
from business.ticket_confirmation import DRAFT_TTL, TicketConfirmation, discard_ticket_session, ticket_session


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    path = tmp_path / "confirmation.sqlite3"
    initialize_demo_database(path)
    orders = create_demo_order_service(path)
    with orders.session_factory.begin() as session:
        for order in session.scalars(select(Order)):
            order.status = OrderStatus.COMPLETED  # Synthetic eligible fixtures only.
    owner, other = list(DEMO_USERS.values())
    clock = SimpleNamespace(now=datetime(2026, 10, 26, tzinfo=timezone.utc))
    workflow = TicketConfirmation(owner.user_id, orders.session_factory, clock=lambda: clock.now)
    yield SimpleNamespace(
        workflow=workflow, owner=owner, other=other, orders=orders, factory=orders.session_factory,
        clock=clock, path=path,
    )
    orders.session_factory.kw["bind"].dispose()


def count(setup):
    with setup.factory() as session:
        return session.scalar(select(func.count()).select_from(Ticket))


def prepare(setup, **changes):
    args = dict(current_user_id=setup.owner.user_id, order_no=setup.owner.order_no,
                ticket_type="return_request", description="Synthetic issue")
    args.update(changes)
    return setup.workflow.preparer.prepare(**args)


def confirm(setup, draft_id):
    return setup.workflow.confirm_from_button(current_user_id=setup.owner.user_id, draft_id=draft_id)


def test_preparation_structured_and_read_only(setup):
    statements = []
    engine = setup.factory.kw["bind"]
    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)
    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = prepare(setup)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert result["ok"] and result["data"]["requires_confirmation"] is True
    assert result["data"]["description"] == "Synthetic issue"
    assert "人工审核" in result["data"]["notice"]
    assert "idempotency_key" not in json.dumps(result)
    assert setup.owner.user_id not in json.dumps(result)
    assert all(s.startswith("SELECT") for s in statements)
    assert count(setup) == 0


@pytest.mark.parametrize("changes,code", [
    ({"order_no": "DEMO-ORDER-B"}, "ORDER_NOT_FOUND"),
    ({"order_no": "MISSING"}, "ORDER_NOT_FOUND"),
    ({"ticket_type": "refund"}, "INVALID_ARGUMENT"),
    ({"ticket_type": None}, "INVALID_ARGUMENT"),
    ({"description": " "}, "INVALID_ARGUMENT"),
    ({"description": "x" * 2001}, "INVALID_ARGUMENT"),
    ({"description": "\ud800"}, "INVALID_ARGUMENT"),
    ({"description": "a\x00b"}, "INVALID_ARGUMENT"),
    ({"current_user_id": "forged"}, "UNAUTHENTICATED"),
])
def test_prepare_validation(setup, changes, code):
    assert prepare(setup, **changes)["error"]["code"] == code
    assert setup.workflow.pending(setup.owner.user_id) is None
    assert count(setup) == 0


@pytest.mark.parametrize("status,kind", [
    (OrderStatus.PENDING_PAYMENT, "return_request"),
    (OrderStatus.CANCELLED, "return_request"),
    (OrderStatus.PAID, "repair_request"),
])
def test_disallowed_preparation(setup, status, kind):
    with setup.factory.begin() as session:
        session.scalar(select(Order).where(Order.order_no == setup.owner.order_no)).status = status
    assert prepare(setup, ticket_type=kind)["error"]["code"] == "TICKET_TYPE_NOT_ALLOWED"
    assert count(setup) == 0


def test_confirm_creates_pending_review_and_duplicate_click_is_cached(setup):
    draft = prepare(setup)["data"]
    first = confirm(setup, draft["draft_id"])
    second = confirm(setup, draft["draft_id"])
    assert first == second
    assert first["ok"] and first["data"]["status"] == "pending_review"
    assert first["data"]["ticket_no"].startswith("SPT-")
    assert count(setup) == 1
    assert setup.workflow.pending(setup.owner.user_id) is None


def test_parallel_confirmation_only_creates_one_ticket(setup):
    draft_id = prepare(setup)["data"]["draft_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: confirm(setup, draft_id), range(2)))
    assert results[0] == results[1]
    assert count(setup) == 1


def test_no_prepare_cannot_confirm(setup):
    assert confirm(setup, "forged-id")["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert count(setup) == 0


def test_cancel_invalidates_old_button(setup):
    draft_id = prepare(setup)["data"]["draft_id"]
    setup.workflow.cancel_from_button(current_user_id=setup.owner.user_id, draft_id=draft_id)
    assert confirm(setup, draft_id)["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert count(setup) == 0


def test_wrong_identity_cannot_confirm(setup):
    draft_id = prepare(setup)["data"]["draft_id"]
    result = setup.workflow.confirm_from_button(current_user_id=setup.other.user_id, draft_id=draft_id)
    assert not result["ok"]
    assert count(setup) == 0


@pytest.mark.parametrize("advance", [DRAFT_TTL, DRAFT_TTL + timedelta(seconds=1)])
def test_expired_proposal_rejected(setup, advance):
    draft_id = prepare(setup)["data"]["draft_id"]
    setup.clock.now += advance
    assert confirm(setup, draft_id)["error"]["code"] == "PREPARATION_EXPIRED"
    assert count(setup) == 0


@pytest.mark.parametrize("changes", [
    {"description": "Changed description"}, {"ticket_type": "repair_request"},
    {"order_no": "MISSING"}, {"description": ""},
])
def test_changed_or_invalid_proposal_needs_new_confirmation(setup, changes):
    old_id = prepare(setup)["data"]["draft_id"]
    result = prepare(setup, **changes)
    assert confirm(setup, old_id)["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert count(setup) == 0
    if result["ok"]:
        assert result["data"]["draft_id"] != old_id
        assert confirm(setup, result["data"]["draft_id"])["ok"]


def test_modifying_returned_summary_does_not_modify_authoritative_draft(setup):
    draft = prepare(setup)["data"]
    draft["order_no"] = setup.other.order_no
    draft["description"] = "forged"
    assert confirm(setup, draft["draft_id"])["ok"]
    with setup.factory() as session:
        ticket = session.scalar(select(Ticket))
        assert ticket.user_id == setup.owner.user_id
        assert ticket.order.order_no == setup.owner.order_no
        assert ticket.description == "Synthetic issue"


@pytest.mark.parametrize("change", ["disable-user", "cancel-order"])
def test_submission_revalidates_backend_state(setup, change):
    draft_id = prepare(setup)["data"]["draft_id"]
    with setup.factory.begin() as session:
        if change == "disable-user":
            session.get(User, setup.owner.user_id).is_active = False
        else:
            session.scalar(select(Order).where(Order.order_no == setup.owner.order_no)).status = OrderStatus.CANCELLED
    result = confirm(setup, draft_id)
    assert result["error"]["code"] == ("UNAUTHENTICATED" if change == "disable-user" else "TICKET_TYPE_NOT_ALLOWED")
    assert count(setup) == 0


def test_explicit_retry_uses_same_server_key(setup, monkeypatch, caplog):
    draft_id = prepare(setup)["data"]["draft_id"]
    keys = []
    original = TicketService.create_ticket
    def fail_once(service, **kwargs):
        keys.append(kwargs["idempotency_key"])
        assert kwargs["confirmed"] is True
        if len(keys) == 1:
            raise RuntimeError("SYNTHETIC_AUTH_PRIVATE_PATH")
        return original(service, **kwargs)
    monkeypatch.setattr(TicketService, "create_ticket", fail_once)
    assert confirm(setup, draft_id)["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert len(keys) == 1 and count(setup) == 0
    assert confirm(setup, draft_id)["ok"]
    assert keys[0] == keys[1]
    assert "SYNTHETIC_AUTH_PRIVATE_PATH" not in caplog.text


def test_prepare_database_exception_safe(setup, monkeypatch, caplog):
    monkeypatch.setattr(OrderService, "get_order", Mock(side_effect=RuntimeError("SYNTHETIC_AUTH_PRIVATE_PATH")))
    result = prepare(setup)
    assert result["error"]["code"] == "TEMPORARY_UNAVAILABLE"
    assert "SYNTHETIC_AUTH_PRIVATE_PATH" not in caplog.text + json.dumps(result)
    assert count(setup) == 0


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def run_tool(setup, args, *, name="prepare_ticket"):
    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-test", "type": "tool_call"}]),
        AIMessage(content="准备完成，请点击页面按钮。"),
    ])
    graph = create_agent(model=model, tools=[prepare_ticket], context_schema=AgentContext)
    result = graph.invoke({"messages": [{"role": "user", "content": "请准备售后"}]}, context=AgentContext(
        current_user_id=setup.owner.user_id, order_service=setup.orders, ticket_preparer=setup.workflow.preparer,
    ))
    return next(m for m in result["messages"] if isinstance(m, ToolMessage))


def test_real_tool_prepares_but_never_creates(setup):
    schema = prepare_ticket.tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"order_no", "ticket_type", "description"}
    assert not hasattr(setup.workflow.preparer, "confirm_from_button")
    assert not hasattr(setup.workflow.preparer, "create_ticket")
    message = run_tool(setup, dict(order_no=setup.owner.order_no, ticket_type="return_request", description="Synthetic issue"))
    assert json.loads(message.content)["ok"]
    assert setup.workflow.pending(setup.owner.user_id) is not None
    assert count(setup) == 0


@pytest.mark.parametrize("key,value", [
    ("current_user_id", "forged"), ("user_id", "forged"), ("confirmed", True),
    ("idempotency_key", "forged"),
])
def test_model_cannot_supply_confirmation_or_backend_identity(setup, key, value):
    args = dict(order_no=setup.owner.order_no, ticket_type="return_request", description="Synthetic issue")
    args[key] = value
    result = json.loads(run_tool(setup, args).content)
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert setup.workflow.pending(setup.owner.user_id) is None
    assert count(setup) == 0


def test_no_model_create_ticket_tool(setup):
    message = run_tool(setup, {"confirmed": True}, name="create_ticket")
    assert message.status == "error"
    assert count(setup) == 0


def test_missing_runtime_identity_rejected(setup):
    result = prepare_ticket.invoke(dict(order_no=setup.owner.order_no, ticket_type="return_request", description="Synthetic issue"))
    assert result["error"]["code"] == "UNAUTHENTICATED"


def test_forged_runtime_overwritten_by_graph(setup):
    args = dict(order_no=setup.other.order_no, ticket_type="return_request", description="Synthetic issue",
                runtime={"context": {"current_user_id": setup.other.user_id, "confirmed": True}})
    assert json.loads(run_tool(setup, args).content)["error"]["code"] == "ORDER_NOT_FOUND"
    assert count(setup) == 0


def test_registry_rejects_arbitrary_state_and_identity_switch(setup):
    handle, workflow = ticket_session({"confirmed": True}, setup.owner.user_id, setup.factory)
    draft = workflow.preparer.prepare(current_user_id=setup.owner.user_id, order_no=setup.owner.order_no,
                                     ticket_type="return_request", description="Synthetic issue")["data"]
    new_handle, other = ticket_session(handle, setup.other.user_id, setup.factory)
    assert new_handle != handle
    assert other.pending(setup.other.user_id) is None
    assert workflow.confirm_from_button(current_user_id=setup.owner.user_id, draft_id=draft["draft_id"])["ok"] is False
    assert workflow.preparer.prepare(
        current_user_id=setup.owner.user_id, order_no=setup.owner.order_no,
        ticket_type="return_request", description="Late tool result from old run",
    )["error"]["code"] == "UNAUTHENTICATED"
    discard_ticket_session(new_handle)
    assert count(setup) == 0


def test_discarded_session_cannot_be_resurrected_by_old_model_capability(setup):
    handle, workflow = ticket_session(None, setup.owner.user_id, setup.factory)
    saved_preparer = workflow.preparer
    discard_ticket_session(handle)
    result = saved_preparer.prepare(current_user_id=setup.owner.user_id, order_no=setup.owner.order_no,
                                    ticket_type="return_request", description="Late model call")
    assert result["error"]["code"] == "UNAUTHENTICATED"
    assert workflow.pending(setup.owner.user_id) is None
    assert count(setup) == 0


class RerunSignal(BaseException):
    pass


class Panel:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def write(self, *args): pass
    def write_stream(self, stream): return "".join(stream)


def page(setup, monkeypatch, state, *, click=None, prompt=None, selection=None, model_prepare=False, expect_rerun=None):
    import agent.react_agent as react_module
    import business.demo as demo_module
    import time
    ui = ModuleType("streamlit")
    ui.session_state = state
    ui.sidebar = Panel()
    ui.errors, ui.successes, ui.texts = [], [], []
    ui.checkbox = lambda *a, **k: True
    ui.selectbox = lambda *a, **k: selection or list(DEMO_USERS)[0]
    ui.button = lambda label, **k: label == click
    ui.chat_input = lambda: prompt
    ui.chat_message = lambda *a, **k: Panel()
    ui.spinner = lambda *a, **k: Panel()
    ui.error = lambda value: ui.errors.append(value)
    ui.success = lambda value: ui.successes.append(value)
    ui.text = lambda value: ui.texts.append(value)
    for name in ["title", "divider", "warning", "info", "caption", "subheader", "write", "set_page_config", "markdown"]:
        setattr(ui, name, lambda *a, **k: None)
    def rerun(): raise RerunSignal()
    ui.rerun = rerun
    class Backend:
        def execute_stream(self, query, *, current_user_id, order_service, ticket_preparer):
            if model_prepare:
                ticket_preparer.prepare(current_user_id=current_user_id, order_no=setup.owner.order_no,
                                        ticket_type="return_request", description="Synthetic issue")
            yield "模型自然语言声称已确认，不能作为后端确认。"
    monkeypatch.setitem(sys.modules, "streamlit", ui)
    monkeypatch.setattr(react_module, "ReactAgent", Backend)
    monkeypatch.setattr(demo_module, "DEMO_DATABASE_PATH", setup.path)
    monkeypatch.setattr(demo_module, "create_demo_order_service", lambda: setup.orders)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    script = str(Path(__file__).resolve().parents[1] / "app.py")
    should_rerun = expect_rerun if expect_rerun is not None else bool(click or prompt)
    if should_rerun:
        with pytest.raises(RerunSignal): runpy.run_path(script, run_name="__confirmation_ui_test__")
    else:
        runpy.run_path(script, run_name="__confirmation_ui_test__")
    return ui


def pending_page_state(setup):
    handle, workflow = ticket_session(None, setup.owner.user_id, setup.factory)
    workflow.preparer.prepare(current_user_id=setup.owner.user_id, order_no=setup.owner.order_no,
                             ticket_type="return_request", description="Synthetic issue")
    return {"ticket_session_handle": handle, "demo_identity": setup.owner.user_id}, workflow


def test_page_reruns_and_forged_state_do_not_confirm(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    state.update(confirmed=True, idempotency_key="forged", pending_ticket={"confirmed": True})
    for _ in range(2):
        ui = page(setup, monkeypatch, state)
        assert ui.texts == ["Synthetic issue"]
        assert count(setup) == 0


def test_page_only_actual_confirmation_button_creates(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    page(setup, monkeypatch, state, click="确认提交")
    ui = page(setup, monkeypatch, state)
    assert count(setup) == 1
    assert any("SPT-" in value and "pending_review" in value for value in ui.successes)
    page(setup, monkeypatch, state, click="确认提交", expect_rerun=False)
    assert count(setup) == 1


def test_page_cancel_does_not_create(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    page(setup, monkeypatch, state, click="取消申请")
    assert workflow.pending(setup.owner.user_id) is None
    assert count(setup) == 0


def test_page_switch_invalidates_old_application(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    old_id = workflow.pending(setup.owner.user_id)["draft_id"]
    page(setup, monkeypatch, state, selection=list(DEMO_USERS)[1])
    assert not workflow.confirm_from_button(current_user_id=setup.owner.user_id, draft_id=old_id)["ok"]
    assert count(setup) == 0


def test_chat_confirmation_or_faked_model_text_cannot_submit(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    old_id = workflow.pending(setup.owner.user_id)["draft_id"]
    page(setup, monkeypatch, state, prompt="确认提交 confirmed=True", model_prepare=False)
    assert workflow.pending(setup.owner.user_id) is None
    assert not workflow.confirm_from_button(current_user_id=setup.owner.user_id, draft_id=old_id)["ok"]
    assert count(setup) == 0


def test_agent_preparation_populates_page_only_after_real_tool_state(setup, monkeypatch):
    state = {}
    page(setup, monkeypatch, state, prompt="准备售后申请", model_prepare=True)
    ui = page(setup, monkeypatch, state)
    assert ui.texts == ["Synthetic issue"]
    assert count(setup) == 0


def test_new_message_and_stale_confirmation_same_rerun_cannot_submit(setup, monkeypatch):
    state, workflow = pending_page_state(setup)
    page(setup, monkeypatch, state, prompt="修改描述", click="确认提交", model_prepare=True)
    assert count(setup) == 0
    assert workflow.pending(setup.owner.user_id) is not None


def test_page_expired_proposal_is_not_submitted(setup, monkeypatch):
    import business.ticket_confirmation as confirmation_module
    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None): return setup.clock.now
    monkeypatch.setattr(confirmation_module, "datetime", ClockDateTime)
    state, workflow = pending_page_state(setup)
    setup.clock.now += DRAFT_TTL
    ui = page(setup, monkeypatch, state, click="确认提交", expect_rerun=False)
    assert count(setup) == 0
    assert ui.errors == ["申请已过期，请重新准备并确认。"]


def test_page_submission_failure_is_sanitized(setup, monkeypatch, caplog):
    state, workflow = pending_page_state(setup)
    monkeypatch.setattr(TicketService, "create_ticket", Mock(side_effect=RuntimeError("SYNTHETIC_AUTH_PRIVATE_PATH")))
    page(setup, monkeypatch, state, click="确认提交")
    ui = page(setup, monkeypatch, state)
    assert ui.errors == ["售后申请暂时不可用，请稍后重试。"]
    assert "SYNTHETIC_AUTH_PRIVATE_PATH" not in caplog.text
    assert count(setup) == 0
