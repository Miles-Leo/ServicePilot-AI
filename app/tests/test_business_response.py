"""Exercise user-visible execute_stream with a real graph and unreliable model."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest
from sqlalchemy import update
from sqlalchemy.exc import OperationalError

from agent.react_agent import ReactAgent
from business.models import Order, OrderStatus
from business.repositories import OrderRepository
from business.ticket_confirmation import TicketConfirmation
from test_commerce_tools import (
    ScriptedModel, demo, offline_environment, context_for,
)


BAD = "订单一定不存在，可能超过退货期限，商品类型可能不支持，物流异常，退款已成功。"


class UnreliableModel(ScriptedModel):
    def _generate(self, messages, **kwargs):
        if sum(isinstance(m, ToolMessage) for m in messages) < len(self.script):
            return super()._generate(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=BAD))])


def run(agent, demo, workflow=None):
    return "".join(agent.execute_stream(
        "请处理售后", current_user_id=context_for(demo).current_user_id,
        ticket_preparer=workflow.preparer if workflow else None,
    ))


def make_agent(demo, *, tool="prepare_ticket", order="DEMO-ORDER-A"):
    args = {"order_no": order}
    if tool == "prepare_ticket":
        args.update(ticket_type="return_request", description="虚构商品包装破损")
    return ReactAgent(model=UnreliableModel(script=[{"name": tool, "args": args}]),
                      order_service=demo, rag_tools=[])


@pytest.mark.parametrize("status", [OrderStatus.PENDING_PAYMENT, OrderStatus.CANCELLED])
def test_rejected_status_visible_response_has_no_speculation(demo, status):
    with demo.session_factory.begin() as session:
        session.execute(update(Order).where(Order.order_no == "DEMO-ORDER-A").values(status=status))
    workflow = TicketConfirmation(context_for(demo).current_user_id, demo.session_factory)
    answer = run(make_agent(demo), demo, workflow)
    assert "当前订单状态不符合" in answer
    assert "退货期限" not in answer and "商品类型" not in answer and "物流" not in answer
    assert workflow.pending(context_for(demo).current_user_id) is None


@pytest.mark.parametrize("order", ["DEMO-ORDER-B", "DEMO-ORDER-MISSING"])
def test_missing_and_foreign_orders_have_same_safe_visible_response(demo, order):
    answer = run(make_agent(demo, tool="get_order", order=order), demo)
    assert answer.strip() == "当前账号下未找到可访问的该订单，请核对订单号及当前登录账号。"


def test_database_unavailable_does_not_become_business_denial(demo, monkeypatch):
    def fail(*args, **kwargs):
        raise OperationalError("PRIVATE_DB_PATH", {}, Exception("PRIVATE"))
    monkeypatch.setattr(OrderRepository, "get_user", fail)
    answer = run(make_agent(demo, tool="get_order"), demo)
    assert "业务服务暂时不可用" in answer
    assert "退货期限" not in answer and "PRIVATE" not in answer


@pytest.mark.parametrize("payload", [None, {}, {"ok": False, "error": {}},
    {"ok": False, "error": {"code": "UNKNOWN", "message": BAD}}])
def test_insufficient_tool_reason_cannot_be_embellished(demo, monkeypatch, payload):
    monkeypatch.setattr(demo, "get_order", lambda **kwargs: payload)
    answer = run(make_agent(demo, tool="get_order"), demo)
    assert "暂时无法确认" in answer
    assert BAD not in answer and "退货期限" not in answer


def test_eligible_order_preparation_keeps_normal_model_response(demo):
    with demo.session_factory.begin() as session:
        session.execute(update(Order).where(Order.order_no == "DEMO-ORDER-A").values(status=OrderStatus.COMPLETED))
    workflow = TicketConfirmation(context_for(demo).current_user_id, demo.session_factory)
    model = ScriptedModel(script=[{"name": "prepare_ticket", "args": {
        "order_no": "DEMO-ORDER-A", "ticket_type": "return_request", "description": "虚构商品破损",
    }}])
    agent = ReactAgent(model=model, order_service=demo, rag_tools=[])
    assert run(agent, demo, workflow) == "演示回答完成。\n"
    assert workflow.pending(context_for(demo).current_user_id)


def test_regular_chat_unaffected(demo):
    agent = ReactAgent(model=ScriptedModel(), order_service=demo, rag_tools=[])
    assert run(agent, demo) == "演示回答完成。\n"


def test_later_success_does_not_erase_earlier_access_failure(demo):
    model = UnreliableModel(script=[
        {"name": "get_order", "args": {"order_no": "DEMO-ORDER-B"}},
        {"name": "get_order", "args": {"order_no": "DEMO-ORDER-A"}},
    ])
    agent = ReactAgent(model=model, order_service=demo, rag_tools=[])
    answer = run(agent, demo)
    assert "当前账号下未找到可访问" in answer
    assert BAD not in answer


def test_valid_order_query_preserves_normal_response(demo):
    model = ScriptedModel(script=[{"name": "get_order", "args": {"order_no": "DEMO-ORDER-A"}}])
    agent = ReactAgent(model=model, order_service=demo, rag_tools=[])
    assert run(agent, demo) == "演示回答完成。\n"
