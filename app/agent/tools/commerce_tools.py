"""Commerce tools receive identity/dependencies exclusively from trusted runtime."""

import json
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, ConfigDict, Field

from agent.context import AgentContext
from business.schemas import OrderQueryResponse, PreparedTicketResponse


def _invalid_arguments(_error) -> str:
    # Do not echo rejected inputs or Pydantic's validation internals to the model.
    return json.dumps({
        "ok": False, "data": None,
        "error": {"code": "INVALID_ARGUMENT", "message": "订单查询仅接受字符串参数 order_no。", "retryable": False},
    }, ensure_ascii=False)


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    order_no: str = Field(description="要查询的订单号，不包含用户身份参数")
    # ToolNode supplies this field and overwrites any model-supplied runtime.
    # InjectedToolArg removes it from the schema presented to the model.
    runtime: Annotated[Any, InjectedToolArg] = None


@tool(args_schema=GetOrderInput)
def get_order(order_no: str, runtime: ToolRuntime[AgentContext] = None) -> OrderQueryResponse:
    """查询当前登录用户的订单。只传 order_no，身份由后端提供；不得编造订单信息。"""
    context = runtime.context if isinstance(runtime, ToolRuntime) else None
    if not isinstance(context, AgentContext) or not context.current_user_id:
        return {
            "ok": False, "data": None,
            "error": {"code": "UNAUTHENTICATED", "message": "登录身份无效，请重新登录。", "retryable": False},
        }
    if context.order_service is None:
        return {
            "ok": False, "data": None,
            "error": {"code": "TEMPORARY_UNAVAILABLE", "message": "订单查询暂时不可用，请稍后重试。", "retryable": True},
        }
    return context.order_service.get_order(
        current_user_id=context.current_user_id, order_no=order_no
    )


get_order.handle_validation_error = _invalid_arguments


class PrepareTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    order_no: str
    ticket_type: str = Field(description="return_request、exchange_request 或 repair_request")
    description: str = Field(description="用户的问题描述，准备后仍需页面按钮确认")
    runtime: Annotated[Any, InjectedToolArg] = None


@tool(args_schema=PrepareTicketInput)
def prepare_ticket(
    order_no: str, ticket_type: str, description: str, runtime: ToolRuntime[AgentContext] = None,
) -> PreparedTicketResponse:
    """只准备待确认售后申请，不创建工单；用户必须在页面点击确认提交。"""
    context = runtime.context if isinstance(runtime, ToolRuntime) else None
    if not isinstance(context, AgentContext) or not context.current_user_id:
        return {"ok": False, "data": None, "error": {
            "code": "UNAUTHENTICATED", "message": "登录身份无效，请重新登录。", "retryable": False,
        }}
    if context.ticket_preparer is None:
        return {"ok": False, "data": None, "error": {
            "code": "TEMPORARY_UNAVAILABLE", "message": "申请准备暂时不可用，请稍后重试。", "retryable": True,
        }}
    return context.ticket_preparer.prepare(
        current_user_id=context.current_user_id, order_no=order_no,
        ticket_type=ticket_type, description=description,
    )


prepare_ticket.handle_validation_error = lambda _: json.dumps({
    "ok": False, "data": None, "error": {
        "code": "INVALID_ARGUMENT", "message": "准备申请仅接受订单号、售后类型和问题描述。", "retryable": False,
    },
}, ensure_ascii=False)
