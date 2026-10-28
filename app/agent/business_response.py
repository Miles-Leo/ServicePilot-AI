"""Render failed commerce observations without model-invented explanations."""

import json

from langchain_core.messages import AIMessage, ToolMessage


UNKNOWN = "工具未提供足够信息，暂时无法确认订单状态或售后资格，请稍后重试。"
ERROR_MESSAGES = {
    "TICKET_TYPE_NOT_ALLOWED": "当前订单状态不符合该类型的售后申请条件，暂时无法准备申请。",
    "ORDER_NOT_FOUND": "当前账号下未找到可访问的该订单，请核对订单号及当前登录账号。",
    "TEMPORARY_UNAVAILABLE": "业务服务暂时不可用，请稍后重试；目前无法确认订单状态或售后资格。",
    "UNAUTHENTICATED": "当前登录身份无效，请重新登录后再试。",
    "INVALID_ARGUMENT": "申请或查询参数不符合要求，请核对订单号、售后类型及问题描述。",
    "CONFIRMATION_REQUIRED": "请重新准备申请，并在页面点击“确认提交”。",
    "PREPARATION_EXPIRED": "申请已过期，请重新准备并确认。",
    "ACTIVE_TICKET_EXISTS": "该订单已存在同类型的未结束售后工单，不能重复申请。",
    "IDEMPOTENCY_CONFLICT": "此次提交与已有请求不一致，请重新准备申请。",
}


def commerce_failure_messages(messages):
    """Only trust observations linked to actual commerce tool calls in this run.

    An earlier failure remains visible even if the model subsequently changes
    tools or produces a confident answer. Never echo arbitrary error messages.
    """
    calls = {
        call["id"]: call["name"]
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] in {"get_order", "prepare_ticket"}
    }
    failures = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.tool_call_id not in calls:
            continue
        if message.name != calls[message.tool_call_id]:
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            result = None
        if isinstance(result, dict) and result.get("ok") is True:
            continue
        error = result.get("error") if isinstance(result, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        text = ERROR_MESSAGES.get(code, UNKNOWN) if isinstance(code, str) else UNKNOWN
        if text not in failures:
            failures.append(text)
    return failures
