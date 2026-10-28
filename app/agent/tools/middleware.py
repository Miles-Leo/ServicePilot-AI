from typing import Callable
import json
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        return result
    except Exception as e:
        if request.tool_call['name'] == "get_order":
            logger.error("订单工具执行失败，内部详情已隐藏")
            return ToolMessage(
                content=json.dumps({"ok": False, "data": None, "error": {
                    "code": "TEMPORARY_UNAVAILABLE", "message": "订单查询暂时不可用，请稍后重试。", "retryable": True,
                }}, ensure_ascii=False),
                tool_call_id=request.tool_call['id'], name="get_order",
            )
        logger.error(f"工具{request.tool_call['name']}调用失败，内部详情已隐藏")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")


    return None
