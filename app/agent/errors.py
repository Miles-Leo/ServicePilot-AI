"""Public AI failures and metadata-only diagnostics; never log exception payloads."""

import logging
from typing import Literal


AI_UNAVAILABLE_MESSAGE = "AI 服务暂时不可用，请稍后重试。"
logger = logging.getLogger("agent")


class _ProviderLogFilter(logging.Filter):
    """SDK retry/debug records can contain exception text or request bodies."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "ai_provider_event details_omitted"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


_provider_filter = _ProviderLogFilter()
def install_provider_log_filters() -> None:
    for name in (
        "langchain_community.chat_models.tongyi", "langchain_community.llms.tongyi",
        "langchain_community.embeddings.dashscope", "dashscope",
        "langchain_openai.chat_models.base", "openai._base_client", "openai._client",
        "openai._streaming", "httpx", "httpx2", "httpcore",
        "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy",
    ):
        logging.getLogger(name).addFilter(_provider_filter)


install_provider_log_filters()


class AIServiceUnavailable(RuntimeError):
    def __init__(self):
        super().__init__(AI_UNAVAILABLE_MESSAGE)


def log_ai_failure(
    stage: Literal["initialization", "stream", "response_render"], *, partial_output: bool
) -> None:
    # Do not pass the exception, traceback, request, response or user input here.
    logger.error("ai_service_unavailable stage=%s partial_output=%s", stage, partial_output)
