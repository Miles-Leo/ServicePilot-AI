"""Application diagnostics with explicit ownership and metadata-only output."""

from datetime import datetime
import logging
from pathlib import Path
import re
from threading import RLock

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")
DEFAULT_LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
_setup_lock = globals().get("_setup_lock", RLock())
_OWNER = "servicepilot.logging.v1"
_TOOL = r"(?:get_order|prepare_ticket|rag_summarize)"
_SAFE_MESSAGES = tuple(re.compile(pattern) for pattern in (
    r"(?:knowledge_file_missing|knowledge_not_a_file|knowledge_file_hash_failed) stage=hash(?: details_omitted)?",
    r"knowledge_directory_unavailable stage=scan files=0",
    r"knowledge_directory_read_failed stage=scan details_omitted",
    r"knowledge_import_empty stage=(?:scan files|loading documents|splitting documents)=0",
    r"knowledge_import_skipped stage=deduplication",
    r"knowledge_import_succeeded stage=indexing documents=\d+",
    r"knowledge_import_failed stage=(?:import|initialization) details_omitted",
    r"rag_failed stage=(?:retrieval|generation|initialization|prompt_loading) details_omitted",
    r"rag_empty stage=retrieval documents=0",
    r"rag_retrieved stage=generation documents=\d+",
    r"rag_ready stage=initialization",
    r"prompt_loading_failed stage=system details_omitted",
    r"\[load_system_prompt\]在yaml配置文件中未找到配置项main_prompt_path",
    r"\[load_rag_prompt\]在yaml配置文件中未找到配置项rag_summarize_prompt_path",
    rf"\[tool monitor\]执行工具：{_TOOL}",
    rf"\[tool monitor\]工具{_TOOL}调用成功",
    rf"工具{_TOOL}调用失败，内部详情已隐藏",
    r"订单工具执行失败，内部详情已隐藏",
    r"\[log_before_model\]即将调用模型，带有\d+条消息。",
    r"ai_service_unavailable stage=(?:initialization|stream|response_render) partial_output=(?:True|False)",
    r"ticket_prepare_failed details_omitted",
    r"ticket_confirmation_(?:submit_requested|failed details_omitted|submit_succeeded status=pending_review)",
))


class _DiagnosticFilter(logging.Filter):
    """Only known diagnostic shapes may carry message data to a sink."""

    _servicepilot_filter = _OWNER

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = ""
        record.msg = message if any(rule.fullmatch(message) for rule in _SAFE_MESSAGES) else "application_event details_omitted"
        record.args = ()
        record.exc_info = record.exc_text = record.stack_info = None
        return True


def _protect(target) -> None:
    if not any(getattr(item, "_servicepilot_filter", None) == _OWNER for item in target.filters):
        target.addFilter(_DiagnosticFilter())


def get_logger(name: str = "agent", console_level: int = logging.INFO,
               file_level: int = logging.DEBUG, log_file=None) -> logging.Logger:
    """Configure only our sinks; foreign handlers and filters remain intact."""
    with _setup_lock:
        target = logging.getLogger(name)
        target.setLevel(logging.DEBUG)
        _protect(target)
        destination = Path(log_file) if log_file is not None else Path(LOG_ROOT) / f"{name}_{datetime.now():%Y%m%d}.log"
        destination = destination.absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        owned = {getattr(h, "_servicepilot_sink", None): h for h in target.handlers
                 if getattr(h, "_servicepilot_owner", None) == _OWNER}
        file_handler = owned.get("file")
        if file_handler is not None and file_handler.baseFilename != str(destination):
            target.removeHandler(file_handler)
            file_handler.close()
            owned.pop("file")
        for role, level in (("console", console_level), ("file", file_level)):
            handler = owned.get(role)
            if handler is None:
                handler = logging.StreamHandler() if role == "console" else logging.FileHandler(destination, encoding="utf-8")
                handler._servicepilot_owner = _OWNER
                handler._servicepilot_sink = role
                target.addHandler(handler)
            handler.setLevel(level)
            handler.setFormatter(DEFAULT_LOG_FORMAT)
            _protect(handler)
        return target


logger = get_logger()
