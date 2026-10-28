"""Failure boundaries must not expose provider payloads or persist partial answers."""

import logging
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest
from requests.exceptions import HTTPError, Timeout

from agent.errors import AIServiceUnavailable, AI_UNAVAILABLE_MESSAGE
from agent.react_agent import ReactAgent


PRIVATE_DETAIL = (
    "SYNTHETIC_API_KEY Authorization: Bearer SYNTHETIC_TOKEN "
    "C:/private/customer.sqlite3 DEMO-PRIVATE-ORDER customer-sensitive-data"
)


def failing_chunks(error, partial=False):
    if partial:
        yield {"messages": [AIMessage(content="不完整回答")]}
    raise error


def bare_agent(stream):
    agent = ReactAgent.__new__(ReactAgent)
    agent.order_service = None
    agent.agent = SimpleNamespace(stream=stream)
    return agent


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, Timeout, HTTPError, RuntimeError])
@pytest.mark.parametrize("partial", [False, True])
def test_model_and_stream_errors_are_safe_and_not_retried(error_type, partial, caplog):
    stream = Mock(return_value=failing_chunks(error_type(PRIVATE_DETAIL), partial))
    agent = bare_agent(stream)
    with caplog.at_level(logging.ERROR, logger="agent"):
        with pytest.raises(AIServiceUnavailable) as caught:
            list(agent.execute_stream("synthetic question"))
    assert str(caught.value) == AI_UNAVAILABLE_MESSAGE
    assert caught.value.__suppress_context__
    assert stream.call_count == 1
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.exc_info is None
    assert record.stack_info is None
    assert record.getMessage() == f"ai_service_unavailable stage=stream partial_output={partial}"
    for marker in PRIVATE_DETAIL.split():
        assert marker not in caplog.text


def test_error_when_creating_stream_is_also_caught(caplog):
    agent = bare_agent(Mock(side_effect=HTTPError(PRIVATE_DETAIL)))
    with pytest.raises(AIServiceUnavailable, match=AI_UNAVAILABLE_MESSAGE):
        list(agent.execute_stream("synthetic question"))
    assert PRIVATE_DETAIL not in caplog.text


@pytest.mark.parametrize("chunks", [[], [{"messages": [AIMessage(content=" ")]}]])
def test_empty_response_is_failure(chunks):
    agent = bare_agent(Mock(return_value=iter(chunks)))
    with pytest.raises(AIServiceUnavailable):
        list(agent.execute_stream("synthetic question"))


def test_agent_can_answer_next_question_after_stream_failure():
    stream = Mock(side_effect=[
        failing_chunks(ConnectionError(PRIVATE_DETAIL), partial=True),
        iter([{"messages": [AIMessage(content="完整回答")]}]),
    ])
    agent = bare_agent(stream)
    with pytest.raises(AIServiceUnavailable):
        list(agent.execute_stream("first question"))
    assert list(agent.execute_stream("second question")) == ["完整回答\n"]
    assert stream.call_count == 2  # One attempt per explicit user question.


class RerunSignal(BaseException):
    """Like Streamlit control flow: reruns are not ordinary application errors."""


class Panel:
    def __init__(self, ui):
        self.ui = ui

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def write(self, content):
        self.ui.visible.append(content)

    def write_stream(self, stream):
        for chunk in stream:
            self.ui.visible.append(chunk)
        if self.ui.render_error is not None:
            raise self.ui.render_error


def run_page(monkeypatch, state, backend, prompt=None, render_error=None):
    """Run the actual page script; only Streamlit transport and Agent are replaced."""
    import agent.react_agent as agent_module
    import time

    ui = ModuleType("streamlit")
    ui.session_state = state
    ui.visible = []
    ui.errors = []
    ui.render_error = render_error
    ui.sidebar = Panel(ui)
    ui.checkbox = lambda *a, **kw: False
    ui.chat_input = lambda: prompt
    ui.chat_message = lambda *a, **kw: Panel(ui)
    ui.spinner = lambda *a, **kw: Panel(ui)
    ui.error = lambda message: ui.errors.append(message)
    for name in ("title", "divider", "warning", "caption", "set_page_config", "markdown"):
        setattr(ui, name, lambda *a, **kw: None)

    def rerun():
        # A real rerun removes this render's transient output.
        ui.visible.clear()
        raise RerunSignal()

    ui.rerun = rerun
    monkeypatch.setitem(sys.modules, "streamlit", ui)
    monkeypatch.setattr(agent_module, "ReactAgent", backend)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    script = Path(__file__).resolve().parents[1] / "app.py"
    if prompt:
        with pytest.raises(RerunSignal):
            runpy.run_path(str(script), run_name="__failure_ui_test__")
    else:
        runpy.run_path(str(script), run_name="__failure_ui_test__")
    return ui


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("safe_error", [False, True])
def test_page_discards_partial_answer_shows_only_banner_and_recovers(monkeypatch, caplog, partial, safe_error):
    calls = []

    class Backend:
        def execute_stream(self, query, **kwargs):
            calls.append(query)
            if len(calls) == 1:
                if partial:
                    yield "不完整回答"
                if safe_error:
                    raise AIServiceUnavailable()
                raise HTTPError(PRIVATE_DETAIL)
            yield "完整回答"

    state = {}
    first = run_page(monkeypatch, state, Backend, prompt="first question")
    assert first.visible == []
    assert state["message"] == [{"role": "user", "content": "first question"}]
    assert state["ai_service_error"] is True
    banner = run_page(monkeypatch, state, Backend)
    assert banner.errors == [AI_UNAVAILABLE_MESSAGE]
    assert "不完整回答" not in "".join(banner.visible)
    assert PRIVATE_DETAIL not in repr(banner.errors) + repr(state) + caplog.text
    run_page(monkeypatch, state, Backend, prompt="second question")
    final = run_page(monkeypatch, state, Backend)
    assert final.errors == []
    assert state["message"][-1] == {"role": "assistant", "content": "完整回答"}
    assert calls == ["first question", "second question"]


def test_agent_initialization_failure_is_safe_and_next_question_can_reinitialize(monkeypatch, caplog):
    class Backend:
        def execute_stream(self, query, **kwargs):
            yield "完整回答"

    constructor = Mock(side_effect=[HTTPError(PRIVATE_DETAIL), Backend()])
    state = {}
    run_page(monkeypatch, state, constructor, prompt="first question")
    assert "agent" not in state
    assert all(message["role"] != "assistant" for message in state["message"])
    banner = run_page(monkeypatch, state, constructor)
    assert banner.errors == [AI_UNAVAILABLE_MESSAGE]
    assert constructor.call_count == 1  # Rendering the banner does not retry.
    run_page(monkeypatch, state, constructor, prompt="second question")
    assert constructor.call_count == 2
    assert state["message"][-1]["content"] == "完整回答"
    assert PRIVATE_DETAIL not in caplog.text
    assert "stage=initialization" in caplog.text


def test_response_renderer_failure_does_not_save_success(monkeypatch, caplog):
    class Backend:
        def execute_stream(self, query, **kwargs):
            yield "已生成但未完成展示的回答"

    state = {}
    run_page(monkeypatch, state, Backend, prompt="question", render_error=RuntimeError(PRIVATE_DETAIL))
    assert state["message"] == [{"role": "user", "content": "question"}]
    assert run_page(monkeypatch, state, Backend).errors == [AI_UNAVAILABLE_MESSAGE]
    assert "stage=response_render" in caplog.text
    assert PRIVATE_DETAIL not in caplog.text


def test_middleware_logs_neither_message_bodies_nor_tool_arguments(monkeypatch, caplog):
    from agent.context import AgentContext

    # Avoid configuring the tutorial's real log files just to test log policy.
    module = ModuleType("utils.logger_handler")
    module.logger = logging.getLogger("agent")
    monkeypatch.setitem(sys.modules, "utils.logger_handler", module)
    import agent.tools.middleware as middleware

    monkeypatch.setattr(middleware, "logger", logging.getLogger("agent"))
    with caplog.at_level(logging.DEBUG, logger="agent"):
        middleware.log_before_model.before_model(
            {"messages": [AIMessage(content=PRIVATE_DETAIL)]}, SimpleNamespace(context=AgentContext())
        )
        request = SimpleNamespace(
            tool_call={"name": "rag_summarize", "args": {"query": PRIVATE_DETAIL}},
            runtime=SimpleNamespace(context=AgentContext()),
        )
        with pytest.raises(RuntimeError):
            middleware.monitor_tool.wrap_tool_call(request, Mock(side_effect=RuntimeError(PRIVATE_DETAIL)))
    assert PRIVATE_DETAIL not in caplog.text
    assert "SYNTHETIC_API_KEY" not in caplog.text
    assert "Authorization" not in caplog.text
    assert "调用失败" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("provider", [
    "langchain_community.chat_models.tongyi", "langchain_community.llms.tongyi", "dashscope",
])
@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_provider_retry_and_debug_logs_are_redacted(caplog, provider, level):
    with caplog.at_level(logging.DEBUG, logger=provider):
        logging.getLogger(provider).log(
            level, "Provider payload: %s", PRIVATE_DETAIL,
            exc_info=(RuntimeError, RuntimeError(PRIVATE_DETAIL), None), stack_info=True,
        )
    assert PRIVATE_DETAIL not in caplog.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "ai_provider_event details_omitted"
    assert record.args == ()
    assert record.exc_info is record.exc_text is record.stack_info is None
