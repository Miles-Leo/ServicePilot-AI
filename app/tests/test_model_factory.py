"""Unified Qwen initialization and real OpenAI adapter using fake clients only."""

import json
import logging
from pathlib import Path
import runpy
from unittest.mock import Mock

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
import pytest

from agent.errors import AIServiceUnavailable
from agent.react_agent import ReactAgent
from agent.tools.commerce_tools import get_order
from test_commerce_tools import demo, offline_environment, context_for, rag_tools


FACTORY = Path(__file__).resolve().parents[1] / "model/factory.py"
DEFAULT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
SECRET = "SYNTHETIC_MODEL_SECRET_NEVER_SENT"


def load_factory(monkeypatch, *, key=SECRET, base_url=None):
    if key is None: monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    else: monkeypatch.setenv("DASHSCOPE_API_KEY", key)
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC_WRONG_PROVIDER_KEY")
    if base_url is None: monkeypatch.delenv("DASHSCOPE_CHAT_BASE_URL", raising=False)
    else: monkeypatch.setenv("DASHSCOPE_CHAT_BASE_URL", base_url)
    import langchain.chat_models as models
    import langchain_community.embeddings as embeddings
    chat = Mock(return_value=object()); embed = Mock(return_value=object())
    monkeypatch.setattr(models, "init_chat_model", chat)
    monkeypatch.setattr(embeddings, "DashScopeEmbeddings", embed)
    exports = runpy.run_path(str(FACTORY))
    return exports, chat, embed


@pytest.mark.parametrize("base_url", [None, DEFAULT_URL,
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
    "https://synthetic-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"])
def test_factory_passes_qwen_provider_credentials_and_url(monkeypatch, base_url, caplog, capsys):
    exports, chat, embed = load_factory(monkeypatch, base_url=base_url)
    chat.assert_called_once_with(model="qwen-max", model_provider="openai", api_key=SECRET,
        base_url=(base_url or DEFAULT_URL).rstrip("/"), use_responses_api=False)
    embed.assert_called_once_with(model="text-embedding-v4")
    assert exports["chat_model"] is chat.return_value
    assert exports["embed_model"] is embed.return_value
    assert SECRET not in caplog.text + capsys.readouterr().out


@pytest.mark.parametrize("key", [None, "", " ", "REPLACE_WITH_YOUR_DASHSCOPE_API_KEY"])
def test_missing_key_fails_safely_without_openai_key_fallback(monkeypatch, key):
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY") as caught:
        load_factory(monkeypatch, key=key)
    assert "SYNTHETIC_WRONG_PROVIDER_KEY" not in str(caught.value)


@pytest.mark.parametrize("base_url", ["", "http://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://example.invalid/compatible-mode/v1", "https://dashscope.aliyuncs.com/api/v1",
    "https://secret@dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1?key=PRIVATE",
    "https://dashscope.aliyuncs.com:bad/compatible-mode/v1"])
def test_invalid_base_url_error_does_not_echo_input(monkeypatch, base_url):
    with pytest.raises(ValueError) as caught: load_factory(monkeypatch, base_url=base_url)
    assert str(caught.value) == "DASHSCOPE_CHAT_BASE_URL 不是有效的 DashScope HTTPS 兼容接口地址。"


def provider(client):
    model = init_chat_model(model="qwen-max", model_provider="openai", api_key=SECRET,
        base_url=DEFAULT_URL, use_responses_api=False, client=client, async_client=Mock())
    assert isinstance(model, ChatOpenAI)
    return model


def completion(content=None, tool_calls=None):
    return {"id": "synthetic-completion", "object": "chat.completion", "created": 0, "model": "qwen-max",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                     "message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}


def test_actual_provider_tool_call_and_agent_final_response(demo):
    client = Mock()
    raw_tool, raw_final = Mock(), Mock()
    raw_tool.parse.return_value = completion(tool_calls=[{"id": "synthetic-order-call", "type": "function",
        "function": {"name": "get_order", "arguments": json.dumps({"order_no": "DEMO-ORDER-A"})}}])
    raw_final.parse.return_value = completion("订单状态为待支付。")
    client.with_raw_response.create.side_effect = [raw_tool, raw_final]
    agent = ReactAgent(model=provider(client), order_service=demo, rag_tools=rag_tools())
    assert list(agent.execute_stream("查询订单", current_user_id=context_for(demo).current_user_id)) == ["订单状态为待支付。\n"]
    payloads = [call.kwargs for call in client.with_raw_response.create.call_args_list]
    assert all(payload["model"] == "qwen-max" for payload in payloads)
    assert {item["function"]["name"] for item in payloads[0]["tools"]} == {"get_order", "prepare_ticket", "rag_summarize"}
    observation = next(message for message in payloads[1]["messages"] if message["role"] == "tool")
    assert json.loads(observation["content"])["data"]["status"] == "pending_payment"


class StreamResponse:
    def __init__(self, deltas): self.deltas = deltas
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self):
        for delta in self.deltas:
            yield {"id": "synthetic-stream", "model": "qwen-max", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def test_actual_provider_streams_standard_message_chunks():
    client = Mock(); client.create.return_value = StreamResponse([
        {"role": "assistant", "content": "你好"}, {"content": "，欢迎。"},
    ])
    chunks = list(provider(client).stream("synthetic-question"))
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "你好，欢迎。"


def test_actual_provider_accumulates_streamed_tool_arguments():
    client = Mock(); client.create.return_value = StreamResponse([
        {"role": "assistant", "tool_calls": [{"index": 0, "id": "synthetic-call", "type": "function",
          "function": {"name": "get_order", "arguments": '{"order_no":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"DEMO-ORDER-A"}'}}]},
    ])
    chunks = list(provider(client).bind_tools([get_order]).stream("synthetic-question"))
    merged = chunks[0]
    for chunk in chunks[1:]: merged += chunk
    assert merged.tool_calls[0]["name"] == "get_order"
    assert merged.tool_calls[0]["args"] == {"order_no": "DEMO-ORDER-A"}


def test_actual_provider_failure_stays_safe(demo, caplog):
    client = Mock(); client.with_raw_response.create.side_effect = RuntimeError(SECRET)
    agent = ReactAgent(model=provider(client), order_service=demo, rag_tools=rag_tools())
    with pytest.raises(AIServiceUnavailable) as caught:
        list(agent.execute_stream("synthetic-question", current_user_id=context_for(demo).current_user_id))
    assert SECRET not in str(caught.value) + caplog.text
    assert client.with_raw_response.create.call_count == 1


@pytest.mark.parametrize("name", ["openai._base_client", "langchain_openai.chat_models.base", "httpx2", "httpcore.http11"])
def test_new_provider_logs_do_not_expose_secrets(name, caplog):
    from agent.errors import install_provider_log_filters
    install_provider_log_filters()
    with caplog.at_level(logging.DEBUG, logger=name):
        logging.getLogger(name).debug("Authorization: %s", SECRET, exc_info=(ValueError, ValueError(SECRET), None))
    assert SECRET not in caplog.text and "Authorization" not in caplog.text
