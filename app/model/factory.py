from abc import ABC, abstractmethod
from typing import Optional
import os
from urllib.parse import urlsplit

from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from utils.config_handler import rag_conf
from agent.errors import install_provider_log_filters


install_provider_log_filters()
DEFAULT_CHAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _chat_credentials() -> tuple[str, str]:
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key or key == "REPLACE_WITH_YOUR_DASHSCOPE_API_KEY":
        raise ValueError("DASHSCOPE_API_KEY 未配置。")
    base_url = os.getenv("DASHSCOPE_CHAT_BASE_URL", DEFAULT_CHAT_BASE_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme == "https" and parsed.hostname is not None
            and parsed.hostname.endswith(".aliyuncs.com")
            and parsed.path == "/compatible-mode/v1"
            and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and parsed.port in (None, 443)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("DASHSCOPE_CHAT_BASE_URL 不是有效的 DashScope HTTPS 兼容接口地址。")
    return key, base_url


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        key, base_url = _chat_credentials()
        return init_chat_model(
            model=rag_conf["chat_model_name"], model_provider="openai",
            api_key=key, base_url=base_url, use_responses_api=False,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model = rag_conf["embedding_model_name"])


chat_model =ChatModelFactory().generator()
embed_model =EmbeddingsFactory().generator()
