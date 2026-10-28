"""Resource compatibility and metadata-only application logging, entirely offline."""

from datetime import datetime
import importlib
import logging
from pathlib import Path
import runpy

import pytest

APP = Path(__file__).resolve().parents[1]
_paths = runpy.run_path(str(APP / 'utils/path_tool.py'))
get_abs_path = _paths['get_abs_path']
get_project_root = _paths['get_project_root']


@pytest.mark.parametrize('resource', ['config/rag.yml', 'config/chroma.yml', 'config/prompts.yml', 'prompts/main_prompt.txt', 'prompts/rag_summarize.txt', 'rag/chroma_db', 'logs', 'md5.txt'])
def test_paths_keep_application_resource_locations(resource, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_project_root() == str(APP)
    assert isinstance(get_project_root(), str)
    assert get_abs_path(resource) == str(APP / resource)


def test_absolute_path_is_preserved_without_creating_it(tmp_path):
    missing = tmp_path / 'not-created' / 'resource.txt'
    assert get_abs_path(str(missing)) == str(missing)
    assert not missing.parent.exists()


def test_empty_path_returns_app_root():
    assert get_abs_path('') == str(APP)


@pytest.fixture
def diagnostics(tmp_path, monkeypatch):
    from io import StringIO
    import utils.path_tool as paths
    target = logging.Logger('agent')
    retained_filter = logging.Filter()
    target.addFilter(retained_filter)
    monkeypatch.setattr(paths, 'get_abs_path', lambda path: str(tmp_path / path))
    original = logging.getLogger
    monkeypatch.setattr(logging, 'getLogger', lambda name=None: target if name == 'agent' else original(name))
    module = runpy.run_path(str(APP / 'utils/logger_handler.py'))
    target._test_console = StringIO()
    for handler in target.handlers:
        if getattr(handler, '_servicepilot_sink', None) == 'console':
            handler.setStream(target._test_console)
    yield module, target, tmp_path, retained_filter
    for handler in target.handlers[:]:
        handler.close()
        target.removeHandler(handler)


def test_repeat_initialization_preserves_handlers_and_filters(diagnostics):
    module, logger, root, retained = diagnostics
    handlers = tuple(logger.handlers)
    filters = tuple(logger.filters)
    for _ in range(3):
        assert module['get_logger']() is logger
    assert tuple(logger.handlers) == handlers
    assert tuple(logger.filters) == filters
    assert retained in logger.filters
    assert len(handlers) == 2


def test_console_and_utf8_daily_file_receive_safe_diagnostics(diagnostics):
    module, logger, root, _ = diagnostics
    logger.info('订单工具执行失败，内部详情已隐藏')
    for handler in logger.handlers:
        handler.flush()
    assert '订单工具执行失败' in logger._test_console.getvalue()
    destination = root / 'logs' / f'agent_{datetime.now():%Y%m%d}.log'
    assert '订单工具执行失败' in destination.read_text(encoding='utf-8')


@pytest.mark.parametrize('message', [
    'API_KEY=SYNTHETIC_SECRET', 'Authorization: Bearer SYNTHETIC_SECRET',
    '完整Prompt SYNTHETIC_SECRET', '用户地址电话 SYNTHETIC_SECRET',
    '知识库正文 SYNTHETIC_SECRET', 'rag_retrieved stage=generation documents=1 SYNTHETIC_SECRET',
])
def test_unknown_or_sensitive_content_is_not_written(diagnostics, message):
    module, logger, root, _ = diagnostics
    try:
        raise RuntimeError('SYNTHETIC_EXCEPTION_SECRET')
    except RuntimeError:
        logger.exception('%s', message, stack_info=True)
    for handler in logger.handlers:
        handler.flush()
    text = logger._test_console.getvalue() + next((root / 'logs').glob('*.log')).read_text(encoding='utf-8')
    assert 'application_event details_omitted' in text
    assert 'SYNTHETIC' not in text
    assert 'Traceback' not in text and str(APP) not in text
    assert message not in text


def test_existing_provider_filter_survives_initialization(diagnostics):
    module, logger, root, _ = diagnostics
    from agent.errors import _ProviderLogFilter
    existing = _ProviderLogFilter()
    logger.addFilter(existing)
    module['get_logger']()
    assert existing in logger.filters


def test_module_reexecution_does_not_duplicate_owned_handlers(diagnostics):
    module, logger, root, _ = diagnostics
    handlers = tuple(logger.handlers)
    runpy.run_path(str(APP / 'utils/logger_handler.py'))
    assert tuple(logger.handlers) == handlers
    assert sum(getattr(f, '_servicepilot_filter', None) is not None for f in logger.filters) == 1


def test_foreign_handler_is_preserved_and_cannot_receive_raw_body(diagnostics):
    from io import StringIO
    module, logger, root, _ = diagnostics
    buffer = StringIO()
    foreign = logging.StreamHandler(buffer)
    logger.addHandler(foreign)
    module['get_logger']()
    logger.info('SYNTHETIC_SECRET')
    assert foreign in logger.handlers and len(logger.handlers) == 3
    assert 'SYNTHETIC_SECRET' not in buffer.getvalue()


def test_agent_rag_and_business_import_without_model_initialization(monkeypatch):
    import sys
    from types import ModuleType
    fake = ModuleType('model.factory')
    monkeypatch.setitem(sys.modules, 'model.factory', fake)
    for name in ['agent.react_agent', 'rag.rag_service', 'business.services', 'utils.config_handler', 'utils.prompt_loader']:
        assert importlib.import_module(name) is not None
    from utils.config_handler import rag_conf, chroma_conf, prompts_conf
    assert rag_conf['chat_model_name'] == 'qwen-max'
    for path in [chroma_conf['persist_directory'], chroma_conf['data_path'], chroma_conf['md5_hex_store'], *prompts_conf.values()]:
        assert get_abs_path(path) == str(APP / path)
