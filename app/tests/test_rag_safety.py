"""Offline RAG boundaries: no real Chroma, embedding or model requests."""

import logging
from pathlib import Path
import runpy
import sys
from types import ModuleType
from unittest.mock import ANY, Mock

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import pytest

from rag.rag_service import (
    RagSummarizeService, RAGRetrievalUnavailable, NO_KNOWLEDGE_MESSAGE,
    RETRIEVAL_UNAVAILABLE_MESSAGE, GENERATION_UNAVAILABLE_MESSAGE,
)
from utils import file_handler


PRIVATE = "SYNTHETIC_QUESTION DOCUMENT_TEXT API_KEY_MARKER Authorization: PRIVATE_TOKEN C:/private/path"


@pytest.fixture
def importer(tmp_path, monkeypatch):
    model = ModuleType("model.factory")
    model.embed_model = Mock()
    monkeypatch.setitem(sys.modules, "model.factory", model)
    exports = runpy.run_path(str(Path(__file__).resolve().parents[1] / "rag/vector_store.py"))
    cls = exports["SectorStoreService"]
    scope = cls.load_document.__globals__
    monkeypatch.setitem(scope, "chroma_conf", {
        "data_path": str(tmp_path / "knowledge"), "md5_hex_store": str(tmp_path / "md5.txt"),
        "allow_knowledge_type": ["txt", "pdf"],
    })
    monkeypatch.setitem(scope, "get_abs_path", lambda path: path)
    service = cls.__new__(cls)
    service.vector_store = Mock()
    indexed = {}
    service.vector_store.get.side_effect = lambda **kwargs: {
        "ids": [key for key in kwargs.get("ids", []) if key in indexed],
        "documents": [indexed[key] for key in kwargs.get("ids", []) if key in indexed],
    }
    def add_documents(documents, *, ids):
        indexed.update({key: doc.page_content for key, doc in zip(ids, documents)})
    service.vector_store.add_documents.side_effect = add_documents
    service.spliter = Mock()
    return service, scope, tmp_path


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_directory_has_no_import_side_effects(importer, exists, caplog):
    service, scope, root = importer
    directory = root / "knowledge"
    if exists:
        directory.mkdir()
    with caplog.at_level(logging.INFO):
        assert file_handler.listdir_with_allowed_type(str(directory), ("txt", "pdf")) == []
        service.load_document()
    assert directory.exists() is exists
    assert not (root / "md5.txt").exists()
    service.vector_store.add_documents.assert_not_called()
    service.spliter.split_documents.assert_not_called()
    assert str(root) not in caplog.text
    assert "files=0" in caplog.text


def test_missing_sensitive_directory_does_not_expose_path(tmp_path, caplog):
    directory = tmp_path / "SYNTHETIC_PRIVATE_CUSTOMER_DIRECTORY"
    assert file_handler.listdir_with_allowed_type(str(directory), ("txt",)) == []
    assert not directory.exists()
    assert str(tmp_path) not in caplog.text
    assert "SYNTHETIC_PRIVATE_CUSTOMER_DIRECTORY" not in caplog.text


def test_directory_scan_only_returns_existing_supported_files(tmp_path):
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.txt").touch()
    (tmp_path / "ignored.csv").touch()
    (tmp_path / "directory.txt").mkdir()
    assert file_handler.listdir_with_allowed_type(str(tmp_path), ("txt", "pdf")) == [
        str(tmp_path / "a.txt"), str(tmp_path / "b.pdf"),
    ]


def test_unreadable_directory_is_safe(tmp_path, monkeypatch, caplog):
    with monkeypatch.context() as patch:
        patch.setattr(file_handler.os, "listdir", Mock(side_effect=PermissionError(PRIVATE)))
        assert file_handler.listdir_with_allowed_type(str(tmp_path), ("txt",)) == []
    assert PRIVATE not in caplog.text
    assert "knowledge_directory_read_failed" in caplog.text


def test_unreadable_hash_is_safe(tmp_path, monkeypatch, caplog):
    path = tmp_path / "synthetic.txt"
    path.touch()
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", Mock(side_effect=OSError(PRIVATE)))
        assert file_handler.get_file_md5_hex(str(path)) is None
    assert PRIVATE not in caplog.text and str(path) not in caplog.text


def test_hash_failure_does_not_embed(importer, monkeypatch):
    service, scope, root = importer
    directory = root / "knowledge"
    directory.mkdir()
    (directory / "synthetic.txt").touch()
    monkeypatch.setitem(scope, "get_file_md5_hex", lambda path: None)
    service.load_document()
    service.vector_store.add_documents.assert_not_called()
    assert not (root / "md5.txt").exists()


@pytest.mark.parametrize("failure_stage", ["loading", "indexing", "deduplication"])
def test_import_errors_do_not_log_content_paths_or_tracebacks(importer, monkeypatch, caplog, capsys, failure_stage):
    service, scope, root = importer
    directory = root / "knowledge"
    directory.mkdir()
    (directory / "synthetic.txt").write_text(PRIVATE, encoding="utf-8")
    docs = [Document(page_content=PRIVATE, metadata={"source": PRIVATE})]
    loader = Mock(return_value=docs)
    monkeypatch.setitem(scope, "txt_loader", loader)
    service.spliter.split_documents.return_value = docs
    if failure_stage == "loading": loader.side_effect = RuntimeError(PRIVATE)
    elif failure_stage == "indexing": service.vector_store.add_documents.side_effect = RuntimeError(PRIVATE)
    else: (root / "md5.txt").mkdir()  # Opening this as a file fails inside the boundary.
    with caplog.at_level(logging.INFO): service.load_document()
    assert "knowledge_import_failed" in caplog.text
    assert PRIVATE not in caplog.text and str(root) not in caplog.text
    assert not capsys.readouterr().out
    assert all(record.exc_info is None for record in caplog.records)


def test_successful_import_and_deduplication_keep_contents_out_of_logs(importer, monkeypatch, caplog):
    service, scope, root = importer
    directory = root / "knowledge"
    directory.mkdir()
    (directory / "synthetic.txt").write_text(PRIVATE, encoding="utf-8")
    docs = [Document(page_content=PRIVATE)]
    monkeypatch.setitem(scope, "txt_loader", Mock(return_value=docs))
    service.spliter.split_documents.return_value = docs
    with caplog.at_level(logging.INFO):
        service.load_document()
        service.load_document()
    service.vector_store.add_documents.assert_called_once_with(docs, ids=ANY)
    assert "knowledge_import_succeeded" in caplog.text
    assert "knowledge_import_skipped" in caplog.text
    assert PRIVATE not in caplog.text and str(root) not in caplog.text


def make_rag(documents=None, error=None, generation_error=None):
    retriever = Mock()
    retriever.invoke.return_value = documents if documents is not None else []
    retriever.invoke.side_effect = error
    generation = Mock(return_value="根据知识资料给出的虚构回答。", side_effect=generation_error)
    service = RagSummarizeService(retriever=retriever, model=RunnableLambda(generation))
    return service, retriever, generation


def test_empty_retrieval_does_not_generate(caplog, capsys):
    service, retriever, generation = make_rag()
    with caplog.at_level(logging.INFO): answer = service.rag_summarize(PRIVATE)
    assert answer == NO_KNOWLEDGE_MESSAGE
    retriever.invoke.assert_called_once_with(PRIVATE)
    generation.assert_not_called()
    assert PRIVATE not in caplog.text
    assert "documents=0" in caplog.text
    assert not capsys.readouterr().out


def test_retriever_exception_is_distinct_and_safe(caplog, capsys):
    service, retriever, generation = make_rag(error=RuntimeError(PRIVATE))
    assert service.rag_summarize(PRIVATE) == RETRIEVAL_UNAVAILABLE_MESSAGE
    assert RETRIEVAL_UNAVAILABLE_MESSAGE != NO_KNOWLEDGE_MESSAGE
    generation.assert_not_called()
    assert PRIVATE not in caplog.text
    assert "stage=retrieval" in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
    assert not capsys.readouterr().out
    with pytest.raises(RAGRetrievalUnavailable) as caught:
        service.retriever_docs(PRIVATE)
    assert str(caught.value) == RETRIEVAL_UNAVAILABLE_MESSAGE
    assert caught.value.__suppress_context__


def test_normal_retrieval_preserves_prompt_sources_and_answer(caplog, capsys):
    docs = [Document(page_content="DOCUMENT_TEXT", metadata={"source": "PRIVATE_SOURCE"})]
    service, retriever, generation = make_rag(documents=docs)
    with caplog.at_level(logging.INFO): answer = service.rag_summarize(PRIVATE)
    assert answer == "根据知识资料给出的虚构回答。"
    prompt = generation.call_args.args[0].to_string()
    assert PRIVATE in prompt and "DOCUMENT_TEXT" in prompt and "PRIVATE_SOURCE" in prompt
    assert "[参考资料]:1" in prompt and "[参考资料来源]" in prompt
    assert PRIVATE not in caplog.text and "DOCUMENT_TEXT" not in caplog.text and "PRIVATE_SOURCE" not in caplog.text
    assert "documents=1" in caplog.text
    assert not capsys.readouterr().out


def test_generation_exception_is_safe(caplog):
    service, _, generation = make_rag(documents=[Document(page_content=PRIVATE)], generation_error=RuntimeError(PRIVATE))
    assert service.rag_summarize(PRIVATE) == GENERATION_UNAVAILABLE_MESSAGE
    generation.assert_called_once()
    assert PRIVATE not in caplog.text and "stage=generation" in caplog.text


@pytest.mark.parametrize("error", [False, True])
def test_real_rag_tool_returns_safe_empty_or_error_message(monkeypatch, error):
    service, _, generation = make_rag(error=RuntimeError(PRIVATE) if error else None)
    module = ModuleType("rag.rag_service")
    module.RagSummarizeService = lambda: service
    monkeypatch.setitem(sys.modules, "rag.rag_service", module)
    exports = runpy.run_path(str(Path(__file__).resolve().parents[1] / "agent/tools/agent_tools.py"))
    result = exports["rag_summarize"].invoke({"query": PRIVATE})
    assert result == (RETRIEVAL_UNAVAILABLE_MESSAGE if error else NO_KNOWLEDGE_MESSAGE)
    generation.assert_not_called()


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_embedding_sdk_logs_are_redacted(level, caplog):
    with caplog.at_level(level, logger="langchain_community.embeddings.dashscope"):
        logging.getLogger("langchain_community.embeddings.dashscope").log(
            level, "%s", PRIVATE, exc_info=(RuntimeError, RuntimeError(PRIVATE), None), stack_info=True,
        )
    assert PRIVATE not in caplog.text
    for marker in ["SYNTHETIC_QUESTION", "DOCUMENT_TEXT", "API_KEY_MARKER", "PRIVATE_TOKEN", "C:/private/path"]:
        assert marker not in caplog.text
    assert all(r.exc_info is None and r.stack_info is None and not r.args for r in caplog.records)


@pytest.mark.parametrize("loader", ["load_rag_prompt", "load_system_prompt"])
def test_prompt_loading_error_is_safe(monkeypatch, caplog, loader):
    from utils import prompt_loader
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", Mock(side_effect=OSError(PRIVATE)))
        with pytest.raises(RuntimeError) as caught: getattr(prompt_loader, loader)()
    assert PRIVATE not in str(caught.value) + caplog.text
    assert caught.value.__suppress_context__
