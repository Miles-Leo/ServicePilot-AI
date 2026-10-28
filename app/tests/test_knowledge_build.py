"""Sample/CLI contracts and real temporary Chroma with offline embeddings."""

import hashlib
from pathlib import Path
from types import ModuleType
import sys
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from rag.vector_store import ImportResult, SectorStoreService
from scripts.build_knowledge_base import PROJECT_ROOT, build_parser, main


class OfflineEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(token in text) for token in ["return_request", "exchange_request", "repair_request", "pending_review"]] + [0.1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "false")
    return SectorStoreService(embedding=OfflineEmbedding(), persist_directory=str(tmp_path / "chroma_db"), collection_name="synthetic-samples")


def test_original_sample_files_exist_and_are_readable():
    files = sorted((PROJECT_ROOT / "sample_knowledge").glob("*.txt"))
    assert len(files) == 5
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "原创示例资料" in text and "DEMO" in text
        assert "ServicePilot" in text


def test_cli_defaults_and_help(capsys):
    args = build_parser().parse_args([])
    assert args.source == PROJECT_ROOT / "sample_knowledge" and not args.yes
    with pytest.raises(SystemExit) as result: main(["--help"])
    assert result.value.code == 0
    assert "--source" in capsys.readouterr().out


def test_cli_requires_explicit_fee_acknowledgement(capsys):
    assert main([]) == 2
    assert "费用" in capsys.readouterr().out


@pytest.mark.parametrize("key", [None, "", " ", "REPLACE_WITH_YOUR_DASHSCOPE_API_KEY"])
def test_missing_key_rejected_before_import(key, monkeypatch, capsys):
    if key is None: monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    else: monkeypatch.setenv("DASHSCOPE_API_KEY", key)
    factory = Mock(side_effect=AssertionError("Must not initialize storage"))
    module = ModuleType("rag.vector_store")
    module.SectorStoreService = factory
    monkeypatch.setitem(sys.modules, "rag.vector_store", module)
    assert main(["--yes"]) == 2
    assert "DASHSCOPE_API_KEY" in capsys.readouterr().out
    factory.assert_not_called()


@pytest.mark.parametrize("exists", [False, True])
def test_no_files_exits_without_storage_creation(tmp_path, monkeypatch, capsys, exists):
    source = tmp_path / "PRIVATE_EMPTY_DIRECTORY"
    if exists: source.mkdir()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "SYNTHETIC_KEY_NEVER_SENT")
    factory = Mock(side_effect=AssertionError("Must not initialize storage"))
    module = ModuleType("rag.vector_store"); module.SectorStoreService = factory
    monkeypatch.setitem(sys.modules, "rag.vector_store", module)
    assert main(["--yes", "--source", str(source)]) == 2
    output = capsys.readouterr().out
    assert "没有支持" in output and str(tmp_path) not in output
    assert source.exists() is exists
    factory.assert_not_called()


def test_cli_reuses_default_store_and_passes_sample_source(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "SYNTHETIC_KEY_NEVER_SENT")
    service = Mock(); service.load_document.return_value = ImportResult(files=5, imported=5, chunks_added=10)
    module = ModuleType("rag.vector_store"); module.SectorStoreService = Mock(return_value=service)
    monkeypatch.setitem(sys.modules, "rag.vector_store", module)
    assert main(["--yes"]) == 0
    module.SectorStoreService.assert_called_once_with()
    service.load_document.assert_called_once_with(source_dir=PROJECT_ROOT / "sample_knowledge")
    assert "新增分片=10" in capsys.readouterr().out


def test_cli_does_not_expose_exception_or_key(monkeypatch, capsys, caplog):
    secret = "SYNTHETIC_PRIVATE_KEY Authorization: PRIVATE_HEADER C:/PRIVATE_PATH"
    monkeypatch.setenv("DASHSCOPE_API_KEY", secret)
    module = ModuleType("rag.vector_store"); module.SectorStoreService = Mock(side_effect=RuntimeError(secret))
    monkeypatch.setitem(sys.modules, "rag.vector_store", module)
    assert main(["--yes"]) == 1
    output = capsys.readouterr().out + caplog.text
    for marker in [secret, "PRIVATE_HEADER", "PRIVATE_PATH", "SYNTHETIC_PRIVATE_KEY"]: assert marker not in output


def test_invalid_cli_argument_does_not_echo_value(capsys):
    with pytest.raises(SystemExit) as result: main(["--PRIVATE_SECRET_ARGUMENT"])
    assert result.value.code == 2
    assert "PRIVATE_SECRET" not in capsys.readouterr().err


def test_real_chroma_sample_import_retrieval_and_repeat(store, tmp_path):
    ledger = tmp_path / "md5.txt"
    store.vector_store.add_documents([Document(page_content="Synthetic preexisting record", metadata={"source": "synthetic-existing"})], ids=["synthetic-existing"])
    before = store.vector_store.get(ids=["synthetic-existing"], include=["documents", "metadatas"])
    first = store.load_document(source_dir=PROJECT_ROOT / "sample_knowledge", md5_store=ledger)
    count = len(store.vector_store.get()["ids"])
    second = store.load_document(source_dir=PROJECT_ROOT / "sample_knowledge", md5_store=ledger)
    assert first.imported == 5 and first.failed == 0
    assert second.skipped == 5 and second.failed == 0 and second.chunks_added == 0
    assert len(store.vector_store.get()["ids"]) == count
    assert store.vector_store.get(ids=["synthetic-existing"], include=["documents", "metadatas"]) == before
    docs = store.get_retriever().invoke("exchange_request")
    assert any("exchange_request" in doc.page_content for doc in docs)


def test_stale_ledger_does_not_skip_empty_collection(store, tmp_path):
    source = tmp_path / "source"; source.mkdir()
    path = source / "synthetic.txt"; path.write_text("Synthetic return_request rule", encoding="utf-8")
    ledger = tmp_path / "md5.txt"
    ledger.write_text(hashlib.md5(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    result = store.load_document(source_dir=source, md5_store=ledger)
    assert result.imported == 1 and result.failed == 0


def test_retry_after_sidecar_failure_does_not_duplicate(store, tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "synthetic.txt").write_text("Synthetic repair_request rule", encoding="utf-8")
    result = store.load_document(source_dir=source, md5_store=tmp_path / "missing-parent" / "md5.txt")
    assert result.failed == 1
    count = len(store.vector_store.get()["ids"])
    retry = store.load_document(source_dir=source, md5_store=tmp_path / "md5.txt")
    assert retry.failed == 0 and retry.skipped == 1
    assert len(store.vector_store.get()["ids"]) == count


def test_legacy_random_ids_are_preserved_and_not_reinserted(store, tmp_path):
    from utils.file_handler import txt_loader
    source = tmp_path / "source"; source.mkdir()
    path = source / "synthetic.txt"; path.write_text("Synthetic legacy return_request", encoding="utf-8")
    chunks = store.spliter.split_documents(txt_loader(str(path)))
    store.vector_store.add_documents(chunks, ids=["synthetic-legacy"])
    ledger = tmp_path / "md5.txt"; ledger.write_text(hashlib.md5(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    assert store.load_document(source_dir=source, md5_store=ledger).skipped == 1
    assert store.vector_store.get()["ids"] == ["synthetic-legacy"]


def test_changed_source_adds_without_deleting_old_chunks(store, tmp_path):
    source = tmp_path / "source"; source.mkdir()
    path = source / "synthetic.txt"; path.write_text("Synthetic original rule", encoding="utf-8")
    ledger = tmp_path / "md5.txt"
    store.load_document(source_dir=source, md5_store=ledger)
    before = set(store.vector_store.get()["ids"])
    path.write_text("Synthetic changed rule", encoding="utf-8")
    store.load_document(source_dir=source, md5_store=ledger)
    assert before < set(store.vector_store.get()["ids"])


def test_reserved_id_with_different_content_is_not_overwritten(store, tmp_path):
    source = tmp_path / "source"; source.mkdir()
    content = "Synthetic reserved ID rule"
    path = source / "synthetic.txt"; path.write_text(content, encoding="utf-8")
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    identity = "servicepilot-" + hashlib.sha256(f"v1:{digest}:0:{content}".encode()).hexdigest()
    store.vector_store.add_documents([Document(page_content="Synthetic preexisting conflicting content")], ids=[identity])
    before = store.vector_store.get(ids=[identity], include=["documents", "metadatas"])
    assert store.load_document(source_dir=source, md5_store=tmp_path / "md5.txt").failed == 1
    assert store.vector_store.get(ids=[identity], include=["documents", "metadatas"]) == before


def test_default_store_uses_current_config(monkeypatch):
    import rag.vector_store as module
    from utils.path_tool import get_abs_path
    factory = Mock()
    monkeypatch.setattr(module, "Chroma", factory)
    embedding = OfflineEmbedding()
    SectorStoreService(embedding=embedding)
    factory.assert_called_once_with(embedding_function=embedding,
        persist_directory=get_abs_path(module.chroma_conf["persist_directory"]),
        collection_name=module.chroma_conf["collection_name"])


def test_partial_failure_cli_returns_failure(monkeypatch, capsys):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "SYNTHETIC_KEY_NEVER_SENT")
    service = Mock(); service.load_document.return_value = ImportResult(files=5, imported=4, failed=1)
    module = ModuleType("rag.vector_store"); module.SectorStoreService = Mock(return_value=service)
    monkeypatch.setitem(sys.modules, "rag.vector_store", module)
    assert main(["--yes"]) == 1
    assert "部分资料导入失败" in capsys.readouterr().out
