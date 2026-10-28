"""Local knowledge inputs with safe diagnostics and eager document results."""

import hashlib
import os.path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger

def get_file_md5_hex(filepath: str):
    if not os.path.exists(filepath):
        logger.error("knowledge_file_missing stage=hash")
        return None

    if not os.path.isfile(filepath):
        logger.error("knowledge_not_a_file stage=hash")
        return

    try:
        with open(filepath, "rb") as source:
            return hashlib.file_digest(source, "md5").hexdigest()
    except Exception:
        logger.error("knowledge_file_hash_failed stage=hash details_omitted")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """Return supported regular files in deterministic order, or a safe empty result."""
    files = []

    if not os.path.isdir(path):
        logger.warning("knowledge_directory_unavailable stage=scan files=0")
        return []

    try:
        for f in sorted(os.listdir(path)):
            if f.endswith(allowed_types) and os.path.isfile(os.path.join(path, f)):
                files.append(os.path.join(path, f))
    except OSError:
        logger.error("knowledge_directory_read_failed stage=scan details_omitted")
        return []

    return files


def _document_list(loader_type, filepath: str, **options) -> list[Document]:
    # Consume the iterator here: callers receive documents or the loader exception,
    # never a deferred read that could fail outside the importer's error boundary.
    reader = loader_type(file_path=filepath, **options)
    return list(reader.lazy_load())


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return _document_list(PyPDFLoader, filepath, password=passwd)

def txt_loader(filepath: str) -> list[Document]:
    return _document_list(TextLoader, filepath, encoding="utf-8")
