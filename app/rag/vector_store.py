import os
import hashlib
from dataclasses import dataclass
from pathlib import Path

from agent.errors import install_provider_log_filters
from utils.logger_handler import logger
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config_handler import chroma_conf
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.path_tool import get_abs_path


install_provider_log_filters()


@dataclass
class ImportResult:
    files: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_added: int = 0


class SectorStoreService:
    def __init__(self, *, embedding=None, persist_directory=None, collection_name=None):
        if embedding is None:
            from model.factory import embed_model
            embedding = embed_model
        self.vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory or get_abs_path(chroma_conf["persist_directory"]),
            collection_name=collection_name or chroma_conf["collection_name"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        """Build a similarity-search adapter with an independent configuration snapshot."""
        tags = [type(self.vector_store).__name__]
        embedding = self.vector_store.embeddings
        if embedding:
            tags.append(type(embedding).__name__)
        return VectorStoreRetriever(
            vectorstore=self.vector_store,
            search_type="similarity",
            search_kwargs=dict(k=chroma_conf["k"]),
            tags=tags,
        )

    def load_document(self, *, source_dir=None, md5_store=None):
        """Import existing TXT/PDF files only; never create the knowledge directory."""
        files = listdir_with_allowed_type(
            str(source_dir) if source_dir is not None else get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_type"]),
        )
        result = ImportResult(files=len(files))
        if not files:
            logger.info("knowledge_import_empty stage=scan files=0")
            return result
        md5_path = str(md5_store) if md5_store is not None else get_abs_path(chroma_conf["md5_hex_store"])
        for path in files:
            try:
                digest = get_file_md5_hex(path)
                if digest is None:
                    result.failed += 1
                    continue
                known_hashes = set()
                if os.path.exists(md5_path):
                    with open(md5_path, encoding="utf-8") as source:
                        known_hashes = {line.strip() for line in source}
                if path.endswith(".txt"):
                    documents = txt_loader(path)
                elif path.endswith(".pdf"):
                    documents = pdf_loader(path)
                else:
                    result.failed += 1
                    continue
                if not documents:
                    logger.warning("knowledge_import_empty stage=loading documents=0")
                    result.failed += 1
                    continue
                chunks = self.spliter.split_documents(documents)
                if not chunks:
                    logger.warning("knowledge_import_empty stage=splitting documents=0")
                    result.failed += 1
                    continue
                ids = ["servicepilot-" + hashlib.sha256(
                    f"v1:{digest}:{index}:{chunk.page_content}".encode("utf-8")
                ).hexdigest() for index, chunk in enumerate(chunks)]
                stored = self.vector_store.get(ids=ids, include=["documents"])
                existing = dict(zip(stored["ids"], stored["documents"]))
                if any(identity in existing and existing[identity] != chunk.page_content
                       for identity, chunk in zip(ids, chunks)):
                    raise ValueError("Reserved chunk identity conflict")
                missing = [(identity, chunk) for identity, chunk in zip(ids, chunks) if identity not in existing]
                # Legacy random-ID imports may already exist. Verify their actual
                # content in this collection rather than trusting the sidecar alone.
                if missing and digest in known_hashes:
                    legacy = self.vector_store.get(where={"source": str(Path(path))}, include=["documents"])
                    if all(chunk.page_content in legacy["documents"] for chunk in chunks):
                        missing = []
                if missing:
                    self.vector_store.add_documents([chunk for _, chunk in missing], ids=[identity for identity, _ in missing])
                    result.imported += 1
                    result.chunks_added += len(missing)
                else:
                    result.skipped += 1
                    logger.info("knowledge_import_skipped stage=deduplication")
                if digest not in known_hashes:
                    with open(md5_path, "a", encoding="utf-8") as destination:
                        destination.write(digest + "\n")
                logger.info("knowledge_import_succeeded stage=indexing documents=%d", len(missing))
            except Exception:
                result.failed += 1
                logger.error("knowledge_import_failed stage=import details_omitted")
        return result


if __name__ == '__main__':
    try:
        SectorStoreService().load_document()
    except Exception:
        logger.error("knowledge_import_failed stage=initialization details_omitted")
