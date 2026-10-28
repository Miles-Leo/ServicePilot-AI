"""Offline compatibility contracts for configuration, file loaders and retrieval."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from langchain_core.documents import Document
from pypdf import PdfWriter

from utils import config_handler as config
from utils import file_handler as files
from test_knowledge_build import store

APP = Path(__file__).resolve().parents[1]
LOADERS = [config.load_rag_config, config.load_chroma_config, config.load_prompts_config]


@pytest.mark.parametrize('name,loader', [('rag', LOADERS[0]), ('chroma', LOADERS[1]), ('prompts', LOADERS[2])])
def test_default_config_matches_legacy_parser_independent_of_cwd(name, loader, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with (APP / 'config' / f'{name}.yml').open(encoding='utf-8') as stream:
        expected = yaml.load(stream, Loader=yaml.FullLoader)
    assert loader() == expected
    assert loader.__defaults__[0] == str(APP / 'config' / f'{name}.yml')


@pytest.mark.parametrize('loader', LOADERS)
def test_custom_relative_config_full_loader_and_encoding(loader, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path('synthetic.yml')
    path.write_text('name: 虚构\nvalues: !!python/tuple [1, 2]\n', encoding='utf-16')
    assert loader(str(path), 'utf-16') == {'name': '虚构', 'values': (1, 2)}


@pytest.mark.parametrize('loader', LOADERS)
def test_missing_config_keeps_os_error(loader, tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        loader(str(tmp_path / 'missing.yml'))
    assert caught.value.filename == str(tmp_path / 'missing.yml')


@pytest.mark.parametrize('loader', LOADERS)
def test_yaml_exception_and_source_marks_are_unchanged(loader, tmp_path):
    path = tmp_path / 'invalid.yml'
    path.write_text('value: [unterminated', encoding='utf-8')
    with path.open(encoding='utf-8') as stream:
        with pytest.raises(yaml.YAMLError) as old:
            yaml.load(stream, Loader=yaml.FullLoader)
    with pytest.raises(type(old.value)) as new:
        loader(str(path))
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize('loader', LOADERS)
def test_empty_yaml_still_returns_none(loader, tmp_path):
    path = tmp_path / 'empty.yml'
    path.touch()
    assert loader(str(path)) is None


@pytest.mark.parametrize('loader', LOADERS)
def test_invalid_encoding_is_not_hidden(loader, tmp_path):
    path = tmp_path / 'encoding.yml'
    path.write_bytes(b'\xff\xfe\xfa')
    with pytest.raises(UnicodeDecodeError):
        loader(str(path))


def test_text_loader_matches_existing_api(tmp_path):
    path = tmp_path / 'example.txt'
    path.write_text('ServicePilot 虚构资料\n第二行', encoding='utf-8')
    expected = files.TextLoader(str(path), encoding='utf-8').load()
    assert files.txt_loader(str(path)) == expected


def test_pdf_loader_matches_existing_api_with_password(tmp_path):
    path = tmp_path / 'synthetic.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt('synthetic-demo-password')
    with path.open('wb') as output:
        writer.write(output)
    expected = files.PyPDFLoader(str(path), 'synthetic-demo-password').load()
    assert files.pdf_loader(str(path), 'synthetic-demo-password') == expected
    assert len(expected) == 1


@pytest.mark.parametrize('kind,options', [('text', {'encoding': 'utf-8'}), ('pdf', {'password': 'synthetic'})])
def test_loaders_consume_iterator_and_forward_options(kind, options, monkeypatch):
    docs = [Document(page_content='fictional', metadata={'page': 0})]
    reader = Mock()
    reader.lazy_load.return_value = iter(docs)
    constructor = Mock(return_value=reader)
    monkeypatch.setattr(files, 'TextLoader' if kind == 'text' else 'PyPDFLoader', constructor)
    result = files.txt_loader('fake.txt') if kind == 'text' else files.pdf_loader('fake.pdf', 'synthetic')
    constructor.assert_called_once_with(file_path='fake.txt' if kind == 'text' else 'fake.pdf', **options)
    assert result == docs and isinstance(result, list)


@pytest.mark.parametrize('kind', ['text', 'pdf'])
def test_lazy_read_errors_remain_inside_call_boundary(kind, monkeypatch):
    failure = OSError('synthetic-read-error')
    def broken():
        yield Document(page_content='partial')
        raise failure
    reader = Mock()
    reader.lazy_load.side_effect = broken
    monkeypatch.setattr(files, 'TextLoader' if kind == 'text' else 'PyPDFLoader', Mock(return_value=reader))
    with pytest.raises(OSError) as caught:
        (files.txt_loader if kind == 'text' else files.pdf_loader)('fake')
    assert caught.value is failure


@pytest.mark.parametrize('content', [b'', b'fictional-data', b'x' * 10000])
def test_file_digest_matches_previous_md5_contract(content, tmp_path):
    path = tmp_path / 'input.txt'
    path.write_bytes(content)
    assert files.get_file_md5_hex(str(path)) == hashlib.md5(content).hexdigest()


def test_retriever_matches_framework_defaults_and_tags(store):
    from utils.config_handler import chroma_conf
    expected = store.vector_store.as_retriever(search_kwargs={'k': chroma_conf['k']})
    actual = store.get_retriever()
    assert actual.vectorstore is store.vector_store
    assert actual.search_type == expected.search_type == 'similarity'
    assert actual.search_kwargs == expected.search_kwargs
    assert actual.tags == expected.tags


def test_retriever_configuration_is_independent_and_k_is_respected(store, monkeypatch):
    from utils.config_handler import chroma_conf
    monkeypatch.setitem(chroma_conf, 'k', 1)
    store.vector_store.add_documents([Document(page_content='return_request'), Document(page_content='exchange_request')], ids=['synthetic-one', 'synthetic-two'])
    first = store.get_retriever()
    assert len(first.invoke('return_request')) == 1
    monkeypatch.setitem(chroma_conf, 'k', 2)
    second = store.get_retriever()
    assert first.search_kwargs == {'k': 1}
    assert second.search_kwargs == {'k': 2}
    second.search_kwargs['k'] = 99
    assert first.search_kwargs == {'k': 1} and chroma_conf['k'] == 2
