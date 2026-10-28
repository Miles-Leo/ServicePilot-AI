"""Read application YAML without changing PyYAML's FullLoader semantics."""

from pathlib import Path

import yaml

from utils.path_tool import get_abs_path

_DEFAULT_PATHS = {name: get_abs_path(f"config/{name}.yml") for name in ('rag', 'chroma', 'prompts')}


def _read_configuration(path: str, encoding: str):
    # Keep the stream open while parsing so YAML errors retain their source marks.
    with Path(path).open(encoding=encoding) as stream:
        parser = yaml.FullLoader(stream)
        try:
            return parser.get_single_data()
        finally:
            parser.dispose()


def load_rag_config(config_path: str = _DEFAULT_PATHS['rag'], encoding: str = 'utf-8'):
    return _read_configuration(config_path, encoding)


def load_chroma_config(config_path: str = _DEFAULT_PATHS['chroma'], encoding: str = 'utf-8'):
    return _read_configuration(config_path, encoding)


def load_prompts_config(config_path: str = _DEFAULT_PATHS['prompts'], encoding: str = 'utf-8'):
    return _read_configuration(config_path, encoding)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
