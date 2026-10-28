"""Locate resources independently of the process working directory."""

from pathlib import Path


def get_project_root() -> str:
    return str(Path(__file__).absolute().parents[1])


def get_abs_path(relative_path: str) -> str:
    """Accept absolute paths too; resolving a resource never touches the disk."""
    return str(Path(get_project_root()) / relative_path)
