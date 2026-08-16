"""Regression tests for optional-dependency import boundaries."""

from __future__ import annotations

import builtins
import importlib

import pytest


@pytest.mark.parametrize(
    ("package_name", "broken_import"),
    [
        ("chronofy.embedding", "chronofy.embedding.fine_tuner"),
        ("chronofy.retrieval", "chronofy.retrieval.graph"),
        ("chronofy.sl", "chronofy.sl.conflict"),
    ],
)
def test_internal_import_errors_are_not_masked(
    package_name: str,
    broken_import: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only absent extras may trigger graceful degradation or install hints."""
    package = importlib.import_module(package_name)
    real_import = builtins.__import__

    def fail_internal_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == broken_import:
            raise ImportError(f"internal defect in {broken_import}")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", fail_internal_import)
            with pytest.raises(ImportError, match="internal defect"):
                importlib.reload(package)
    finally:
        # A failed reload retains the same module object. Restore its complete
        # public namespace for tests collected after this one.
        importlib.reload(package)
