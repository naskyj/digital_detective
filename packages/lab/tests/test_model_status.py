"""A configured relevance model that cannot load must say so.

Both ONNX model classes swallow every load failure and fall back to returning
None per artifact. That is right at runtime — a missing model must never abort
a case — but it makes a broken model invisible: classification still runs,
every artifact still gets a category, and the relevance model contributes
nothing to any of it while the settings page shows a configured path.

That is not hypothetical. It was reported from a real session as "the
artifacts are classified and categorized even though there are no models",
and two distinct causes have been hit:

- `onnxruntime` missing, because the `classify` extra does not install on
  every platform;
- `tokenizers` missing, because it used to be satisfied only transitively via
  gliner, so installing the relevance model's own extra was not sufficient.
  The ONNX session opens and every score is silently None.

These tests pin the diagnosis, not the loading: `model_status` checks
preconditions so the settings endpoint can call it on every request.
"""

from __future__ import annotations

import builtins
import json

import pytest

from dd_lab.classify.model import model_status


def test_no_directory_configured_is_reported_as_rules_only(tmp_path):
    """Not an error — the supported offline default (R1). But it must be
    visible, because it changes what the numbers mean."""
    for value in ("", None):
        st = model_status(value)
        assert st["configured"] is False
        assert st["usable"] is False
        assert "rules" in st["reason"]


def test_a_missing_directory_names_the_path(tmp_path):
    st = model_status(tmp_path / "nope")
    assert st["configured"] is True and st["usable"] is False
    assert "not found" in st["reason"] and "nope" in st["reason"]


def test_a_directory_missing_the_model_file_says_which(tmp_path):
    d = tmp_path / "relevance"
    d.mkdir()
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    st = model_status(d)
    assert st["usable"] is False
    assert "model.onnx" in st["reason"]
    assert "tokenizer.json" not in st["reason"], "only the MISSING file is named"


def test_a_directory_missing_the_tokenizer_says_which(tmp_path):
    d = tmp_path / "relevance"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"not really onnx")
    st = model_status(d)
    assert st["usable"] is False
    assert "tokenizer.json" in st["reason"]


@pytest.mark.parametrize("absent", ["onnxruntime", "tokenizers"])
def test_a_missing_runtime_dependency_is_named_with_the_fix(tmp_path, monkeypatch, absent):
    """The two real failures. The message has to name the package AND the
    command, because the person hitting this is an examiner setting the tool
    up on their own machine, not its author."""
    d = tmp_path / "relevance"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"x")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "labels.json").write_text(json.dumps(["not_relevant", "relevant"]), encoding="utf-8")

    real_import = builtins.__import__

    def _fake(name, *a, **kw):
        if name == absent:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake)
    st = model_status(d)
    assert st["configured"] is True and st["usable"] is False
    assert absent in st["reason"]
    assert "uv sync" in st["reason"] and "--extra classify" in st["reason"]


def test_a_complete_directory_with_the_deps_present_is_usable(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    d = tmp_path / "relevance"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"x")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    st = model_status(d)
    assert st["usable"] is True and st["reason"] == "loaded"


def test_the_settings_endpoint_exposes_it(tmp_path, monkeypatch):
    """It has to reach the UI, or it diagnoses nothing."""
    from fastapi.testclient import TestClient

    import dd_lab.api.main as main

    monkeypatch.setattr(main._settings, "relevance_model_dir", str(tmp_path / "gone"))
    body = TestClient(main.app).get("/api/settings").json()
    assert "relevance_model" in body, "the settings payload lost the status field"
    assert body["relevance_model"]["usable"] is False
    assert "not found" in body["relevance_model"]["reason"]
