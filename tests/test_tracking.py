from pathlib import Path
import sys

from src.common import tracking


def test_init_run_defaults_to_offline(monkeypatch, tmp_path):
    calls = {}

    class FakeWandb:
        def init(self, **kwargs):
            calls.update(kwargs)
            return object()

    monkeypatch.setattr(tracking, "Path", lambda *args: tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
    monkeypatch.delenv("WANDB_MODE", raising=False)
    tracking.init_run(project="fedguard", name="test-run", config={"seed": 42})
    assert calls["mode"] == "offline"
    assert calls["project"] == "fedguard"


def test_wandb_directory_is_ignored():
    assert "wandb/" in Path(".gitignore").read_text(encoding="utf-8")

