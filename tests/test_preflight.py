"""Phase 6: adapter resolution against llama-server, preflight checks."""

import json

import pytest

import orchestrator as orch
import preflight
import tags


class Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


@pytest.fixture
def reg(monkeypatch):
    state = {"adapters": [{"version": "n1", "gguf": "adapters/n1/adapter.gguf", "status": "promoted",
                           "grammar_hash": tags.grammar_hash()},
                          {"version": "n2", "gguf": "adapters/n2/adapter.gguf", "status": "candidate"}],
             "current": "n1"}
    monkeypatch.setattr(orch, "_read_registry", lambda: state)
    monkeypatch.setattr(orch, "_lora_state", {"key": object(), "field": None, "applied": None})
    monkeypatch.setattr(orch, "REACTION_TRANSPORT", "completion")
    return state


def test_unique_gguf_names_map_exactly(reg, monkeypatch):
    reg["adapters"][0]["gguf"] = "adapters/n1/n1.gguf"
    reg["adapters"][1]["gguf"] = "adapters/n2/n2.gguf"
    monkeypatch.setattr(orch.requests, "get", lambda url, timeout=0: Resp(
        [{"id": 0, "path": "a/n2.gguf"}, {"id": 1, "path": "a/n1.gguf"}]))
    assert orch._llamacpp_lora_field() == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]
    assert orch._applied_adapter_version() == "n1"
    reg["current"] = "n2"                                   # promotion = pointer move, no restart
    assert orch._llamacpp_lora_field() == [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]
    assert orch._applied_adapter_version() == "n2"


def test_adapter_not_loaded_logs_base_core(reg, monkeypatch):
    reg["adapters"][0]["gguf"] = "adapters/n1/n1.gguf"
    monkeypatch.setattr(orch.requests, "get", lambda url, timeout=0: Resp([]))
    assert orch._applied_adapter_version() == "base_core"


def test_unreachable_server_is_not_cached(reg, monkeypatch):
    reg["adapters"][0]["gguf"] = "adapters/n1/n1.gguf"

    def boom(url, timeout=0):
        raise ConnectionError

    monkeypatch.setattr(orch.requests, "get", boom)
    assert orch._applied_adapter_version() == "base_core"
    monkeypatch.setattr(orch.requests, "get", lambda url, timeout=0: Resp([{"id": 0, "path": "n1.gguf"}]))
    assert orch._applied_adapter_version() == "n1"


def test_chat_transport_logs_model_name(reg, monkeypatch):
    monkeypatch.setattr(orch, "REACTION_TRANSPORT", "chat")
    assert orch._applied_adapter_version().startswith("chat:")


@pytest.mark.parametrize("explicit,direct,expected", [
    ("", True, "chat"), ("", False, "completion"), ("completion", True, "completion"), ("chat", False, "chat"),
])
def test_transport_auto(monkeypatch, explicit, direct, expected):
    monkeypatch.setattr(orch, "REACTION_TRANSPORT", explicit)
    monkeypatch.setattr(orch, "_direct_mode", direct)
    assert orch._transport() == expected


def test_preflight_fails_on_grammar_mismatch(reg, monkeypatch, tmp_path):
    reg["adapters"][0]["grammar_hash"] = "deadbeef0000"
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    r = preflight.Report()
    preflight.check_versions(r)
    assert not r.ok and any(n == "adapter grammar" for _, n, _ in r.items)


def test_serve_cmd_includes_registered_loras(monkeypatch, tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"adapters": [{"version": "n1", "gguf": "adapters/n1/n1.gguf", "status": "promoted"},
                                         {"version": "n0", "gguf": "adapters/n0/n0.gguf", "status": "rejected"}],
                             "current": "n1"}))
    monkeypatch.setattr(preflight, "ADAPTER_REGISTRY", p)
    cmd = preflight.serve_cmd()
    assert "n1.gguf" in cmd and "n0.gguf" not in cmd and "--lora-init-without-apply" in cmd
