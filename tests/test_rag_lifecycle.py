from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml_labs import rag_lifecycle as rag
from ml_labs.llmops import create_release_promptfoo_config

DOCUMENTS = [
    {
        "id": "registry", "text": "Model registry versions models with champion aliases. "
        "Validation must pass before promotion.",
    },
    {
        "id": "features", "text": "Feast offline store retrieves historical features. "
        "The online store serves latest features.",
    },
]
CASES = [
    {
        "id": "registry", "query": "model registry champion",
        "expected_ids": ["registry"], "answer_contains": ["Validation"],
    },
    {
        "id": "features", "query": "Feast offline historical features",
        "expected_ids": ["features"], "answer_contains": ["online"],
    },
    {"id": "unknown", "query": "quasar teleportation warranty", "unanswerable": True},
]


def candidate(tmp_path, name="candidate", config=None, prompt=None, documents=None):
    return rag.create_candidate(
        documents or DOCUMENTS, config or {"top_k": 1},
        prompt or rag.DEFAULT_PROMPT, CASES, tmp_path / name,
    )


def release(tmp_path, name="candidate", config=None):
    path = candidate(tmp_path, name, config)
    report = tmp_path / f"{name}-report.json"
    result = rag.evaluate_candidate(path, report)
    assert result["passed"]
    activated = rag.activate_release(tmp_path / "store", path, report)
    return path, activated


def test_demo_runs_complete_release_and_no_grounding(tmp_path):
    inputs = rag.init_demo(tmp_path / "inputs")
    path = rag.create_candidate(**inputs, output=tmp_path / "candidate")
    report = tmp_path / "report.json"
    evaluation = rag.evaluate_candidate(path, report)
    assert evaluation["passed"]
    activated = rag.activate_release(tmp_path / "store", path, report)
    answer = rag.query_release(tmp_path / "store", "model registry champion")
    assert answer["release_id"] == activated["release_id"]
    assert "doc-mlflow-registry" in answer["citations"]
    assert len(answer["trace_id"]) == 32
    assert answer["metrics"]["total_ms"] >= answer["metrics"]["latency_ms"] >= 0
    assert answer["metrics"]["input_tokens"] is None
    assert answer["metrics"]["output_tokens"] is None
    assert answer["metrics"]["cost_usd"] == 0
    assert answer["metrics"]["llm_calls"] == 0
    unknown = rag.query_release(tmp_path / "store", "quasar teleportation warranty")
    assert unknown["answer"] == rag.NO_GROUNDING
    assert unknown["citations"] == []
    assert unknown["status"] == "no_grounding"
    assert unknown["citations_valid"]


def test_prompt_registry_is_versioned_and_candidate_freezes_inputs(tmp_path):
    path = rag.register_prompt(tmp_path / "store", "grounded", "1", "{context} {question}")
    assert rag.register_prompt(
        tmp_path / "store", "grounded", "1", "{context} {question}"
    ) == path
    with pytest.raises(ValueError, match="already exists"):
        rag.register_prompt(tmp_path / "store", "grounded", "1", "{question} {context}")
    version2 = rag.register_prompt(
        tmp_path / "store", "grounded", "2", "Question: {question}; context: {context}"
    )
    first = candidate(tmp_path, "first", prompt=path)
    second = candidate(tmp_path, "second", prompt=version2)
    assert json.loads(first.read_text())["candidate_id"] != json.loads(
        second.read_text()
    )["candidate_id"]
    path.unlink()
    assert rag.query_candidate(first, "model registry")["citations"] == ["registry"]


def test_release_survives_original_candidate_removal_and_pins_previous(tmp_path):
    first, activated1 = release(tmp_path, "first")
    _, activated2 = release(tmp_path, "second", {"top_k": 2})
    for file in first.parent.iterdir():
        file.unlink()
    first.parent.rmdir()
    assert rag.query_release(tmp_path / "store", "registry")["release_id"] == activated2[
        "release_id"
    ]
    pinned = rag.query_release(
        tmp_path / "store", "registry", release_id=activated1["release_id"]
    )
    assert pinned["release_id"] == activated1["release_id"]
    rollback = rag.rollback_release(tmp_path / "store")
    assert rollback["release_id"] == activated1["release_id"]
    assert rag.query_release(tmp_path / "store", "registry")["release_id"] == activated1[
        "release_id"
    ]
    assert rag.rollback_release(tmp_path / "store")["release_id"] == activated2["release_id"]
    with pytest.raises(ValueError, match="previously activated"):
        rag.rollback_release(tmp_path / "store", "not-known")


def test_http_serves_active_release_rejects_regression_and_observes_rollback(tmp_path):
    from fastapi.testclient import TestClient

    from ml_labs.api import create_app

    store = tmp_path / "store"
    baseline, first = release(tmp_path, "first")
    with TestClient(create_app(rag_release=store)) as client:
        initial = client.post("/rag/query", json={"query": "model registry champion"})
        assert initial.status_code == 200
        answer = initial.json()
        assert answer["release_id"] == first["release_id"]
        assert answer["citations"] == ["registry"]
        assert answer["trace_id"]
        assert answer["metrics"]["llm_calls"] == 0
        assert answer["metrics"]["latency_ms"] >= 0
        unknown = client.post("/rag/query", json={"query": "quasar teleportation warranty"})
        assert unknown.status_code == 200
        assert unknown.json()["abstained"] and unknown.json()["citations"] == []

        worse = candidate(tmp_path, "worse", {"min_score": 1})
        bad_report = tmp_path / "bad-report.json"
        failed = rag.evaluate_candidate(
            worse, bad_report, baseline=baseline, min_pass_rate=0.3, max_regression=0,
        )
        assert failed["summary"]["regressions"] == ["registry", "features"]
        assert not failed["passed"]
        with pytest.raises(ValueError, match="gate failed"):
            rag.activate_release(store, worse, bad_report)
        after_rejection = client.post("/rag/query", json={"query": "model registry champion"})
        assert after_rejection.status_code == 200
        assert after_rejection.json()["release_id"] == first["release_id"]

        _, second = release(tmp_path, "second", {"top_k": 2})
        after_activation = client.post("/rag/query", json={"query": "model registry champion"})
        assert after_activation.status_code == 200
        assert after_activation.json()["release_id"] == second["release_id"]
        assert second["release_id"] != first["release_id"]
        rag.rollback_release(store)
        after_rollback = client.post("/rag/query", json={"query": "model registry champion"})
        assert after_rollback.status_code == 200
        assert after_rollback.json()["release_id"] == first["release_id"]
        assert after_rollback.json()["citations"] == answer["citations"]
        assert after_rollback.json()["trace_id"] != answer["trace_id"]


def test_regression_report_is_persisted_and_cannot_activate(tmp_path):
    baseline = candidate(tmp_path, "baseline")
    worse = candidate(tmp_path, "worse", {"min_score": 1})
    report = tmp_path / "report.json"
    result = rag.evaluate_candidate(
        worse, report, baseline=baseline, min_pass_rate=0.3, max_regression=0
    )
    assert not result["passed"]
    assert result["summary"]["regressions"] == ["registry", "features"]
    assert result["summary"]["baseline_pass_rate"] == 1
    assert json.loads(report.read_text())["report_id"] == result["report_id"]
    with pytest.raises(ValueError, match="gate failed"):
        rag.activate_release(tmp_path / "store", worse, report)
    assert not (tmp_path / "store" / "active.json").exists()


def test_improvements_do_not_cancel_lost_baseline_cases(tmp_path):
    baseline_docs = [DOCUMENTS[0], {"id": "features", "text": "unrelated material"}]
    candidate_docs = [{"id": "registry", "text": "unrelated material"}, DOCUMENTS[1]]
    baseline = candidate(tmp_path, "baseline", documents=baseline_docs)
    changed = candidate(tmp_path, "changed", documents=candidate_docs)
    result = rag.evaluate_candidate(
        changed, tmp_path / "report.json", baseline=baseline, min_pass_rate=0.5
    )
    assert result["summary"]["pass_rate"] == result["summary"]["baseline_pass_rate"]
    assert result["summary"]["regressions"] == ["registry"]
    assert not result["passed"]


@pytest.mark.parametrize("filename", ["config.json", "corpus.json", "prompt.json", "evaluation.json"])
def test_changed_snapshot_is_rejected_after_evaluation(tmp_path, filename):
    path = candidate(tmp_path)
    report = tmp_path / "report.json"
    rag.evaluate_candidate(path, report)
    file = path.parent / filename
    value = json.loads(file.read_text())
    if isinstance(value, list):
        value.reverse()
    else:
        value["tampered"] = True
    file.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        rag.activate_release(tmp_path / "store", path, report)
    with pytest.raises(ValueError, match="integrity"):
        rag.query_candidate(path, "model registry")


def test_report_bound_to_exact_candidate_and_report_bytes(tmp_path):
    first = candidate(tmp_path, "first")
    second = candidate(tmp_path, "second", {"top_k": 2})
    report = tmp_path / "report.json"
    rag.evaluate_candidate(first, report)
    with pytest.raises(ValueError, match="different snapshot"):
        rag.activate_release(tmp_path / "store", second, report)
    changed = json.loads(report.read_text())
    changed["policy"]["min_pass_rate"] = 0
    report.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        rag.activate_release(tmp_path / "store", first, report)


def test_serving_revalidates_release_snapshot(tmp_path):
    _, activated = release(tmp_path)
    stored = tmp_path / "store" / "releases" / activated["release_id"] / "corpus.json"
    documents = json.loads(stored.read_text())
    documents[0]["text"] = "Changed after activation."
    stored.write_text(json.dumps(documents), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        rag.query_release(tmp_path / "store", "registry")


def test_pointer_swap_failure_preserves_active_release(tmp_path, monkeypatch):
    _, first = release(tmp_path, "first")
    second = candidate(tmp_path, "second", {"top_k": 2})
    report = tmp_path / "second-report.json"
    rag.evaluate_candidate(second, report)
    original = rag.os.replace

    def fail_pointer(source, destination):
        if Path(destination).name == "active.json":
            raise OSError("simulated failed atomic replacement")
        return original(source, destination)

    monkeypatch.setattr(rag.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="simulated"):
        rag.activate_release(tmp_path / "store", second, report)
    assert rag.query_release(tmp_path / "store", "registry")["release_id"] == first["release_id"]
    assert not (tmp_path / "store" / ".release.lock").exists()
    assert not list((tmp_path / "store").glob("*.pending"))


def test_latency_budget_uses_observed_runtime_measurement(tmp_path, monkeypatch):
    path = candidate(tmp_path)
    original = rag.RAGRuntime.query

    def slow_query(self, query, trace=False):
        response = original(self, query, trace)
        response["metrics"]["latency_ms"] = 15.0
        return response

    monkeypatch.setattr(rag.RAGRuntime, "query", slow_query)
    result = rag.evaluate_candidate(path, tmp_path / "report.json", max_latency_ms=10)
    assert result["summary"]["p95_latency_ms"] == 15
    assert not result["passed"]


def test_azure_uses_same_retrieval_prompt_and_actual_response_usage(tmp_path, monkeypatch):
    path = candidate(tmp_path, config={
        "provider": "azure_openai", "deployment": "existing-chat-deployment", "top_k": 1,
    })
    prompts = []

    def generated(prompt, config):
        prompts.append(prompt)
        assert "[registry]" in prompt
        assert "champion aliases" in prompt
        assert "[features]" not in prompt
        assert config["deployment"] == "existing-chat-deployment"
        return {
            "content": json.dumps({
                "answer": "Validation before promotion [registry]",
                "citations": ["registry"], "abstained": False,
            }),
            "input_tokens": 83, "output_tokens": 12,
            "provider_request_id": "mock-provider-id", "model": "mock-model",
        }

    monkeypatch.setattr(rag, "_azure_generate", generated)
    answer = rag.query_candidate(path, "model registry champion")
    assert answer["citations"] == ["registry"]
    assert answer["metrics"]["input_tokens"] == 83
    assert answer["metrics"]["output_tokens"] == 12
    assert answer["metrics"]["cost_usd"] is None
    assert answer["metrics"]["llm_calls"] == 1
    assert answer["model"] == "mock-model"
    assert answer["citations_valid"]
    unknown = rag.query_candidate(path, "quasar teleportation warranty")
    assert unknown["abstained"] and unknown["metrics"]["llm_calls"] == 0
    assert len(prompts) == 1


@pytest.mark.parametrize("output", [
    "not JSON",
    {"answer": "Unsupported [not-retrieved]", "citations": ["not-retrieved"], "abstained": False},
    {"answer": "Missing inline ID", "citations": ["registry"], "abstained": False},
    {"answer": "Uncited claim", "citations": [], "abstained": False},
    {"answer": "Contradiction [registry]", "citations": ["registry"], "abstained": True},
])
def test_generated_citation_validation_fails_closed(tmp_path, monkeypatch, output):
    path = candidate(tmp_path, config={"provider": "azure_openai", "deployment": "existing"})
    monkeypatch.setattr(rag, "_azure_generate", lambda *_: {
        "content": output if isinstance(output, str) else json.dumps(output),
        "input_tokens": None, "output_tokens": None, "provider_request_id": "mock", "model": "mock",
    })
    result = rag.query_candidate(path, "model registry")
    assert result["answer"] == rag.NO_GROUNDING
    assert result["citations"] == []
    assert result["abstained"]
    assert not result["citations_valid"]


def test_promptfoo_provider_calls_runtime_not_static_prompt(tmp_path):
    path = candidate(tmp_path)
    suite_path = create_release_promptfoo_config(path, tmp_path / "promptfoo.yaml")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    assert suite["providers"][0]["id"].startswith("file://")
    assert suite["providers"][0]["config"]["candidate"] == str(path.resolve())
    result = rag.call_api(
        "not used", {"config": {"candidate": str(path)}},
        {"vars": {"query": "model registry champion"}},
    )
    answer = json.loads(result["output"])
    assert answer["citations"] == ["registry"]
    assert answer["candidate_id"] == json.loads(path.read_text())["candidate_id"]
    assert all(case["assert"][0]["type"] == "javascript" for case in suite["tests"])


def test_promptfoo_assertions_are_expressions_accepting_runtime_json_and_objects(tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute native JavaScript assertions.")
    path = candidate(tmp_path)
    suite_path = create_release_promptfoo_config(path, tmp_path / "promptfoo.yaml")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    checks = []
    for case in suite["tests"]:
        output = json.loads(rag.call_api(
            "", {"config": {"candidate": str(path)}}, {"vars": case["vars"]},
        )["output"])
        assertion = case["assert"][0]["value"]
        checks.extend([
            {"assertion": assertion, "output": output, "expected": True},
            {"assertion": assertion, "output": json.dumps(output), "expected": True},
            {
                "assertion": assertion, "output": {**output, "citations_valid": False},
                "expected": False,
            },
            {
                "assertion": assertion, "output": {**output, "abstained": not output["abstained"]},
                "expected": False,
            },
        ])
    script = (
        "const checks = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        "for (const c of checks) {"
        "const result = new Function('output', 'return (' + c.assertion + ');')(c.output);"
        "if (result !== c.expected) throw new Error('Unexpected assertion verdict');"
        "}"
    )
    subprocess.run(
        [node, "-e", script], input=json.dumps(checks), text=True,
        capture_output=True, check=True,
    )


def test_local_promptfoo_suite_does_not_require_openai_key(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ml_labs import llmops

    path = candidate(tmp_path)
    suite = create_release_promptfoo_config(path, tmp_path / "promptfoo.yaml")
    calls = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llmops.shutil, "which", lambda _: "promptfoo")

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="evaluated", stderr="")

    monkeypatch.setattr(llmops.subprocess, "run", run)
    assert llmops.run_promptfoo(suite)["stdout"] == "evaluated"
    assert "--no-cache" in calls[0]


def test_windows_powershell_bom_input_is_supported(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"top_k": 1}), encoding="utf-8-sig")
    path = candidate(tmp_path, config=config)
    assert rag.query_candidate(path, "model registry")["citations"] == ["registry"]


def test_optional_deepeval_verdict_is_part_of_persisted_activation_gate(tmp_path, monkeypatch):
    from ml_labs import llmops

    path = candidate(tmp_path)
    seen = []

    def judge(response, threshold):
        seen.append(response)
        return {
            "framework": "deepeval", "passed": False, "threshold": threshold,
            "metrics": [{"metric": "FaithfulnessMetric", "score": 0.1, "passed": False}],
        }

    monkeypatch.setattr(llmops, "evaluate_release_answer", judge)
    report = tmp_path / "report.json"
    result = rag.evaluate_candidate(path, report, deepeval=True)
    assert len(seen) == 2 and seen[0]["retrieved"]
    assert result["summary"]["pass_rate"] == 1
    assert not result["passed"]
    with pytest.raises(ValueError, match="gate failed"):
        rag.activate_release(tmp_path / "store", path, report)


def test_langfuse_trace_wraps_actual_runtime_only_when_requested(tmp_path, monkeypatch):
    from ml_labs import llmops

    path = candidate(tmp_path)
    seen = []

    def traced(runtime, question):
        result = runtime.query(question, trace=False)
        seen.append(result)
        return {**result, "trace_id": "test-langfuse-id", "sent_to_langfuse": True}

    monkeypatch.setattr(llmops, "trace_release_query", traced)
    assert not rag.query_candidate(path, "registry").get("sent_to_langfuse")
    assert seen == []
    response = rag.query_candidate(path, "registry", trace=True)
    assert response["trace_id"] == "test-langfuse-id"
    assert response["citations"] == seen[0]["citations"]


@pytest.mark.parametrize("config", [
    {"top_k": True}, {"min_score": float("nan")}, {"top_k": 0},
    {"provider": "azure_openai"}, {"retriever": "pretend-vector-search"},
    {"ngram_range": [2, 1]}, {"unknown": "setting"},
])
def test_invalid_config_never_leaves_candidate(tmp_path, config):
    with pytest.raises(ValueError):
        candidate(tmp_path, config=config)
    assert not (tmp_path / "candidate").exists()


def test_evaluation_requires_real_negative_cases_and_valid_queries(tmp_path):
    with pytest.raises(ValueError, match="both answerable and unanswerable"):
        rag.create_candidate(
            DOCUMENTS, {}, rag.DEFAULT_PROMPT, CASES[:2], tmp_path / "candidate"
        )
    with pytest.raises(ValueError, match="nonempty"):
        rag.query_candidate(candidate(tmp_path), " ")
