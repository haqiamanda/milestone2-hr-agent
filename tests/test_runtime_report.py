"""Runtime artifacts are immutable, scoped to a run, and contain no history."""
from pathlib import Path

import httpx
import pytest

import src.run_evaluation as evaluator


def fake_http(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={
            "answer": "Safe synthetic response", "outcome": "answered",
            "citations": [], "tool_calls": [], "usage": {
                "input_tokens": 0, "output_tokens": 0, "latency_ms": 1,
            },
        })

    original = httpx.Client
    monkeypatch.setattr(evaluator.httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))
    return seen


def test_default_template_contains_only_this_run_and_preserves_build_identity(monkeypatch, tmp_path):
    fake_http(monkeypatch)
    path = tmp_path / "runtime.md"
    summary = evaluator.run("https://offline.test", "synthetic-bearer", output_path=path,
                            build_id="sha256:synthetic", qa_project="hr-qa-b21b139719e8")
    report = path.read_text(encoding="utf-8")
    assert "Official Runtime Evaluation" in report
    assert "sha256:synthetic" in report and "hr-qa-b21b139719e8" in report
    assert summary["run_identity"]["started_at_utc"]
    assert len(summary["run_identity"]["case_inputs_sha256"]) == 64
    assert "Actual outcome" in report and "UNKNOWN" in report
    for historical in ("299 PASSED", "Historical Mentor QA", "Confirmed Runtime Observations",
                       "Assessment Responses", "synthetic-bearer"):
        assert historical not in report


def test_existing_output_is_rejected_before_any_api_activity(monkeypatch, tmp_path):
    seen = fake_http(monkeypatch)
    path = tmp_path / "original.md"
    path.write_text("Original immutable result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluator.run("https://offline.test", "synthetic-bearer", output_path=path)
    assert seen == []
    assert path.read_text(encoding="utf-8") == "Original immutable result"


def test_default_output_is_unique_and_does_not_replace_previous_run(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake_http(monkeypatch)
    monkeypatch.chdir(tmp_path)
    kwargs = dict(cases_path=root / "data/evaluation_cases.json",
                  template_path=root / "templates/evaluation-runtime.md")
    first = evaluator.run("https://offline.test", "synthetic-bearer", **kwargs)
    second = evaluator.run("https://offline.test", "synthetic-bearer", **kwargs)
    assert first["output_path"] != second["output_path"]
    assert Path(first["output_path"]).is_file()
    assert Path(second["output_path"]).is_file()


def test_qa_state_option_refuses_remote_api_before_network_or_database_access(monkeypatch, tmp_path):
    seen = fake_http(monkeypatch)
    with pytest.raises(ValueError, match="inside the API container"):
        evaluator.run("https://outside.invalid", "synthetic-bearer", qa_read_state=True,
                      qa_project="hr-qa-b21b139719e8", output_path=tmp_path / "unused.md")
    assert seen == []


def test_exclusive_export_protects_an_artifact_created_during_the_run(monkeypatch, tmp_path):
    fake_http(monkeypatch)
    output = tmp_path / "concurrent.md"
    original_render = evaluator.render_report

    def render(summary, template):
        output.write_text("Other immutable result", encoding="utf-8")
        return original_render(summary, template)

    monkeypatch.setattr(evaluator, "render_report", render)
    with pytest.raises(FileExistsError):
        evaluator.run("https://offline.test", "synthetic-bearer", output_path=output)
    assert output.read_text(encoding="utf-8") == "Other immutable result"
