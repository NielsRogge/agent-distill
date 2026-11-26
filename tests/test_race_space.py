"""spaces/race: its configuration comes from the run configs and profiles, and its endpoint serves the student exactly
as the `eval` stage does (CPU only, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RACE = Path(__file__).resolve().parent.parent / "spaces" / "race"
sys.path.insert(0, str(RACE))

import config as race_config  # noqa: E402
import record  # noqa: E402

from agent_distill.stages.eval import vllm_command  # noqa: E402


def test_races_come_from_run_configs() -> None:
    cfg = race_config.load()
    assert {r.id for r in cfg.races} == {"cli", "mcp"}
    for race in cfg.races:
        assert race.config.harness == "pi@1.6.0"
        assert race.config.student == "qwen3.5-4b-think"
        assert race.teacher_endpoint().headers == {"X-HF-Bill-To": race.config.hub.bill_to}
        assert race.teacher_endpoint().key_env == "HF_TOKEN"  # reached through the key proxy, never by the agent
        assert race.limits().max_tool_calls == race.config.eval.max_tool_calls
        student = race.student_endpoint("https://example.endpoints.huggingface.cloud/")
        assert student.base_url == "https://example.endpoints.huggingface.cloud/v1"
        assert student.model == race.id and student.key_env == "HF_TOKEN"
        assert student.max_output_tokens == race.student().serve.max_output_tokens == 8192
    assert cfg.race("mcp").task().mcp_servers and not cfg.race("cli").task().mcp_servers


def test_endpoint_serves_the_student_like_the_eval_stage() -> None:
    cfg = race_config.load()
    student = cfg.student()
    args = cfg.endpoint_args()
    reference = vllm_command(student, student.model, "ADAPTER")[3:]
    for flag in ("--host", "--port"):
        index = reference.index(flag)
        del reference[index : index + 2]
    modules = reference.index("--lora-modules")
    assert args[:modules] == reference[:modules]  # parsers, context, eager mode, LoRA rank: unchanged
    assert args[modules + 1 : modules + 3] == [f"{r.id}={r.adapter}" for r in cfg.races]
    assert "--reasoning-parser" in args and "qwen3" in args
    assert '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' in args
    assert args[-2:] == ["--max-loras", "2"]


def test_one_endpoint_serves_one_student(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = race_config.load()
    monkeypatch.setattr(cfg.races[1].config, "student", "qwen3.5-4b-nothink")
    with pytest.raises(ValueError, match="one student profile"):
        cfg.student()


def _frame(t: float, rows: int, text: str = "") -> dict:
    run = {"status": "running", "rows": [{"kind": "think", "text": text}] * rows, "answer": None, "grade": None}
    return {"t": t, "teacher": run, "student": run}


def test_replay_thinning_keeps_every_structural_change() -> None:
    frames = [_frame(i, 1 + i // 50, text="x" * i) for i in range(300)] + [_frame(300, 7, "x" * 299)] * 5
    thinned = record.thin(frames, 40)
    assert len(thinned) <= 40 and thinned[0] is frames[0] and thinned[-1]["t"] == 300
    counts = {len(f["teacher"]["rows"]) for f in thinned}
    assert counts == {1, 2, 3, 4, 5, 6, 7}  # every row count that ever appeared
    long = record.slim(_frame(0, 1, "y" * 10_000))
    assert len(long["teacher"]["rows"][0]["text"]) == record.THOUGHT_CHARS + 1
