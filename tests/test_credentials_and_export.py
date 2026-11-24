from __future__ import annotations

import json

from agent_distill.credentials import record_line, redact_credentials, scan_files, scan_text, strip_credentials
from agent_distill.episode import Episode, ToolEvent
from agent_distill.export.pi_session import episode_to_pi_session, session_filename, write_session
from agent_distill.harnesses.opencode_v1_reader import episode_from_messages

HF = "hf_" + "A1b2C3d4" * 4 + "xyzw1"
GOOGLE = "AIza" + "S" * 35
PAT = "github_pat_" + "a1" * 15


def test_redaction_covers_every_credential_shape():
    text = f"HF_TOKEN={HF} key={GOOGLE} gh={PAT} sk=sk-{'x' * 24}"
    redacted = redact_credentials(text)
    for secret in (HF, GOOGLE, PAT):
        assert secret not in redacted
    assert redacted.count("***REDACTED***") == 4


def test_record_line_redacts_and_survives_lone_surrogates():
    line = record_line({"output": f"env: {HF}", "math": "x \ud835 y"})
    assert HF not in line
    assert json.loads(line)["math"] == "x \ud835 y"


def test_scan_never_returns_full_values(tmp_path):
    (tmp_path / "a.jsonl").write_text(f'{{"x": "{HF}"}}\n')
    (tmp_path / "b.jsonl").write_text("clean\n")
    assert scan_files(sorted(tmp_path.iterdir())) == {str(tmp_path / "a.jsonl"): 1}
    assert all(HF not in hit for hit in scan_text(HF))


def test_strip_credentials():
    env = {"HF_TOKEN": "x", "GEMINI_API_KEY": "y", "PATH": "/bin"}
    assert strip_credentials(env) == {"PATH": "/bin"}


def opencode_messages():
    return [
        {
            "info": {"role": "user", "time": {"created": 1_700_000_000_000}},
            "parts": [{"type": "text", "text": "find it"}],
        },
        {
            "info": {
                "role": "assistant",
                "time": {"created": 1_700_000_001_000},
                "providerID": "local-vllm",
                "modelID": "policy",
                "tokens": {"input": 10, "output": 5, "total": 15},
                "cost": 0.01,
            },
            "parts": [
                {"type": "reasoning", "text": "think"},
                {"type": "text", "text": "running"},
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "bash",
                    "state": {"status": "completed", "input": {"command": f"echo {HF}"}, "output": f"{HF}\n"},
                },
                {
                    "type": "tool",
                    "callID": "c2",
                    "tool": "webfetch",
                    "state": {"status": "error", "input": {"url": "u"}, "error": "404"},
                },
            ],
        },
    ]


def test_v1_reader_messages_and_kinds():
    from pydantic import BaseModel

    class Answer(BaseModel):
        x: int

    ep = episode_from_messages(
        opencode_messages(), answer_model=Answer, task="t", row_id="r", model="local-vllm/policy", system_prompt="sys"
    )
    assert [e.kind for e in ep.tool_events] == ["shell", "fetch"]
    assert [e.success for e in ep.tool_events] == [True, False]
    roles = [m["role"] for m in ep.messages]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant", "tool"]
    assert ep.messages[2]["reasoning_content"] == "think" and ep.messages[2]["content"] == "running"
    assert ep.messages[-1]["is_error"] and ep.messages[-1]["content"] == "404"
    assert ep.usage == {"input": 10, "output": 5, "cost": 0.01}
    assert ep.answer is None and ep.harness_id == "opencode@1.2.4"


def test_pi_session_structure_and_redaction(tmp_path):
    from pydantic import BaseModel

    class Answer(BaseModel):
        x: int

    ep = episode_from_messages(
        opencode_messages(),
        answer_model=Answer,
        task="paper-urls",
        row_id="42",
        model="local-vllm/policy",
        system_prompt="sys",
    )
    lines = episode_to_pi_session(ep, verification={"scores": {"a": 1.0}, "reward": 1.0, "passed": True})
    header = lines[0]
    assert header["type"] == "session" and header["version"] == 3
    assert header["harness"] == "opencode" and header["harness_version"] == "1.2.4"
    assert header["name"] == "42 · policy · passed"
    entries = lines[1:]
    assert entries[0]["type"] == "model_change"
    assert entries[0]["provider"] == "local-vllm" and entries[0]["modelId"] == "policy"
    messages = [e["message"] for e in entries if e["type"] == "message"]
    kinds = [m["role"] if m["role"] != "custom" else m["customType"] for m in messages]
    assert kinds == ["system_prompt", "user", "assistant", "toolResult", "assistant", "toolResult", "verification"]
    first_assistant = messages[2]
    assert [c["type"] for c in first_assistant["content"]] == ["thinking", "text", "toolCall"]
    assert first_assistant["stopReason"] == "toolUse"
    assert messages[5]["isError"] is True and messages[5]["toolName"] == "webfetch"
    # parent chain is linear
    ids = [e["id"] for e in entries]
    assert [e["parentId"] for e in entries] == [None, *ids[:-1]]
    text = json.dumps(lines)
    assert HF not in text
    path = tmp_path / session_filename(ep)
    write_session(lines, path)
    assert path.name == "42_policy.jsonl" and len(path.read_text().splitlines()) == len(lines)


def test_pi_session_without_timestamps():
    ep = Episode(task="t", row_id="r", harness="pi", harness_version="1.6.0", model="m",
                 messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                 tool_events=[ToolEvent(kind="submit", name="submit")])  # fmt: skip
    lines = episode_to_pi_session(ep)
    assert lines[0]["timestamp"].endswith("Z") and lines[-1]["message"]["role"] == "assistant"


# --------------------------------------------------------------------------------------------------
# Review fixes: line breaks, redaction boundaries, header metadata, truncation
# --------------------------------------------------------------------------------------------------

LINE_BREAKS = "\u2028\u2029\x85\x0b\x0c\x1c\x1d\x1e\r"


def test_record_line_escapes_every_splitlines_break():
    record = {"output": f"a{LINE_BREAKS}b", "nested": [{"x": "\u2028"}]}
    line = record_line(record)
    assert line.endswith("\n") and len(line.splitlines()) == 1
    assert json.loads(line) == record


def test_write_session_escapes_line_breaks_and_redacts(tmp_path):
    lines = [{"type": "session", "note": f"x{LINE_BREAKS}y"}, {"type": "message", "text": f"\n{HF}"}]
    path = tmp_path / "s.jsonl"
    write_session(lines, path)
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2 and HF not in text
    assert json.loads(text.splitlines()[0])["note"] == f"x{LINE_BREAKS}y"


def test_redaction_after_json_escapes_and_colour_codes():
    sk = "sk-" + "Q" * 24
    for prefix in ("\n", "\x1b[0m", "A", "token="):
        line = record_line({"out": f"{prefix}{HF} {prefix}{GOOGLE} {prefix}{PAT}"})
        assert HF not in line and GOOGLE not in line and PAT not in line, prefix
    for prefix in ("\n", "\x1b[0m", " ", "="):
        assert sk not in record_line({"out": f"{prefix}{sk}"}), prefix
    assert "task-" + "b" * 30 in record_line({"out": "task-" + "b" * 30})  # words with "sk-" inside stay


def test_pi_session_header_metadata_is_redacted():
    ep = Episode(task="t", row_id="r", harness="opencode", harness_version="1.2.4", model="m",
                 messages=[{"role": "user", "content": "q"}],
                 metadata={"failure_reason": f"provider error: invalid key {HF}"})  # fmt: skip
    header = episode_to_pi_session(ep)[0]
    assert HF not in json.dumps(header) and "REDACTED" in header["failure_reason"]


def test_tool_output_is_redacted_before_truncation():
    from agent_distill.export.pi_session import MAX_TOOL_OUTPUT_CHARS

    # The token straddles the truncation limit: truncating first would leave a partial token unredacted.
    output = "x" * (MAX_TOOL_OUTPUT_CHARS - 10) + HF + "tail"
    ep = Episode(task="t", row_id="r", harness="opencode", harness_version="1.2.4", model="m",
                 messages=[{"role": "tool", "tool_call_id": "c", "content": output}])  # fmt: skip
    lines = episode_to_pi_session(ep)
    text = lines[-1]["message"]["content"][0]["text"]
    assert HF[:12] not in text and "truncated" not in text[:MAX_TOOL_OUTPUT_CHARS]


def test_v1_reader_uses_the_record_opencode_version():
    from pydantic import BaseModel

    from agent_distill.harnesses.opencode_v1_reader import episode_from_record

    class Answer(BaseModel):
        x: int

    record = {"opencode_messages": opencode_messages(), "model": "m", "opencode_version": "1.3.0"}
    assert episode_from_record(record, answer_model=Answer, task="t", row_id="r").harness_version == "1.3.0"
    del record["opencode_version"]
    assert episode_from_record(record, answer_model=Answer, task="t", row_id="r").harness_version == "1.2.4"
