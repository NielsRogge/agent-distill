import json


def test_native_pi_sessions_get_the_harness_and_the_verification(tmp_path):
    from agent_distill.episode import Episode
    from agent_distill.records import RecordWriter
    from agent_distill.rubrics.core import RubricResult

    episode = Episode(task="t", row_id="r1", harness="pi", harness_version="1.6.0", model="m", answer="x")
    native = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-11-24T10:00:00.000Z", "cwd": "/w"},
        {"type": "message", "id": "a1", "parentId": None, "timestamp": "2026-11-24T10:00:01.000Z",
         "message": {"role": "user", "content": "hi"}},
    ]  # fmt: skip
    writer = RecordWriter(tmp_path, "teacher")
    writer.write(episode, RubricResult(scores={"ok": 1.0}, reward=1.0, passed=True), native_trace=native)
    (path,) = writer.sessions.glob("*.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["harness"] == "pi" and lines[0]["harness_version"] == "1.6.0" and lines[0]["id"] == "s1"
    assert lines[-1]["parentId"] == "a1" and lines[-1]["message"]["customType"] == "verification"
    assert "passed: True" in lines[-1]["message"]["content"]
