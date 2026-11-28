# Race Space: a teacher and its distilled student, same question, same harness

A private Hugging Face Space (modeled on url-extraction's `url-agent-race`) where a frontier teacher and the student
distilled from it answer the same question in the same pinned harness. Each agent gets a panel that streams its tool
calls (and, collapsed, its reasoning) as they happen. When both have finished, the task's rubrics grade them, the same
rubric set that accepts traces, rewards GRPO and scores evaluation. A replay button plays back a recorded race, so the
page has something to show while the student's GPU is asleep.

The page is plain HTML (`static/index.html`) on a small Starlette server; it does not use Gradio.

## What drives it

Nothing in the code names a task, model or harness. `race.toml` lists the races, and each race names a run config:

| From the run config | Used for |
|---|---|
| `task` | prompt, tools, skills, MCP servers, rubrics, the dataset the questions come from, the task's `job_setup` (installed at build time) |
| `teacher` profile | the teacher endpoint (`teacher_endpoint`: the HF router, `X-HF-Bill-To` from `[hub] bill_to`) and its prices |
| `student` profile | how the endpoint serves the student: `stages.eval.vllm_command` (reasoning and tool parsers, context, evaluation sampling, LoRA rank), and the output-token limit Pi is told |
| `harness` | the adapter that runs both episodes (`pi@1.6.0`), with its pinned install |
| `[eval] timeout`, `max_tool_calls` | the episode limits: the conditions the published numbers were measured in |

A race adds the adapter repo (served on the endpoint under the race's `id`), the student's display name, and the split
and row limit its questions come from. A new teacher/student pair needs a new `[[races]]` entry, plus a new
`[endpoint]` for a different student.

The first instance pits GLM-5.3 against Qwen3.5-4B recipe (a) (`qwen3.5-4b-think`) on both Hub QA examples:

| Race | Run config | Adapter |
|---|---|---|
| `cli` | `examples/hf-hub-qa/run-think.toml` | `nielsr/qwen35-4b-think-cfx-a-hf-hub-qa-v2-glm-53-pi-160-lora` |
| `mcp` | `examples/hf-hub-qa-mcp/run-think.toml` | `nielsr/qwen35-4b-think-cfx-a-hf-hub-qa-mcp-v2-glm-53-pi-160-lora` |

Questions are the races' test rows of `nielsr/hf-hub-facts-v2` (the first 150 for the CLI, all 377 for MCP, as in
the published evaluations), graded live with the deterministic rubrics (`fact_correct`, `grounded` and the gates).
A typed question runs both agents in the same way, but no grade is shown, since there is no gold answer.

Each panel shows the wall time, the tool calls, the output tokens and the cost. For the student it also shows the
thinking tokens, counted with the student's own tokenizer. Teacher cost uses the profile's prices; student cost is
the race's share of endpoint GPU time.

## The student endpoint

One managed Inference Endpoint serves both adapters as LoRA modules on the base model, so no merged model is needed:

- `huggingface/qwen35-4b-think-hub-qa-race`
- one NVIDIA L4 (aws us-east-1), $0.80/h while up
- vLLM 0.33.0 (the profile's version), with the `eval` stage's serving arguments plus two speed-ups that keep
  the sampling distribution, set in `[endpoint] serve`:
  - CUDA graphs (`enforce_eager = false`);
  - speculative decoding with Qwen3.5's own MTP head (`--speculative-config {"method": "qwen3_5_mtp",
    "num_speculative_tokens": 2}`). The adapter-carrying model verifies the drafts by rejection sampling.
- prefix caching on (vLLM's default): 69% of prompt tokens hit the cache in agent episodes
- at most one replica; scales to zero after 15 idle minutes

Benchmark on the L4: student only, the first 5 test questions of each race (10 episodes per configuration), plus
streamed 600-token completions through the adapter for generation speed:

| Configuration | Mean episode | Median | Generation | Time to first token | Passed |
|---|---|---|---|---|---|
| eager (the `eval` stage's serving) | 52.2 s | 51.6 s | 13.7 tok/s | 0.66 s | 10/10 |
| CUDA graphs | 37.6 s | 31.7 s | 25.2 tok/s | 0.53 s | 10/10 |
| CUDA graphs + MTP (2 tokens), live | 18.9 s | 16.4 s | 44.6 tok/s | 0.54 s | 10/10 |

The student samples at temperature 0.6, so episode lengths vary from run to run: the mean output was 631–773 tokens
per episode across the three runs.

The L4 is the smallest GPU with bf16 that fits a 4B model at the profile's 64k context (a T4 has no bf16).

The first race after the endpoint has scaled to zero waits through a cold start of a few minutes ("waking the GPU").
Both agents start together only once the student answers, so the race stays fair.

## Credentials and isolation

The agents run shell commands and read untrusted Hub content, so they must never see a credential:

- `start.sh` writes `HF_TOKEN` to a one-time file and starts the app without it in its environment. `launch.py`
  reads and deletes that file. The token then lives only in the server process's memory: `/proc/<pid>/environ`
  shows the environment the process started with, which never held it.
- The server process is non-dumpable (`PR_SET_DUMPABLE` 0), so processes of the same user cannot read its memory or
  its `/proc` files.
- Agents get a credential-free environment from the Pi adapter.
- The agents reach both models only through the library's per-episode key proxy. The proxy accepts a random
  session key, adds the token (and `X-HF-Bill-To` for the teacher) on the way out, and revokes the key when the
  episode ends.
- The app and its tools are installed as root, so they are read-only to the app's user. Agents write only under
  `/tmp`.

The image sets no `USER`, and on the deployed Space the app runs as root: the isolation preflight passes there
(`/api/status` reports `isolation: {"mode": "own-uid", "ok": true}`). So the Pi adapter runs every agent as its own
unprivileged UID that owns only its 0700 episode directory (`agent_distill/harnesses/isolation.py`). Such an agent
cannot read the server's environment, memory or files, nor another episode's. If a Space ever ran the container as a
user that cannot switch UIDs, agents would fall back to the app's user (`stage-user`, shown in `/api/status`), and
the token would still be protected by the list above.

## Deploy

```bash
uv run spaces/race/deploy.py endpoint --dry-run   # the endpoint's vLLM args, from the student profile
uv run spaces/race/deploy.py endpoint             # create it (--update to change an existing one)
uv run spaces/race/deploy.py space                # upload the private Space (cpu-basic), set ENDPOINT_URL
```

The bundle holds `spaces/race`, the `agent_distill` package, the races' task folders, `pyproject.toml` and `uv.lock`.
The image installs the locked dependencies (`--extra hub --extra space`, hash-pinned) and runs each task's
`job_setup` (e.g. the pinned `hf` CLI). It installs the pinned harness at build time.

The Space's owner sets the `HF_TOKEN` secret. It needs:

- `inference.endpoints.infer.write` on `huggingface`, for the student endpoint;
- Inference Providers calls billed to `huggingface` (`inference.serverless.write` there), for the teacher.

It needs nothing else, not even repo read access: datasets and adapters are public.

## Replays

```bash
python spaces/race/record.py --url http://127.0.0.1:7860 --race mcp --row <question id>
```

This runs a race against a running server and writes `static/replays/<race>.json`, with frames thinned and long
thoughts cut. `deploy.py space` uploads it with the bundle. Recordings are agent trajectories, so they live only in
the private Space and stay out of git (`static/replays/.gitignore`). A deploy without local recordings leaves the
Space's existing ones in place.

## Local run

```bash
cd spaces/race
ENDPOINT_URL=https://<endpoint>.endpoints.huggingface.cloud HF_TOKEN=... uv run --extra hub --extra space python launch.py
```

The CLI race needs the task's tools on the PATH (`python setup.py | bash` installs them, as the image does). As
root, agents run as their own UIDs, so the harness cache (`XDG_CACHE_HOME`) and the tools must live outside root's
home. Don't run `tests/test_isolation.py` while a local race runs: it kills processes of the same agent-UID pool.
