/**
 * agent-distill's Pi extension (DESIGN.md §6). Loaded with `pi --no-extensions -e <this file>`.
 *
 * Its configuration is a JSON file named by AGENT_DISTILL_PI_CONFIG:
 *   { "submit": { "mode": "schema", "schema": {...} } | { "mode": "file", "path": "index.html" },
 *     "max_tool_calls": 30, "strip_env": ["HF_TOKEN"] }
 *
 * - `submit`: the agent's final action. In schema mode its parameters are the task's answer schema; in file mode it
 *   takes no arguments and marks the workspace file as final (it must exist). A successful submit ends the run.
 * - The tool-call limit is exact: the call past the limit is blocked and the run terminates.
 * - `bash` runs without the environment variables in `strip_env` (the teacher's key, which only Pi itself needs to
 *   call the model) and without Pi's session variables.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createBashTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

interface Config {
	submit: { mode: "schema"; schema: Record<string, unknown> } | { mode: "file"; path: string };
	max_tool_calls?: number;
	strip_env?: string[];
}

export default function (pi: ExtensionAPI) {
	const config: Config = JSON.parse(readFileSync(process.env.AGENT_DISTILL_PI_CONFIG ?? "", "utf8"));
	const cwd = process.cwd();
	const strip = new Set(config.strip_env ?? []);
	let calls = 0;

	pi.on("tool_call", async () => {
		calls += 1;
		if (config.max_tool_calls && calls > config.max_tool_calls) {
			return { block: true, reason: `tool-call limit reached (${config.max_tool_calls})`, terminate: true };
		}
	});

	const bash = createBashTool(cwd, {
		exposeSessionEnvironment: false,
		spawnHook: ({ command, cwd, env }) => ({
			command,
			cwd,
			env: Object.fromEntries(Object.entries(env).filter(([key]) => !strip.has(key))),
		}),
	});
	pi.registerTool({ ...bash, execute: (id, params, signal, onUpdate) => bash.execute(id, params, signal, onUpdate) });

	const submit = config.submit;
	if (submit.mode === "schema") {
		pi.registerTool({
			name: "submit",
			label: "Submit",
			description: "Submit your final answer. Call it once, as your last action; the run ends after it.",
			promptSnippet: "Submit the final answer (ends the run)",
			parameters: submit.schema as any,
			async execute(_id, params) {
				return { content: [{ type: "text", text: "Answer submitted." }], details: { answer: params }, terminate: true };
			},
		});
	} else {
		pi.registerTool({
			name: "submit",
			label: "Submit",
			description: `Submit ${submit.path} as your final result. Call it once, as your last action, after the file is written; the run ends after it.`,
			promptSnippet: `Submit ${submit.path} as the final result (ends the run)`,
			parameters: Type.Object({}),
			async execute() {
				if (!existsSync(join(cwd, submit.path))) {
					return { content: [{ type: "text", text: `${submit.path} does not exist yet; write it first.` }], details: {}, isError: true };
				}
				return { content: [{ type: "text", text: `${submit.path} submitted.` }], details: { file: submit.path }, terminate: true };
			},
		});
	}
}
