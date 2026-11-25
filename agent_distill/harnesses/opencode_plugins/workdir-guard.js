// agent-distill: reject a bash call whose `workdir` does not exist, with an error the model sees.
// Without it, OpenCode 1.2.4 never spawns the command and leaves the tool "running" until the episode times out
// (a model trained elsewhere can name a directory from its training environment, e.g. /tmp/url-traces-...).
import { existsSync, statSync } from "node:fs";

export const AgentDistillWorkdirGuard = async () => ({
  "tool.execute.before": async (input, output) => {
    const dir = output?.args?.workdir;
    if (input.tool === "bash" && dir && !(existsSync(dir) && statSync(dir).isDirectory())) {
      throw new Error(`workdir does not exist: ${dir}`);
    }
  },
});
