#!/bin/bash
# Move the HF token out of the environment before anything else starts: the agents run shell commands in this
# container and must not be able to read it from /proc/*/environ. launch.py reads the one-time file and deletes it.
set -euo pipefail
umask 077
mkdir -p "${HOME:-/tmp/home}"
# The PATH the tasks' setup built at image build time (setup.py), e.g. the pinned hf CLI.
if [ -f /opt/race.path ]; then PATH="$(cat /opt/race.path)"; export PATH; fi
token_file="$(mktemp /tmp/.race-token.XXXXXX)"
if [ -n "${HF_TOKEN:-}" ]; then printf '%s' "$HF_TOKEN" > "$token_file"; fi
exec env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_TOKEN RACE_TOKEN_FILE="$token_file" python launch.py
