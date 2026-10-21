#!/bin/bash
# Agent phase (runs inside the trial pod). Mirrors Harbor's mini-swe-agent agent exactly:
# same install (uv, Python 3.12, mini-swe-agent 2.4.6 + litellm/orjson/fastapi, packages resolved as of
# UV_EXCLUDE_NEWER), same CLI flags and config, same prompt, run from /workspace, 3600 s limit.
# Usage: run_agent.sh <workload>    (expects /workspace and /opt/parity to be populated)
set -uo pipefail
WORKLOAD="$1"
RESULTS=/results
mkdir -p "$RESULTS"

# Prompt: the Harbor adapter's instruction template, rendered the same way the adapter renders it.
python3 - "$WORKLOAD" <<'EOF'
import sys
w = sys.argv[1]
t = open("/opt/parity/instruction_template.md").read()
b = open("/workspace/baseline.py").read().rstrip("\n")
open("/opt/parity/instruction.md", "w").write(t.replace("{{workload}}", w).replace("{{baseline_source}}", b))
EOF

# Install (agent setup; not counted against the agent time limit, as in Harbor).
export UV_EXCLUDE_NEWER="2026-10-17T00:00:00Z"
{
  set -e
  if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
  if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi
  export PATH="$HOME/.local/bin:$PATH"
  uv python install 3.12
  uv tool install --python 3.12 mini-swe-agent==2.4.6 --with litellm --with orjson --with fastapi
  mini-swe-agent --help >/dev/null
} > "$RESULTS/agent_setup.txt" 2>&1
SETUP_EXIT=$?
echo "$SETUP_EXIT" > "$RESULTS/agent_setup_exit"
if [ "$SETUP_EXIT" -ne 0 ]; then echo "setup failed" > "$RESULTS/AGENT_DONE"; exit 0; fi
export PATH="$HOME/.local/bin:$PATH"

# Agent run: same environment and flags as Harbor's mini-swe-agent agent.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:?}" OPENAI_API_BASE="${OPENAI_BASE_URL}"
export MSWEA_CONFIGURED=true MSWEA_COST_TRACKING=ignore_errors
cd /workspace
date -u +%FT%TZ > "$RESULTS/agent_started_at"
timeout --signal=KILL 3600 mini-swe-agent --yolo --model=openai/deepseek-v4.1-flash \
  --task="$(cat /opt/parity/instruction.md)" \
  --output="$RESULTS/mini-swe-agent.trajectory.json" --cost-limit 0 \
  -c mini -c model.model_kwargs.timeout=3600 -c /opt/parity/mini_config.yaml \
  --exit-immediately < /dev/null > "$RESULTS/mini-swe-agent.txt" 2>&1
AGENT_EXIT=$?
date -u +%FT%TZ > "$RESULTS/agent_finished_at"
echo "$AGENT_EXIT" > "$RESULTS/agent_exit"   # 137 = killed at the 3600 s limit
echo done > "$RESULTS/AGENT_DONE"
