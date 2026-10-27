# JAXBench ↔ Harbor parity: original-side runner

This folder runs the **original-side** half of the parity experiment for the
[Harbor JAXBench adapter](https://github.com/harbor-framework/adapters/pull/6): the same coding agent,
model, prompt and settings as the Harbor side, run **outside Harbor** and graded by this repository's own
`python -m JAXBench evaluate`.

Upstream JAXBench ships an evaluation harness but no agent runner, so this follows Harbor's
"Scenario 2" (fork the benchmark and add a runner for a Harbor-compatible agent).

## What is identical to the Harbor side

| | Value |
|---|---|
| Benchmark revision | upstream `7afeb78` + the discovery fix below |
| Tasks | `parity_workloads.txt`: the adapter's 15-task `--split parity` |
| Prompt | `instruction_template.md`: copied verbatim from the adapter's `task-template/instruction.md` and rendered the same way (verified identical for all 15 tasks) |
| Agent | mini-swe-agent 2.4.6, installed with `uv tool install --python 3.12 mini-swe-agent==2.4.6 --with litellm --with orjson --with fastapi`, packages resolved as of `UV_EXCLUDE_NEWER=2026-10-17T00:00:00Z` |
| Agent flags | `--yolo --model=openai/deepseek-v4.1-flash --cost-limit 0 -c mini -c model.model_kwargs.timeout=3600 -c mini_config.yaml -c model.model_kwargs.extra_headers.X-Session-ID=<trial> --exit-immediately`, run from `/workspace` (Harbor sends the same per-trial `X-Session-ID` header; the instruction is passed verbatim, including its trailing newline) |
| Agent config | `mini_config.yaml`: `step_limit: 150`, `temperature: 1.0`, `top_p: 0.95` |
| Model / endpoint | DeepSeek V4.1 Flash via `https://app-us.ppapi.ai/v1` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`) |
| Agent time limit | 3600 s (killed at the limit; the last `/workspace/kernel.py` is graded) |
| Hardware / image | 1 TPU v6e chip (GKE `ct6e-standard-1t`, topology 1x1), 8 CPU / 32 GiB memory, 20 GiB ephemeral storage; `Dockerfile` pins jax 0.12.0, jaxlib 0.12.0, libtpu 0.0.50, numpy 2.5.4 (the versions in the Harbor task image) |
| Workspace | `/workspace/baseline.py` + the harness with only this workload's `baseline.py` (no `optimized.py`) |
| Grading | `run_verify.sh` mirrors the adapter's `tests/test.sh`: the grader is copied in only after the agent finishes, leftover processes holding the TPU are stopped, then `python -m JAXBench evaluate --workload <w> --kernel /workspace/kernel.py --tpu v6e --json`, retried (same kernel) while the TPU is still being released by a just-stopped process |
| Reward | `compute_reward.py`, copied verbatim from the adapter: `reward = max(speedup, 1)` if correct else 1; plus `speedup`, `correct`, `fast_1` |

## Changes to upstream JAXBench in this fork

1. **Workload discovery** (`JAXBench/benchmark/__init__.py`): workloads moved to
   `benchmark/level1/<name>/`, but discovery only looked in `benchmark/<name>/`, so `list` found 0 workloads
   and `evaluate` failed. Discovery now also searches `benchmark/level*/`. No grading code is changed.
2. **TPU type**: the runner passes `--tpu v6e`, because `--tpu auto` looks for `"v6e"` in `device_kind`
   but a v6e reports `"TPU v6 lite"`. (Only affects the TFLOPS/utilization report, not correctness or speedup.)

## Reproduce

```bash
# 1. Build the environment image (Cloud Build -> Artifact Registry)
gcloud builds submit JAXBench/harbor_parity --project PROJECT --region us-central1 \
  --tag us-central1-docker.pkg.dev/PROJECT/REPO/jaxbench-original:v1

# 2. Run (needs a GKE cluster with a TPU v6e 1x1 node pool, kubectl + gke-gcloud-auth-plugin)
export LLM_API_KEY=...        # stored in a Kubernetes Secret via stdin, never on a command line
python3 JAXBench/harbor_parity/launch.py \
  --context gke_PROJECT_us-central1_CLUSTER \
  --image us-central1-docker.pkg.dev/PROJECT/REPO/jaxbench-original:v1 \
  --runs 3 --out parity_results/original
```

Outputs per run: `parity_results/original/run<k>/<workload>/` (trajectory, agent log, kernel,
`evaluation.json`, `reward.json`), plus `results.json` and `summary.json` (correct rate, fast₁ rate,
floored geomean speedup, mean reward). Runs whose pod or node disappears (e.g. spot preemption) are
retried up to `--retries` times; completed workloads are skipped on re-launch.

Optional flags: `--node-pool` / `--toleration` pin pods to a dedicated (tainted) node pool, `--retries` sets how
many times a run is retried after its pod or node disappears.

## Results

Parity (3 runs × 15 tasks per side, mini-swe-agent 2.4.6 + DeepSeek V4.1 Flash, mean ± sample SEM):

| Metric | Original (this runner) | Harbor adapter |
|---|---|---|
| Geomean speedup, floored (×) | 1.80 ± 0.16 | 1.60 ± 0.15 |
| Correct rate (%) | 71.11 ± 12.37 | 64.44 ± 8.89 |
| fast₁ (%) | 55.56 ± 8.01 | 53.33 ± 7.70 |
| Mean reward (×) | 2.74 ± 0.39 | 2.45 ± 0.34 |

Per-run values are in the adapter's
[`parity_experiment.json`](https://github.com/harbor-framework/adapters/blob/main/src/jaxbench/parity_experiment.json).

Harbor side, for comparison (from the adapters repository):

```bash
cd src/jaxbench && uv run jaxbench --output-dir ../../datasets/jaxbench-parity --split parity && cd ../..
uvx --from 'harbor[gke]==0.23.0' harbor run -c src/jaxbench/run_jaxbench_gke_tpu_v6e_parity.yaml   # once per run
```
