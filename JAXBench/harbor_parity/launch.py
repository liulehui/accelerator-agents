#!/usr/bin/env python3
"""Original-side runner for Harbor parity: mini-swe-agent on JAXBench, graded by upstream `evaluate`.

For each workload (and run) it starts a pod on a TPU v6e 1x1 GKE node and mirrors Harbor's trial flow:
  1. copy the agent workspace into /workspace: baseline.py + the JAXBench harness with only this
     workload's baseline.py (no optimized.py, no other workloads);
  2. agent phase: run_agent.sh (same mini-swe-agent install, flags, config and prompt as Harbor);
  3. copy the grader into /grader only after the agent finished (pristine harness + baseline +
     optimized.py + compute_reward.py), then run_verify.sh (upstream `python -m JAXBench evaluate`);
  4. copy /results back and delete the pod.
A run whose pod or node disappears (e.g. spot preemption) is retried up to --retries times.

Requires kubectl (+ gke-gcloud-auth-plugin) and LLM_API_KEY in the environment. The key is stored in a
Kubernetes Secret via stdin, never on a command line.

Example:
  export LLM_API_KEY=...   # e.g. source ~/.config/harbor-jaxbench/llm.env
  python3 JAXBench/harbor_parity/launch.py \\
      --context gke_PROJECT_REGION_CLUSTER \\
      --image REGION-docker.pkg.dev/PROJECT/REPO/jaxbench-original:TAG \\
      --runs 1 --out parity_results/original
"""

import argparse
import json
import math
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAXBENCH = HERE.parent  # .../JAXBench
HARNESS_FILES = [
    "__init__.py", "__main__.py", "benchmark/__init__.py",
    "harness/__init__.py", "harness/correctness.py", "harness/evaluator.py", "harness/loader.py",
    "harness/profiler.py", "harness/runner.py", "harness/tpu_specs.py",
]
SECRET = "jaxbench-parity-llm"
BASE_URL = "https://app-us.ppapi.ai/v1"
POD_READY_TIMEOUT = 1200   # Harbor's TPU pod-ready allowance
AGENT_PHASE_TIMEOUT = 3600 + 1800  # agent limit (enforced in-pod) + generous install/slack
VERIFY_TIMEOUT = 1800 + 300


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


class PodGone(Exception):
    """The pod or its node disappeared (e.g. spot preemption)."""


class Kube:
    def __init__(self, context, namespace):
        self.base = ["kubectl", "--context", context, "-n", namespace]

    def run(self, *args, input_text=None, check=True, timeout=600):
        r = subprocess.run(self.base + list(args), input=input_text, capture_output=True, text=True,
                           timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args[:3])} failed: {r.stderr.strip()[:500]}")
        return r

    def phase(self, pod):
        r = self.run("get", "pod", pod, "-o", "jsonpath={.status.phase}", check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def exec(self, pod, cmd, timeout=600, check=True):
        return self.run("exec", pod, "--", "bash", "-c", cmd, check=check, timeout=timeout)

    def alive(self, pod):
        return self.phase(pod) == "Running"


def ensure_secret(kube):
    key = os.environ.get("LLM_API_KEY")
    if not key:
        sys.exit("LLM_API_KEY is not set")
    manifest = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": SECRET},
                "type": "Opaque", "stringData": {"OPENAI_API_KEY": key}}
    kube.run("apply", "-f", "-", input_text=json.dumps(manifest))


def pod_manifest(name, image):
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "labels": {"app": "jaxbench-parity-original"}},
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 5,
            "nodeSelector": {"cloud.google.com/gke-tpu-accelerator": "tpu-v6e-slice",
                             "cloud.google.com/gke-tpu-topology": "1x1"},
            "tolerations": [{"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}],
            "containers": [{
                "name": "main", "image": image, "command": ["sleep", "infinity"],
                "resources": {"requests": {"cpu": "8", "memory": "32Gi", "google.com/tpu": "1"},
                              "limits": {"google.com/tpu": "1"}},
                "env": [{"name": "OPENAI_BASE_URL", "value": BASE_URL},
                        {"name": "OPENAI_API_KEY",
                         "valueFrom": {"secretKeyRef": {"name": SECRET, "key": "OPENAI_API_KEY"}}}],
            }],
        },
    }


def stage_harness(dest, workload_dir, include_optimized):
    for rel in HARNESS_FILES:
        (dest / "JAXBench" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(JAXBENCH / rel, dest / "JAXBench" / rel)
    wl = dest / "JAXBench" / workload_dir.relative_to(JAXBENCH)
    wl.mkdir(parents=True)
    shutil.copy2(workload_dir / "baseline.py", wl / "baseline.py")
    if include_optimized and (workload_dir / "optimized.py").exists():
        shutil.copy2(workload_dir / "optimized.py", wl / "optimized.py")


def wait_for(kube, pod, cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ph = kube.phase(pod)
        if ph in (None, "Failed", "Succeeded"):
            raise PodGone(f"pod {pod} is {ph or 'gone'} while waiting for {what}")
        if cond(ph):
            return
        time.sleep(30)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


def run_one(kube, image, workload, workload_dir, out_dir):
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    pod = f"jbo-{workload.lower().replace('_', '-')[:40]}-{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {"pod": pod, "started_at": time.strftime("%FT%T")}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ws, grader, parity = tmp / "workspace", tmp / "grader", tmp / "parity"
        ws.mkdir(); grader.mkdir(); parity.mkdir()
        shutil.copy2(workload_dir / "baseline.py", ws / "baseline.py")
        stage_harness(ws, workload_dir, include_optimized=False)
        stage_harness(grader, workload_dir, include_optimized=True)
        for f in ("compute_reward.py", "run_verify.sh"):
            shutil.copy2(HERE / f, grader / f)
        for f in ("instruction_template.md", "mini_config.yaml", "run_agent.sh"):
            shutil.copy2(HERE / f, parity / f)

        kube.run("apply", "-f", "-", input_text=json.dumps(pod_manifest(pod, image)))
        try:
            wait_for(kube, pod, lambda ph: ph == "Running", POD_READY_TIMEOUT, "pod Running")
            kube.exec(pod, "mkdir -p /workspace /opt/parity /results")
            kube.run("cp", f"{ws}/.", f"{pod}:/workspace")
            kube.run("cp", f"{parity}/.", f"{pod}:/opt/parity")

            log(f"{workload}: agent phase on {pod}")
            kube.exec(pod, f"nohup bash /opt/parity/run_agent.sh {workload} > /results/run_agent.out 2>&1 &")
            wait_for(kube, pod, lambda ph: kube.exec(pod, "test -f /results/AGENT_DONE", check=False).returncode == 0,
                     AGENT_PHASE_TIMEOUT, "agent phase")

            log(f"{workload}: verifier phase")
            kube.exec(pod, "mkdir -p /grader")
            kube.run("cp", f"{grader}/.", f"{pod}:/grader")
            r = kube.exec(pod, f"bash /grader/run_verify.sh {workload}", timeout=VERIFY_TIMEOUT, check=False)
            (out_dir / "verifier-stdout.txt").write_text(r.stdout + r.stderr)
            if not kube.alive(pod):
                raise PodGone(f"pod {pod} disappeared during verification")
            kube.run("cp", f"{pod}:/results/.", str(out_dir), timeout=900)
        finally:
            kube.run("delete", "pod", pod, "--wait=false", check=False)
    timing["finished_at"] = time.strftime("%FT%T")
    (out_dir / "pod.json").write_text(json.dumps(timing, indent=2))
    return json.loads((out_dir / "reward.json").read_text())


def summarize(results):
    v = [r for r in results if r.get("reward") is not None]
    n = len(v)
    if not n:
        return {"n_tasks": 0}
    floored = [max(float(r["reward"]), 1.0) for r in v]
    return {
        "n_tasks": n,
        "correct_rate": sum(int(r["correct"]) for r in v) / n,
        "fast_1_rate": sum(int(r["fast_1"]) for r in v) / n,
        "geomean_speedup": math.exp(sum(math.log(x) for x in floored) / n),
        "mean_reward": sum(floored) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True, help="kubectl context of the GKE cluster with the v6e node")
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--image", required=True, help="image built from JAXBench/harbor_parity/Dockerfile")
    ap.add_argument("--workloads", default=str(HERE / "parity_workloads.txt"))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3, help="retries per run when the pod/node disappears")
    ap.add_argument("--out", default="parity_results/original")
    args = ap.parse_args()

    from JAXBench.benchmark import get_workload_dir  # noqa: E402  (run from the repo root)

    kube = Kube(args.context, args.namespace)
    ensure_secret(kube)
    workloads = [w.strip() for w in Path(args.workloads).read_text().split() if w.strip()]
    out_root = Path(args.out)
    for run in range(1, args.runs + 1):
        results = []
        for w in workloads:
            out_dir = out_root / f"run{run}" / w
            if (out_dir / "reward.json").exists() and (out_dir / "pod.json").exists():
                results.append({"workload": w, **json.loads((out_dir / "reward.json").read_text())})
                log(f"run {run} {w}: already done, skipping")
                continue
            reward = None
            for attempt in range(args.retries + 1):
                try:
                    reward = run_one(kube, args.image, w, Path(get_workload_dir(w)), out_dir)
                    break
                except (PodGone, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as e:
                    log(f"run {run} {w}: attempt {attempt + 1} failed: {e}")
                    shutil.rmtree(out_dir, ignore_errors=True)
                    time.sleep(min(60 * 2 ** attempt, 300))
            log(f"run {run} {w}: {reward}")
            results.append({"workload": w, **(reward or {"reward": None})})
            (out_root / f"run{run}" / "results.json").write_text(json.dumps(results, indent=2))
        summary = summarize(results)
        (out_root / f"run{run}" / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"run {run} summary: {summary}")


if __name__ == "__main__":
    main()
