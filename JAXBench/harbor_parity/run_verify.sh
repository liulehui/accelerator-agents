#!/bin/bash
# Verifier phase (runs inside the trial pod after the agent finishes). launch.py copies the grader
# (pristine JAXBench harness + this workload's baseline/optimized + compute_reward.py) to /grader only
# now, so the agent never sees or edits it. Mirrors the Harbor adapter's tests/test.sh.
# Usage: run_verify.sh <workload>
set -uo pipefail
WORKLOAD="$1"
RESULTS=/results
mkdir -p "$RESULTS"
echo '{"reward": 1.0, "speedup": 0.0, "correct": 0, "fast_1": 0}' > "$RESULTS/reward.json"

KERNEL=/workspace/kernel.py
RESULT="$RESULTS/evaluation.json"

if [ ! -f "$KERNEL" ]; then
  echo "No kernel found at $KERNEL"
  exit 1
fi

echo "--- Candidate kernel ---"
cat "$KERNEL"
echo "--- End candidate kernel ---"
cp "$KERNEL" "$RESULTS/kernel.py"

# The TPU can be opened by only one process; stop anything the agent left holding it.
for proc in /proc/[0-9]*; do
  pid=${proc#/proc/}
  [ "$pid" = "$$" ] && continue
  if ls -l "$proc/fd" 2>/dev/null | grep -q "/dev/vfio"; then
    echo "Stopping leftover process $pid holding the TPU: $(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null | cut -c1-120)"
    kill -9 "$pid" 2>/dev/null
  fi
done
sleep 2

cd /grader
# A killed process can keep the TPU busy for a few seconds while the device is released, so
# retry (without changing the kernel) while the evaluator reports the TPU as busy.
for attempt in 1 2 3 4 5 6; do
  timeout 1800 python -m JAXBench evaluate \
    --workload "$WORKLOAD" \
    --kernel "$KERNEL" \
    --tpu v6e \
    --json > "$RESULT"
  echo "JAXBench evaluate exit code: $?"
  grep -q "Device or resource busy" "$RESULT" || break
  echo "TPU still busy (attempt $attempt); retrying in 10 s"
  sleep 10
done
cat "$RESULT"

python /grader/compute_reward.py "$RESULT" "$RESULTS/reward.json"
