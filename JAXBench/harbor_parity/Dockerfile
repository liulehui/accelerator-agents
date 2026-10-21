# Original-side environment for Harbor parity: same base image and the exact JAX / libtpu versions
# observed in the Harbor JAXBench task image (jax 0.12.0, jaxlib 0.12.0, libtpu 0.0.50, numpy 2.5.4).
# The workspace (baseline + harness) and the grader are copied in at run time by launch.py.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "jax[tpu]==0.12.0" "jaxlib==0.12.0" "libtpu==0.0.50" "numpy==2.5.4"

WORKDIR /workspace
