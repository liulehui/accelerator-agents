"""JAXBench workload discovery and loading."""

import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def _workload_dirs():
    """Map workload name -> directory.

    Workloads live in benchmark/<level>/<name>/ (e.g. benchmark/level1/8p_GEMM/);
    benchmark/<name>/ is still accepted. Earlier levels win on a name clash.
    """
    found = {}
    roots = [BENCHMARK_DIR] + sorted(
        os.path.join(BENCHMARK_DIR, d) for d in os.listdir(BENCHMARK_DIR)
        if d.startswith('level') and os.path.isdir(os.path.join(BENCHMARK_DIR, d))
    )
    for root in roots:
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if (os.path.isdir(full)
                    and not d.startswith(('_', '.'))
                    and os.path.exists(os.path.join(full, 'baseline.py'))):
                found.setdefault(d, full)
    return found


def list_workloads():
    """Return sorted list of workload names (e.g. ['1p_Flash_Attention', ...])."""
    return sorted(_workload_dirs(), key=_sort_key)


def get_workload_dir(name):
    """Return absolute path to workload directory. Raises ValueError if not found."""
    path = _workload_dirs().get(name)
    if path is None:
        raise ValueError(f"Workload '{name}' not found in {BENCHMARK_DIR}")
    return path


def has_optimized(name):
    """Check if a workload has an optimized.py variant."""
    path = _workload_dirs().get(name)
    return path is not None and os.path.exists(os.path.join(path, 'optimized.py'))


def _sort_key(name):
    """Sort by numeric prefix (1p before 18k, etc.)."""
    num = ''
    for c in name:
        if c.isdigit():
            num += c
        else:
            break
    return int(num) if num else 999
