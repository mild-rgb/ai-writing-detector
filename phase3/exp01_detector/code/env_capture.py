#!/usr/bin/env python3
"""Environment capture. Phase 2's single cheapest missed gap; closed here.

`snapshot()` returns a dict that pins everything specifiable about the machine
and libraries a run executed on. Every training record embeds it whole -- the
phase 2 lesson was that records must be self-contained, because a sidecar file
is exactly what goes missing. A copy is also written to env/<env_hash>.json so
identical environments are cheap to diff.

Nothing here chases bitwise determinism. It records what produced a number so
that a later reader knows whether a mismatch is FP wobble or a different world.
"""
import json, os, platform, subprocess, sys, hashlib


def _sh(cmd, timeout=30):
    """Run a command, return stripped stdout or None. Never raises."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout or '').strip()
        return out if out else None
    except Exception:                                   # noqa: BLE001
        return None


def pip_freeze():
    """Full `pip freeze` as a list of lines. Empty list if pip is unreachable."""
    out = _sh(f'{sys.executable} -m pip freeze --disable-pip-version-check', timeout=120)
    return sorted(out.splitlines()) if out else []


def gpu_info():
    """GPU model, driver, VRAM, and compute capability, from nvidia-smi + torch."""
    g = {'nvidia_smi_available': False}
    q = _sh('nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap '
            '--format=csv,noheader')
    if q:
        g['nvidia_smi_available'] = True
        g['devices'] = []
        for line in q.splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                g['devices'].append({'name': parts[0], 'driver_version': parts[1],
                                     'memory_total': parts[2],
                                     'compute_cap': parts[3] if len(parts) > 3 else None})
        if g['devices']:
            g['name'] = g['devices'][0]['name']
            g['driver_version'] = g['devices'][0]['driver_version']
            g['memory_total'] = g['devices'][0]['memory_total']
    g['nvidia_smi_raw'] = _sh('nvidia-smi')
    try:
        import torch
        g['torch_cuda_available'] = torch.cuda.is_available()
        g['torch_device_count'] = torch.cuda.device_count()
        if torch.cuda.is_available():
            g.setdefault('name', torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            g['torch_capability'] = f'{props.major}.{props.minor}'
            g['torch_total_memory_bytes'] = props.total_memory
            g['bf16_supported'] = torch.cuda.is_bf16_supported()
    except Exception as e:                              # noqa: BLE001
        g['torch_probe_error'] = f'{type(e).__name__}: {e}'
    return g


def library_versions():
    """Versions of the libraries that actually move numbers."""
    v = {}
    for mod in ('torch', 'transformers', 'tokenizers', 'numpy', 'datasets',
                'accelerate', 'safetensors', 'flash_attn', 'triton'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:                               # noqa: BLE001
            v[mod] = None
    try:
        import torch
        v['torch_cuda'] = torch.version.cuda
        v['torch_cudnn'] = torch.backends.cudnn.version()
        v['torch_git_version'] = torch.version.git_version
        v['torch_hip'] = torch.version.hip
    except Exception:                                   # noqa: BLE001
        pass
    return v


def torch_flags():
    """Numerics-relevant global switches. TF32 alone can move the last bits."""
    f = {}
    try:
        import torch
        f['cuda_matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        f['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32
        f['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        f['cudnn_benchmark'] = torch.backends.cudnn.benchmark
        f['float32_matmul_precision'] = torch.get_float32_matmul_precision()
        f['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    except Exception as e:                              # noqa: BLE001
        f['error'] = f'{type(e).__name__}: {e}'
    return f


RELEVANT_ENV_VARS = ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG',
                     'PYTHONHASHSEED', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS',
                     'HF_HOME', 'TRANSFORMERS_CACHE', 'COLAB_GPU', 'TORCH_COMPILE_DISABLE')


def code_fingerprint(paths):
    """sha256 of the code files that define the training path, individually and
    together. Pins the code as tightly as the data hashes pin the inputs."""
    out = {}
    h_all = hashlib.sha256()
    for p in sorted(paths):
        try:
            b = open(p, 'rb').read()
        except OSError:
            out[os.path.basename(p)] = None
            continue
        out[os.path.basename(p)] = hashlib.sha256(b).hexdigest()[:16]
        h_all.update(os.path.basename(p).encode() + b)
    out['_combined'] = h_all.hexdigest()[:16]
    return out


def snapshot(include_pip_freeze=True, code_paths=()):
    """The full environment record. Embedded verbatim in every training record."""
    env = {
        'python_version': sys.version,
        'python_version_short': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'hostname': platform.node(),
        'libraries': library_versions(),
        'gpu': gpu_info(),
        'torch_flags': torch_flags(),
        'env_vars': {k: os.environ.get(k) for k in RELEVANT_ENV_VARS},
        'nvcc': _sh('nvcc --version'),
        'code': code_fingerprint(code_paths) if code_paths else None,
    }
    env['pip_freeze'] = pip_freeze() if include_pip_freeze else None
    # Hash over everything except the free-text nvidia-smi dump, whose utilisation
    # and memory columns change second to second and would defeat de-duplication.
    stable = {k: v for k, v in env.items() if k != 'gpu'}
    stable['gpu'] = {k: v for k, v in env['gpu'].items() if k != 'nvidia_smi_raw'}
    env['env_hash'] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return env


def summary_line(env):
    """One-line human-readable digest, for run logs."""
    L, G = env['libraries'], env['gpu']
    return (f"env {env['env_hash']} | py {env['python_version_short']} "
            f"torch {L.get('torch')} (cuda {L.get('torch_cuda')}, "
            f"cudnn {L.get('torch_cudnn')}) transformers {L.get('transformers')} | "
            f"{G.get('name', 'no-gpu')} driver {G.get('driver_version', '?')}")


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    e = snapshot(code_paths=[os.path.join(here, f) for f in
                             ('train_cell.py', 'metrics.py', 'env_capture.py')])
    print(summary_line(e))
    print(json.dumps({k: v for k, v in e.items()
                      if k not in ('pip_freeze', 'gpu')}, indent=1, default=str))
    print(f"pip_freeze: {len(e['pip_freeze'])} packages")
