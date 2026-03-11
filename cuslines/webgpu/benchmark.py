#!/usr/bin/env python
"""Cross-backend benchmark for GPUStreamlines.

Runs tractography on the Stanford HARDI dataset using all available backends
(CPU, WebGPU, and optionally Metal or CUDA) and prints a comparison table
with timing, streamline count, fiber lengths, and commissural fiber count.

Usage:
    python -m cuslines.webgpu.benchmark                    # 10k seeds (default)
    python -m cuslines.webgpu.benchmark --nseeds 100000    # 100k seeds
    python -m cuslines.webgpu.benchmark --skip-cpu         # GPU-only (faster)

The script auto-detects which GPU backends are installed. On macOS with
Apple Silicon it will run both Metal and WebGPU; on Linux/Windows with
NVIDIA it will run both CUDA and WebGPU (if installed).
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from math import radians

import numpy as np

# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _get_cpu_info():
    """Return a short CPU description and core counts."""
    system = platform.system()
    machine = platform.machine()
    total = os.cpu_count() or 1

    perf_cores = total
    eff_cores = 0
    name = f"{machine} ({total} threads)"

    if system == "Darwin":
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            name = raw
        except Exception:
            pass
        try:
            perf_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True
                ).strip()
            )
            eff_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel1.logicalcpu"], text=True
                ).strip()
            )
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return name, total, perf_cores, eff_cores


def _get_gpu_info(backend):
    """Return a short GPU description for a given backend."""
    if backend == "metal":
        try:
            import Metal
            dev = Metal.MTLCreateSystemDefaultDevice()
            if dev:
                return dev.name()
        except Exception:
            pass
    elif backend == "webgpu":
        try:
            import wgpu
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            if adapter:
                info = adapter.info
                parts = [info.get("device", ""), info.get("description", "")]
                desc = " ".join(p for p in parts if p).strip()
                return desc or info.get("adapter_type", "WebGPU device")
        except Exception:
            pass
    elif backend == "cuda":
        try:
            from cuda.bindings import runtime
            err, name = runtime.cudaGetDeviceProperties(0)
            if hasattr(name, "name"):
                return name.name.decode().strip("\x00")
        except Exception:
            pass
    return backend.upper()


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def _detect_backends():
    """Return list of available backends in run order."""
    backends = []

    # Metal
    if platform.system() == "Darwin":
        try:
            import Metal
            if Metal.MTLCreateSystemDefaultDevice() is not None:
                backends.append("metal")
        except ImportError:
            pass

    # CUDA
    try:
        from cuda.bindings import runtime
        count = runtime.cudaGetDeviceCount()
        if count[1] > 0:
            backends.append("cuda")
    except (ImportError, Exception):
        pass

    # WebGPU
    try:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync()
        if adapter is not None:
            backends.append("webgpu")
    except (ImportError, Exception):
        pass

    return backends


def _import_backend(name):
    """Import and return (GPUTracker, BootDirectionGetter) for a backend."""
    if name == "metal":
        from cuslines.metal import (
            MetalGPUTracker as GPUTracker,
            MetalBootDirectionGetter as BootDirectionGetter,
        )
    elif name == "cuda":
        from cuslines.cuda_python import (
            GPUTracker,
            BootDirectionGetter,
        )
    elif name == "webgpu":
        from cuslines.webgpu import (
            WebGPUTracker as GPUTracker,
            WebGPUBootDirectionGetter as BootDirectionGetter,
        )
    else:
        raise ValueError(f"Unknown backend: {name}")
    return GPUTracker, BootDirectionGetter


# ---------------------------------------------------------------------------
# Data loading (shared across backends)
# ---------------------------------------------------------------------------

def load_hardi():
    """Load Stanford HARDI dataset. Downloads automatically on first run."""
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere, get_fnames, read_stanford_pve_maps
    from dipy.io import read_bvals_bvecs
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    import nibabel as nib

    print("Loading Stanford HARDI dataset...")
    nifti, bval, bvec = get_fnames(name="stanford_hardi")
    _, _, wm = read_stanford_pve_maps()

    img = nib.load(nifti)
    data = img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    wm_data = wm.get_fdata()
    roi_data = wm_data > 0.5

    print("Fitting tensor model...")
    tenfit = dti.TensorModel(gtab, fit_method="WLS").fit(data, mask=roi_data)
    FA = tenfit.fa
    classifier = ThresholdStoppingCriterion(FA, 0.1)
    sphere = default_sphere

    return img, data, gtab, FA, roi_data, classifier, sphere


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def compute_metrics(sft, ref_img):
    """Compute streamline statistics from a StatefulTractogram."""
    streamlines = sft.streamlines
    n = len(streamlines)
    if n == 0:
        return {"n_streamlines": 0}

    lengths = np.array([len(sl) for sl in streamlines])

    # Commissural fibers: streamlines that cross the volume midline in x
    import nibabel as nib
    dim = ref_img.shape[0]
    midx = dim / 2.0
    n_comm = 0
    for sl in streamlines:
        xs = sl[:, 0]
        if xs.min() < midx and xs.max() > midx:
            n_comm += 1

    return {
        "n_streamlines": n,
        "mean_pts": float(lengths.mean()),
        "median_pts": float(np.median(lengths)),
        "min_pts": int(lengths.min()),
        "max_pts": int(lengths.max()),
        "commissural": n_comm,
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_cpu(data, gtab, FA, classifier, sphere, seeds, img, **kw):
    """Run CPU (DIPY) tractography. Single-threaded."""
    from dipy.direction import BootDirectionGetter as cpu_BootDG
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.reconst.shm import OpdtModel
    from dipy.tracking.local_tracking import LocalTracking

    sh_order = kw.get("sh_order", 4)
    max_angle = kw.get("max_angle_deg", 60)
    step_size = kw.get("step_size", 0.5)
    rel_peak = kw.get("relative_peak_threshold", 0.25)
    min_sep = kw.get("min_separation_angle_deg", 45)

    model = OpdtModel(
        gtab, sh_order_max=sh_order, smooth=0.006, min_signal=1.0
    )
    dg = cpu_BootDG.from_data(
        data,
        model,
        max_angle=max_angle,
        sphere=sphere,
        sh_order=sh_order,
        relative_peak_threshold=rel_peak,
        min_separation_angle=min_sep,
    )

    t0 = time.time()
    gen = LocalTracking(
        dg, classifier, seeds, affine=np.eye(4), step_size=step_size
    )
    sft = StatefulTractogram(gen, img, Space.VOX)
    _ = len(sft.streamlines)  # force evaluation
    elapsed = time.time() - t0

    return sft, elapsed


def run_gpu(backend, data, gtab, FA, classifier, sphere, seeds, img, **kw):
    """Run GPU tractography on a given backend."""
    GPUTracker, BootDG = _import_backend(backend)

    sh_order = kw.get("sh_order", 4)
    max_angle = radians(kw.get("max_angle_deg", 60))
    step_size = kw.get("step_size", 0.5)
    rel_peak = kw.get("relative_peak_threshold", 0.25)
    min_sep = radians(kw.get("min_separation_angle_deg", 45))
    chunk_size = kw.get("chunk_size", 100000)

    dg = BootDG.from_dipy_opdt(
        gtab, sphere, sh_order_max=sh_order, sh_lambda=0.006, min_signal=1.0
    )

    with GPUTracker(
        dg,
        data,
        FA,
        0.1,
        sphere.vertices,
        sphere.edges,
        max_angle=max_angle,
        step_size=step_size,
        relative_peak_thresh=rel_peak,
        min_separation_angle=min_sep,
        ngpus=1,
        rng_seed=0,
        chunk_size=chunk_size,
    ) as tracker:
        t0 = time.time()
        sft = tracker.generate_sft(seeds, img)
        elapsed = time.time() - t0

    return sft, elapsed


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(results, cpu_name, gpu_name):
    """Print a formatted comparison table."""
    headers = ["", *results.keys()]
    sep = "-" * 76

    print()
    print(sep)
    print(f"  {'Backend':<14}", end="")
    for name in results:
        print(f"  {name:>14}", end="")
    print()
    print(sep)

    rows = [
        ("Time", "time", "{:.1f} s"),
        ("Speedup vs CPU", "speedup", "{:.0f}x"),
        ("Streamlines", "n_streamlines", "{:,}"),
        ("Mean length", "mean_pts", "{:.1f} pts"),
        ("Median length", "median_pts", "{:.1f} pts"),
        ("Max length", "max_pts", "{:,} pts"),
        ("Commissural", "commissural", "{:,}"),
    ]

    for label, key, fmt in rows:
        print(f"  {label:<14}", end="")
        for name, m in results.items():
            val = m.get(key)
            if val is None:
                cell = "-"
            else:
                cell = fmt.format(val)
            print(f"  {cell:>14}", end="")
        print()

    print(sep)

    print()
    print(f"  CPU: {cpu_name} (single-threaded)")
    print(f"  GPU: {gpu_name}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPUStreamlines across all available backends."
    )
    parser.add_argument(
        "--nseeds",
        type=int,
        default=10000,
        help="Number of seeds (default: 10000)",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip the CPU (DIPY) benchmark (it can be very slow at high seed counts)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="*",
        default=None,
        help="GPU backends to test (default: all available). Choices: metal, cuda, webgpu",
    )
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(0)

    # Hardware info
    cpu_name, total_threads, perf_cores, eff_cores = _get_cpu_info()
    core_info = f"{perf_cores}P" if eff_cores else f"{total_threads} threads"
    if eff_cores:
        core_info += f"+{eff_cores}E"

    # Detect GPU backends
    available = _detect_backends()
    if args.backends:
        gpu_backends = [b for b in args.backends if b in available]
        missing = [b for b in args.backends if b not in available]
        if missing:
            print(f"WARNING: backends not available: {', '.join(missing)}")
    else:
        gpu_backends = available

    if not gpu_backends and args.skip_cpu:
        print("ERROR: No backends to test. Install a GPU backend or remove --skip-cpu.")
        sys.exit(1)

    gpu_name = _get_gpu_info(gpu_backends[0]) if gpu_backends else "none"

    print(f"CPU:      {cpu_name} ({core_info})")
    print(f"GPU:      {gpu_name}")
    print(f"Backends: {', '.join(gpu_backends) if gpu_backends else 'none'}")
    print(f"Seeds:    {args.nseeds:,}")
    print()

    # Load data
    img, data, gtab, FA, roi_data, classifier, sphere = load_hardi()

    # Generate seeds
    from dipy.tracking import utils
    seeds = np.asarray(
        utils.random_seeds_from_mask(
            roi_data,
            seeds_count=args.nseeds,
            seed_count_per_voxel=False,
            affine=np.eye(4),
        )
    )
    print(f"Generated {seeds.shape[0]:,} seeds")
    print()

    tracking_params = dict(
        sh_order=4,
        max_angle_deg=60,
        step_size=0.5,
        relative_peak_threshold=0.25,
        min_separation_angle_deg=45,
        chunk_size=100000,
    )

    results = {}

    # CPU benchmark
    if not args.skip_cpu:
        print("Running CPU (DIPY, single-threaded)...")
        sft, elapsed = run_cpu(
            data, gtab, FA, classifier, sphere, seeds, img, **tracking_params
        )
        metrics = compute_metrics(sft, img)
        metrics["time"] = elapsed
        metrics["speedup"] = 1.0
        results["CPU"] = metrics
        print(
            f"  -> {metrics['n_streamlines']:,} streamlines in {elapsed:.1f}s"
        )

    # GPU benchmarks
    for backend in gpu_backends:
        label = backend.upper()
        if label == "WEBGPU":
            label = "WebGPU"
        print(f"Running {label}...")
        sft, elapsed = run_gpu(
            backend, data, gtab, FA, classifier, sphere, seeds, img,
            **tracking_params,
        )
        metrics = compute_metrics(sft, img)
        metrics["time"] = elapsed
        if "CPU" in results:
            metrics["speedup"] = results["CPU"]["time"] / elapsed
        results[label] = metrics
        print(
            f"  -> {metrics['n_streamlines']:,} streamlines in {elapsed:.1f}s"
        )

    # Print results
    print_table(results, f"{cpu_name} ({core_info})", gpu_name)


if __name__ == "__main__":
    main()
