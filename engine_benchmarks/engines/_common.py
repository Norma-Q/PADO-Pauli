"""Common helpers shared by all engine subprocess workers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def package_versions(*names: str) -> Dict[str, Any]:
    """Version of each named distribution, or None if it is not installed."""
    out: Dict[str, Any] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return {n: None for n in names}
    for name in names:
        out[name] = None
        for cand in (name, name.replace("_", "-"), name.replace("-", "_")):
            try:
                out[name] = version(cand)
                break
            except PackageNotFoundError:
                continue
            except Exception:
                break
    return out


def load_config(argv: List[str]) -> Dict[str, Any]:
    """Load worker config from argv[1] (input JSON path)."""
    if len(argv) < 3:
        raise SystemExit(f"usage: {argv[0]} <config.json> <result.json>")
    with open(argv[1], "r") as f:
        return json.load(f)


def save_result(argv: List[str], result: Dict[str, Any]) -> None:
    """Save worker result to argv[2] (output JSON path)."""
    out_path = argv[2]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_to_native(result), f, indent=2)


def _to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def read_peak_rss_mb() -> float:
    """Read /proc/self/status VmHWM (peak RSS in MB) on Linux.

    VmHWM is the high water mark since process start. Subtract a baseline
    reading to capture the additional peak reached during a code block.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 2)
    except ImportError:
        return 0.0


def reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter (VmHWM) for this process.

    Writing "5" to /proc/self/clear_refs resets VmHWM to the current RSS
    (Linux >= 4.0), so a subsequent read_peak_rss_mb() reports the high-water
    mark of only the code that ran since the reset. Without this, VmHWM is
    monotone over the process lifetime and per-step deltas after warm-up come
    out structurally 0.
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def read_current_rss_mb() -> float:
    """Read current RSS in MB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 2)
    except ImportError:
        return 0.0


def summarize_history(history: List[Dict[str, Any]], time_key: str = "t_eval_s",
                       mem_key: str = "step_mem_MB") -> Dict[str, Any]:
    """Aggregate per-step measurements into first/steady summary stats."""
    if not history:
        return {
            f"{time_key}_first": float("nan"),
            f"{time_key}_steady": float("nan"),
            f"{mem_key}_first": float("nan"),
            f"{mem_key}_steady": float("nan"),
            "final_n_terms": 0,
            "final_expval": float("nan"),
        }
    first = history[0]
    steady_pool = history[1:] if len(history) > 1 else history
    def _avg(key: str) -> float:
        vals = [float(h[key]) for h in steady_pool if key in h]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        f"{time_key}_first": float(first.get(time_key, float("nan"))),
        f"{time_key}_steady": _avg(time_key),
        "t_bwd_s_first": float(first.get("t_bwd_s", float("nan"))),
        "t_bwd_s_steady": _avg("t_bwd_s"),
        "batch_size": int(history[-1].get("batch_size", 1)),
        "batch_throughput_evals_s_first": float(first.get("throughput_evals_s", float("nan"))),
        "batch_throughput_evals_s_steady": _avg("throughput_evals_s"),
        "batch_throughput_train_s_first": float(first.get("throughput_train_s", float("nan"))),
        "batch_throughput_train_s_steady": _avg("throughput_train_s"),
        "native_batch": bool(history[-1].get("native_batch", False)),
        f"{mem_key}_first": float(first.get(mem_key, float("nan"))),
        f"{mem_key}_steady": _avg(mem_key),
        "final_n_terms": int(history[-1].get("n_terms", 0)),
        "final_expval": float(history[-1].get("expval", float("nan"))),
        # Mean occupancy: the absolute end-of-rep reading averaged over ALL
        # reps (rep 0 included — occupancy is a level, not a warm-up-biased
        # timing). Present only when the worker recorded absolute readings.
        **{
            f"{k}_mean": float(sum(float(h[k]) for h in history) / len(history))
            for k in ("vram_used_end_MB", "ram_used_end_MB")
            if all(k in h for h in history)
        },
    }

