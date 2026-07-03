import gc


def report_reason_counts(reports):
    reason_counts = {}
    for report in reports or []:
        key = report.reason if report.status != "success" else "success"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return reason_counts


def load_yaml_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _cuda_memory_limit_bytes(args) -> int:
    limit_gb = float(getattr(args, "cuda_memory_limit_gb", 20.0) or 0.0)
    return int(max(limit_gb, 0.0) * (1024**3))


def _torch_device_index(device_name: str) -> int:
    if not str(device_name).startswith("cuda"):
        return 0
    parts = str(device_name).split(":", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1] else 0


def _configure_cuda_memory_limit(args) -> None:
    limit_bytes = _cuda_memory_limit_bytes(args)
    if limit_bytes <= 0:
        return
    device_names = {
        str(getattr(args, "q_config_mpc_device", "cuda:0")),
        str(getattr(args, "dream_device", "cuda:0")),
        str(getattr(args, "scene_visibility_device", "cuda:0")),
    }
    sphere_device = str(getattr(args, "curobo_sphere_device", "") or "")
    if sphere_device:
        device_names.add(sphere_device)
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for device_name in sorted(device_names):
            if not device_name.startswith("cuda"):
                continue
            device_index = _torch_device_index(device_name)
            total_bytes = int(
                torch.cuda.get_device_properties(device_index).total_memory
            )
            fraction = min(1.0, limit_bytes / max(float(total_bytes), 1.0))
            torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    except Exception as exc:
        print(
            f"[cuda_memory_limit] warning: failed to set PyTorch limit: {exc}",
            flush=True,
        )


def _empty_cuda_caches(device_name: str = "cuda:0") -> None:
    try:
        import torch

        if torch.cuda.is_available() and str(device_name).startswith("cuda"):
            torch.cuda.synchronize(_torch_device_index(device_name))
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from robocasa.demos.dream import _import_comfree_backend

        wp, _, _, _ = _import_comfree_backend()
        wp.synchronize()
    except Exception:
        pass
    gc.collect()


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)
