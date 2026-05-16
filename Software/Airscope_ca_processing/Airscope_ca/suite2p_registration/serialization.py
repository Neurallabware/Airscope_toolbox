"""Serialization helpers for standalone registration capture/replay."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


Primitive = (type(None), bool, int, float, str)


def _is_binary_file_like(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("filename", "Ly", "Lx", "n_frames"))

# used only for BinaryFile-like objects, so we don't need to worry about recursive structures or other edge cases that would require a more complex approach.
def serialize_object(obj: Any) -> Any:
    """Serialize objects for pickle-safe, explicit replay records."""
    if isinstance(obj, Primitive):
        return obj

    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}

    if isinstance(obj, np.generic):
        return {
            "__type__": "numpy_scalar",
            "dtype": str(obj.dtype),
            "value": obj.item(),
        }

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": np.array(obj, copy=True),
        }

    if torch is not None and isinstance(obj, torch.device):
        return {"__type__": "torch_device", "type": obj.type}

    if _is_binary_file_like(obj):
        return {
            "__type__": "binary_file",
            "kind": "binary_file",
            "filename": str(obj.filename),
            "Ly": int(obj.Ly),
            "Lx": int(obj.Lx),
            "n_frames": int(obj.n_frames),
            "dtype": str(getattr(obj, "dtype", "int16")),
            "write": bool(getattr(obj, "write", False)),
        }

    if isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize_object(v) for v in obj]

    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [serialize_object(v) for v in obj]}

    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")


def deserialize_object(
    obj: Any,
    binary_file_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Any:
    """Deserialize captured objects back to runtime values."""
    if isinstance(obj, Primitive):
        return obj

    if isinstance(obj, list):
        return [deserialize_object(v, binary_file_factory=binary_file_factory) for v in obj]

    if isinstance(obj, dict):
        obj_type = obj.get("__type__")
        if obj_type == "path":
            return Path(obj["value"])
        if obj_type == "tuple":
            return tuple(
                deserialize_object(v, binary_file_factory=binary_file_factory)
                for v in obj["items"]
            )
        if obj_type == "ndarray":
            data = np.array(obj["data"], copy=True)
            data = data.astype(np.dtype(obj["dtype"]), copy=False)
            if tuple(obj["shape"]) != data.shape:
                raise ValueError(
                    f"Shape mismatch during ndarray deserialization: expected {tuple(obj['shape'])}, got {data.shape}"
                )
            return data
        if obj_type == "numpy_scalar":
            return np.dtype(obj["dtype"]).type(obj["value"])
        if obj_type == "torch_device":
            if torch is None:
                raise RuntimeError("torch is required to deserialize torch_device")
            return torch.device(obj["type"])
        if obj_type == "binary_file":
            if binary_file_factory is None:
                return obj
            return binary_file_factory(obj)
        return {
            k: deserialize_object(v, binary_file_factory=binary_file_factory)
            for k, v in obj.items()
        }

    raise TypeError(f"Unsupported serialized object type: {type(obj)!r}")
