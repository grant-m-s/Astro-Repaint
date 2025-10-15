"""
Helpers for distributed training (no mpi4py).
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

class TorchComm:
    """
    Minimal comm adapter for logger:
      - .rank (int)
      - .gather(obj) -> list on rank 0, None on others
    """
    def __init__(self):
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("TorchComm requires torch.distributed to be initialized")
        self.rank = dist.get_rank()

    def gather(self, obj, dst: int = 0):
        world_size = dist.get_world_size()
        if self.rank == dst:
            gather_list = [None] * world_size
            dist.gather_object(obj, gather_list, dst=dst)
            return gather_list
        else:
            dist.gather_object(obj, None, dst=dst)
            return None

def _get_backend():
    return "nccl" if th.cuda.is_available() else "gloo"


def _is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if _is_dist_ready():
        try:
            return dist.get_rank()
        except Exception:
            pass
    # Fallback to env or single-process default
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    if _is_dist_ready():
        try:
            return dist.get_world_size()
        except Exception:
            pass
    return int(os.environ.get("WORLD_SIZE", "1"))


def setup_dist():
    """
    Setup a torch.distributed process group if environment variables are provided.
    If not provided, assume single-process mode and do nothing.
    Compatible with torchrun:
      torchrun --nproc_per_node=... --master_port=... your_script.py
    """
    if _is_dist_ready():
        return

    # Respect torchrun's LOCAL_RANK for device placement
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and th.cuda.is_available():
        try:
            th.cuda.set_device(int(local_rank))
        except Exception:
            pass

    # If torchrun/launcher env vars exist, initialize using them.
    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    if all(k in os.environ for k in required):
        backend = _get_backend()
        dist.init_process_group(backend=backend, init_method="env://")
        return

    # Otherwise, assume single-process. Set sane defaults so downstream code can read them.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def dev():
    """
    Get the device to use for torch.distributed (or single process).
    """
    if th.cuda.is_available():
        # If torchrun set LOCAL_RANK, we already pointed CUDA at the right device in setup_dist().
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file. If distributed is initialized, rank 0 reads the file once
    and broadcasts raw bytes to all other ranks to avoid redundant fetches.
    In single-process mode, this is just a normal load.
    """
    if not _is_dist_ready() or get_world_size() == 1:
        with bf.BlobFile(path, "rb") as f:
            return th.load(f, **kwargs)

    rank = get_rank()

    # Rank 0 reads bytes, then broadcast size and data as tensors.
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        n = th.tensor([len(data)], dtype=th.long)
    else:
        data = None
        n = th.tensor([0], dtype=th.long)

    # Broadcast the length first.
    dist.broadcast(n, src=0)
    n_bytes = int(n.item())

    # Create a tensor buffer for the payload on all ranks.
    if rank == 0:
        # Create a contiguous uint8 tensor from bytes without extra copies.
        t = th.frombuffer(memoryview(data), dtype=th.uint8).clone()
        # Ensure size matches n_bytes
        if t.numel() != n_bytes:
            t = t.reshape(-1)[:n_bytes]
    else:
        t = th.empty(n_bytes, dtype=th.uint8)

    # Broadcast the payload.
    dist.broadcast(t, src=0)

    # Reconstruct BytesIO and load.
    buf = io.BytesIO(bytearray(t.tolist()))
    return th.load(buf, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    No-op in single-process mode.
    """
    if not _is_dist_ready() or get_world_size() == 1:
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, src=0)


def _find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def is_dist_ready():
    return dist.is_available() and dist.is_initialized()

def world_size():
    # if you already have get_world_size(), reuse that instead
    try:
        return dist.get_world_size() if is_dist_ready() else 1
    except Exception:
        return 1

def barrier():
    """Safe barrier: only sync when a real multi-process group exists."""
    if is_dist_ready() and world_size() > 1:
        dist.barrier()