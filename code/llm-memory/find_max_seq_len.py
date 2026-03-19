"""Find the maximum sequence length for inference with KV cache.

Does a single prefill forward pass with use_cache=True at each candidate
sequence length. Binary search finds the maximum that fits in GPU memory.
"""

import gc
import sys
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


from loguru import logger as log


def get_gpu_memory_info(device: str = "cuda:0") -> dict[str, float]:
    """Return current GPU memory usage in GB."""
    idx = int(device.split(":")[-1]) if ":" in device else 0
    allocated = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - reserved, 2),
    }


def cleanup() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def try_prefill(
    model: AutoModelForCausalLM,
    seq_len: int,
    device: str = "cuda:0",
    batch_size: int = 1,
    vocab_size: int = 32000,
) -> bool:
    """Attempt a single prefill forward pass with KV cache.

    Args:
        model: The causal LM (already on device, eval mode).
        seq_len: Total sequence length to test.
        device: CUDA device.
        batch_size: Number of sequences in the batch.
        vocab_size: Model vocabulary size.

    Returns:
        True if the forward pass completed without OOM.
    """
    input_ids = torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    try:
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()

        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        torch.cuda.synchronize(device)

        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        log.info(f"  [OK]  seq_len={seq_len:>6d}  bsz={batch_size}  "
            f"peak_mem={peak_mem:.2f} GB  time={elapsed:.1f}s")
        return True

    except torch.cuda.OutOfMemoryError:
        log.info(f"  [OOM] seq_len={seq_len:>6d}  bsz={batch_size}")
        return False

    finally:
        del input_ids, attention_mask
        cleanup()


def find_max_seq_len(
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda:0",
    batch_size: int = 1,
    low: int = 128,
    high: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Binary-search for the maximum inference sequence length with KV cache.

    Args:
        model_name: HuggingFace model identifier.
        device: CUDA device.
        batch_size: Batch size during inference.
        low: Lower bound for binary search on seq_len.
        high: Upper bound. If None, auto-probed by doubling until OOM.
        dtype: Model dtype.

    Returns:
        The maximum sequence length that fits in GPU memory.
    """
    log.info(f"{'='*60}")
    log.info(f"Model        : {model_name}")
    log.info(f"Device       : {device}")
    log.info(f"Batch size   : {batch_size}")
    log.info(f"Dtype        : {dtype}")
    log.info(f"GPU memory   : {get_gpu_memory_info(device)}")
    log.info(f"{'='*60}\n")

    cleanup()

    log.info("[1/4] Loading model weights to CPU …")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    log.info(f"       done in {time.time() - t0:.1f}s")

    log.info("[2/4] Moving model to GPU …")
    t0 = time.time()
    model = model.to(device)
    torch.cuda.synchronize(device)
    log.info(f"       done in {time.time() - t0:.1f}s")

    model.eval()
    vocab_size = model.config.vocab_size
    log.info(f"       GPU: {get_gpu_memory_info(device)}\n")

    log.info("[3/4] Warmup forward pass (CUDA kernel compilation) …")
    t0 = time.time()
    warmup_ids = torch.randint(100, vocab_size - 100, (1, 16), device=device)
    _ = model(input_ids=warmup_ids, use_cache=True)
    torch.cuda.synchronize(device)
    del warmup_ids, _
    cleanup()
    log.info(f"       done in {time.time() - t0:.1f}s\n")

    log.info("[4/4] Searching for max seq_len …")

    # --- Phase 1: find upper bound by doubling ---
    if high is None:
        high = low
        last_ok = low
        log.info("  Phase 1: probing upper bound …")
        while True:
            ok = try_prefill(model, high, device, batch_size, vocab_size)
            if ok:
                last_ok = high
                high *= 2
            else:
                break
        low = last_ok
        log.info(f"  Bounds: low={low}, high={high}\n")

    # --- Phase 2: binary search ---
    log.info("  Phase 2: binary search …")
    best = low
    while low <= high:
        mid = (low + high) // 2
        if try_prefill(model, mid, device, batch_size, vocab_size):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    log.info(f"\n{'='*60}")
    log.info(f"GPU memory   : {get_gpu_memory_info(device)}")
    log.info(f"{'='*60}")

    del model
    cleanup()
    return best


if __name__ == "__main__":
    configs = [
        # (model_name, batch_size, initial_low)
        ("Qwen/Qwen3-8B", 1, 256),
        ("Qwen/Qwen3-4B", 1, 512),
        ("Qwen/Qwen3-1.7B", 1, 1024),
        ("Qwen/Qwen3-0.6B", 1, 1024),
    ]

    for model_name, bsz, init_low in configs:
        try:
            find_max_seq_len(
                model_name=model_name,
                batch_size=bsz,
                low=init_low,
            )
        except Exception as e:
            log.info(f"Failed for {model_name}: {e}")
