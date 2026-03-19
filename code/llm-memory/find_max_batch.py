"""Find the maximum training batch size for a given LLM at a fixed sequence length.

Uses binary search + proper memory cleanup between trials. The model is loaded
once and reused across all trials to avoid wasting time on repeated downloads.
"""

import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        "free_gb": round(total - allocated, 2),
    }


def cleanup() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_train(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    bsz: int,
    seq_len: int = 512,
    num_iter: int = 3,
    device: str = "cuda:0",
    vocab_size: int = 32000,
) -> bool:
    """Attempt a training loop with the given batch size.

    The model and optimizer are passed in (loaded once externally).
    After each trial, gradients are zeroed and CUDA cache is cleared so
    that the next trial starts from the same baseline.

    Args:
        model: The causal LM (already on device, in train mode).
        optimizer: Pre-built optimizer for the model.
        bsz: Batch size to try.
        seq_len: Fixed sequence length.
        num_iter: Number of forward+backward iterations.
        device: CUDA device.
        vocab_size: Model vocabulary size.

    Returns:
        True if training completed without OOM.
    """
    input_ids = torch.randint(100, vocab_size - 100, (bsz, seq_len + 1), device=device)
    x = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    attention_mask = torch.ones_like(x)

    try:
        torch.cuda.reset_peak_memory_stats(device)

        for i in range(num_iter):
            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs.logits

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"  [OK]  bsz={bsz:>4d}  seq_len={seq_len}  peak_mem={peak_mem:.2f} GB")
        return True

    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] bsz={bsz:>4d}  seq_len={seq_len}")
        return False

    finally:
        del input_ids, x, labels, attention_mask
        optimizer.zero_grad(set_to_none=True)
        cleanup()


def find_max_batch_size(
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda:0",
    seq_len: int = 512,
    num_iter: int = 3,
    low: int = 1,
    high: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Binary-search for the maximum training batch size.

    Args:
        model_name: HuggingFace model identifier.
        device: CUDA device.
        seq_len: Fixed sequence length for all samples.
        num_iter: Training iterations per trial.
        low: Lower bound for binary search.
        high: Upper bound. If None, auto-probed by doubling until OOM.
        dtype: Model dtype.

    Returns:
        The maximum batch size that fits in GPU memory.
    """
    print(f"{'='*60}")
    print(f"Model        : {model_name}")
    print(f"Device       : {device}")
    print(f"Seq length   : {seq_len}")
    print(f"Num iters    : {num_iter}")
    print(f"Dtype        : {dtype}")
    print(f"GPU memory   : {get_gpu_memory_info(device)}")
    print(f"{'='*60}\n")

    cleanup()

    print("Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    model.train()
    vocab_size = model.config.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    print(f"Model loaded. GPU: {get_gpu_memory_info(device)}\n")

    # --- Phase 1: find upper bound by doubling ---
    if high is None:
        high = low
        last_ok = low
        print("Phase 1: probing upper bound …")
        while True:
            ok = try_train(model, optimizer, high, seq_len, num_iter, device, vocab_size)
            if ok:
                last_ok = high
                high *= 2
            else:
                # high is the first failing value; search between last_ok and high
                break
        low = last_ok
        print(f"Bounds: low={low}, high={high}\n")

    # --- Phase 2: binary search between low and high ---
    print("Phase 2: binary search …")
    best = low
    while low <= high:
        mid = (low + high) // 2
        if try_train(model, optimizer, mid, seq_len, num_iter, device, vocab_size):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"\n{'='*60}")
    print(f"Result: max batch size = {best}  (seq_len={seq_len})")
    print(f"GPU memory   : {get_gpu_memory_info(device)}")
    print(f"{'='*60}")

    del model, optimizer
    cleanup()
    return best


if __name__ == "__main__":
    configs = [
        # (model_name, seq_len, initial_low)
        ("Qwen/Qwen3-0.6B", 512, 16),
        # ("Qwen/Qwen3-1.7B", 512, 8),
        # ("Qwen/Qwen3-4B", 512, 4),
        # ("Qwen/Qwen3-8B", 512, 2),
    ]

    for model_name, seq_len, init_low in configs:
        try:
            find_max_batch_size(
                model_name=model_name,
                seq_len=seq_len,
                low=init_low,
            )
        except Exception as e:
            print(f"Failed for {model_name}: {e}")
