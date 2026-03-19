"""Plot seq_len vs peak GPU memory for different Qwen3 models.

Extracts data from inference.txt and produces a publication-quality chart.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Extracted data from inference.txt (only successful [OK] runs) ─────────────
# Format: {model_name: [(seq_len, peak_mem_gb), ...]}

data: dict[str, list[tuple[int, float]]] = {
    "Qwen3-8B": [
        (256, 15.38),
        (512, 15.49),
        (1024, 15.70),
        (2048, 16.14),
        (4096, 17.02),
        (8192, 18.77),
        (16384, 22.28),
        (32768, 29.29),
        (65536, 43.31),
        (98304, 57.34),
        (106496, 60.84),
        (110592, 62.60),
        (112640, 63.47),
        (113664, 63.91),
        (114176, 64.13),
        (114432, 64.24),
        (114560, 64.30),
        (114624, 64.32),
        (114656, 64.34),
        (114672, 64.34),
        (114680, 64.35),
        (114684, 64.35),
        (114686, 64.35),
        (114687, 64.35),
    ],
    "Qwen3-4B": [
        (512, 7.77),
        (1024, 8.01),
        (2048, 8.42),
        (4096, 9.30),
        (8192, 11.04),
        (16384, 14.52),
        (32768, 21.48),
        (65536, 35.41),
        (98304, 49.34),
        (114688, 56.31),
        (118784, 58.05),
        (120832, 58.92),
        (121856, 59.36),
        (122368, 59.57),
        (122624, 59.68),
        (122752, 59.74),
        (122816, 59.76),
        (122848, 59.78),
        (122864, 59.79),
        (122872, 59.79),
        (122876, 59.79),
        (122878, 59.79),
        (122879, 59.79),
    ],
    "Qwen3-1.7B": [
        (1024, 3.62),
        (2048, 4.02),
        (4096, 4.83),
        (8192, 6.44),
        (16384, 9.66),
        (32768, 16.11),
        (65536, 29.01),
        (131072, 54.81),
        (163839, 67.71),
        (172031, 70.93),
        (174079, 71.74),
        (174591, 71.94),
        (174719, 71.99),
        (174751, 72.01),
        (174759, 72.01),
        (174761, 72.01),
        (174762, 72.01),
    ],
    "Qwen3-0.6B": [
        (1024, 1.52),
        (2048, 1.92),
        (4096, 2.72),
        (8192, 4.33),
        (16384, 7.54),
        (32768, 13.95),
        (65536, 26.79),
        (131072, 52.46),
        (163839, 65.30),
        (172031, 68.51),
        (174079, 69.31),
        (174591, 69.51),
        (174719, 69.56),
        (174783, 69.60),
        (174799, 69.60),
        (174803, 69.60),
        (174804, 69.60),
    ],
}

# ── Model weight sizes (from GPU allocated after loading) ─────────────────────
model_weights_gb: dict[str, float] = {
    "Qwen3-8B": 15.26,
    "Qwen3-4B": 7.55,
    "Qwen3-1.7B": 3.21,
    "Qwen3-0.6B": 1.12,
}

model_config: dict[str, dict[str, int]] = {
    "Qwen3-8B": {
        "kv_heads": 8,
        "head_dim": 64,
        "num_layers": 128,
    },
    "Qwen3-4B": {
        "kv_heads": 8,
        "head_dim": 128,
        "num_layers": 36,
    },
    "Qwen3-1.7B": {
        "kv_heads": 8,
        "head_dim": 128,
        "num_layers": 28,
    },
    "Qwen3-0.6B": {
        "kv_heads": 8,
        "head_dim": 128,
        "num_layers": 28,
    },
}


def compute_theory_peak_mem(s, num_layers, kv_heads, head_dim) -> float:
    # 4 means 2 bytes for each float16 and key, value
    return (4 * s * num_layers * kv_heads * head_dim) / 1024**3


theory_peak_mems: dict[str, list[float]] = {}
for model_name, config in model_config.items():
    seq_lens = [x[0] for x in data[model_name]]
    theory_peak_mems[model_name] = [
        compute_theory_peak_mem(s, config["num_layers"], config["kv_heads"], config["head_dim"]) for s in seq_lens
    ]

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 75)
print(f"{'Model':<15} {'Weights (GB)':>12} {'Max SeqLen':>12} {'Peak Mem (GB)':>14} {'Max SeqLen (K)':>14}")
print("-" * 75)
for model_name, points in data.items():
    max_sl, max_mem = points[-1]
    w = model_weights_gb[model_name]
    print(f"{model_name:<15} {w:>12.2f} {max_sl:>12,} {max_mem:>14.2f} {max_sl / 1024:>14.1f}")
print("=" * 75)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

colors = {
    "Qwen3-8B": "#e74c3c",
    "Qwen3-4B": "#3498db",
    "Qwen3-1.7B": "#2ecc71",
    "Qwen3-0.6B": "#9b59b6",
}
markers = {
    "Qwen3-8B": "o",
    "Qwen3-4B": "s",
    "Qwen3-1.7B": "^",
    "Qwen3-0.6B": "D",
}

GPU_TOTAL = 79.33

# ── Left: Full view (linear x-axis, focus on power-of-2 sequence lengths) ────
for model_name, points in data.items():
    # Keep only power-of-2 seq_lens + the final max point for clarity
    filtered = [(s, m) for s, m in points if (s & (s - 1)) == 0 or s == points[-1][0]]
    seq_lens = [s for s, _ in filtered]
    mems = [m for _, m in filtered]
    theory_peak_mems = [
        model_weights_gb[model_name]
        + compute_theory_peak_mem(s, config["num_layers"], config["kv_heads"], config["head_dim"])
        for s in seq_lens
    ]
    ax1.plot(
        np.array(seq_lens) / 1024,
        mems,
        color=colors[model_name],
        marker=markers[model_name],
        markersize=6,
        linewidth=2,
        label=f"{model_name} (weights={model_weights_gb[model_name]:.1f} GB)",
    )
    ax1.plot(
        np.array(seq_lens) / 1024,
        theory_peak_mems,
        color=colors[model_name],
        linestyle="--",
        linewidth=1,
        label=f"{model_name} (theory)",
        alpha=0.5,
    )

ax1.axhline(y=GPU_TOTAL, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"GPU total ({GPU_TOTAL} GB)")
ax1.set_xlabel("Sequence Length (K tokens)", fontsize=13)
ax1.set_ylabel("Peak GPU Memory (GB)", fontsize=13)
ax1.set_title("Prefill Peak Memory vs Sequence Length\n(batch_size=1, bf16, single A100 80GB)", fontsize=14)
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0, top=85)

# ── Right: Log-scale x-axis for detail at small seq_lens ─────────────────────
for model_name, points in data.items():
    filtered = [(s, m) for s, m in points if (s & (s - 1)) == 0 or s == points[-1][0]]
    seq_lens = [s for s, _ in filtered]
    mems = [m for _, m in filtered]
    ax2.plot(
        seq_lens,
        mems,
        color=colors[model_name],
        marker=markers[model_name],
        markersize=6,
        linewidth=2,
        label=f"{model_name}",
    )

ax2.axhline(y=GPU_TOTAL, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"GPU total ({GPU_TOTAL} GB)")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Sequence Length (log₂ scale)", fontsize=13)
ax2.set_ylabel("Peak GPU Memory (GB)", fontsize=13)
ax2.set_title("Prefill Peak Memory vs Sequence Length (log scale)\n(batch_size=1, bf16, single A100 80GB)", fontsize=14)
ax2.legend(fontsize=10, loc="upper left")
ax2.grid(True, alpha=0.3, which="both")
ax2.set_ylim(bottom=0, top=85)

tick_vals = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
ax2.set_xticks(tick_vals)
ax2.set_xticklabels([f"{v // 1024}K" if v >= 1024 else str(v) for v in tick_vals], rotation=45, fontsize=9)

plt.tight_layout()
plt.savefig("seq_len_memory_curve.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to: seq_len_memory_curve.png")


# To visualize where the dynamic memory curve (Peak Memory minus weight memory) crosses the available GPU memory,
# we plot a dynamic memory curve and mark the cross-over (seq_len where theoretical dynamic memory exceeds space left by weights).

# Assumptions:
# - GPU_TOTAL: Already defined as the total GPU memory in GB
# - theory_peak_mems: List of theory peak memories (incl. weights) per model
# - theory_weight_mem: Dict mapping model_name to static weight mem (in GB)
# - The theoretical peak memory = weight_mem + dynamic_mem (from model's profile)

# First, we compute, for each model, the crossover point.
fig3, ax3 = plt.subplots(figsize=(7, 4.5))
for model_name, points in data.items():
    # Compute dynamic memory at each seq_len
    seq_lens = np.array([s for s, _ in points])
    peak_mems = np.array([m for _, m in points])
    dynamic_mems = peak_mems - model_weights_gb[model_name]

    # The limit for dynamic memory = GPU_TOTAL - weight_mem
    available_dyn = GPU_TOTAL - model_weights_gb[model_name]

    ax3.hlines(
        y=model_weights_gb[model_name],
        xmin=0,
        xmax=seq_lens[-1],
        color=colors[model_name],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"{model_name} weights ({model_weights_gb[model_name]:.2f} GB)",
    )

    # Plot the dynamic memory usage curve
    ax3.plot(
        seq_lens,
        dynamic_mems,
        color=colors[model_name],
        label=f"{model_name} (dynamic)",
        marker=markers[model_name],
        markersize=4,
        linewidth=2,
    )

    # Find the crossing (interpolate between points where the dynamic memory crosses the available dynamic memory)
    cross_idx = np.where(dynamic_mems >= available_dyn)[0]
    if len(cross_idx) == 0:
        continue
    first_cross = cross_idx[0]
    if first_cross == 0:
        cross_seq = seq_lens[0]
    else:
        # linear interpolation
        s0, s1 = seq_lens[first_cross - 1], seq_lens[first_cross]
        d0, d1 = dynamic_mems[first_cross - 1], dynamic_mems[first_cross]
        if d1 == d0:
            cross_seq = s0
        else:
            cross_seq = s0 + (available_dyn - d0) * (s1 - s0) / (d1 - d0)
    # Highlight cross point
    ax3.scatter(
        [cross_seq], [available_dyn], color=colors[model_name], marker="X", s=100, edgecolors="black", zorder=10
    )
    ax3.annotate(
        f"{int(cross_seq):,}",
        (cross_seq, available_dyn),
        textcoords="offset points",
        xytext=(10, 0),
        ha="left",
        va="center",
        fontsize=9,
        color=colors[model_name],
    )

ax3.set_xlabel("Sequence Length", fontsize=13)
ax3.set_ylabel("Dynamic Memory Usage (GB)", fontsize=13)
ax3.set_title("Cross Point of Dynamic Memory and Total Memory minus Weights", fontsize=14)
ax3.set_xscale("log", base=2)
ax3.grid(True, alpha=0.3, which="both")
ax3.legend(fontsize=9, loc="upper left")

tick_vals = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
ax3.set_xticks(tick_vals)
ax3.set_xticklabels([f"{v // 1024}K" if v >= 1024 else str(v) for v in tick_vals], rotation=45, fontsize=9)

plt.tight_layout()
plt.savefig("seq_len_dynamic_memory_crosspoint.png", dpi=150, bbox_inches="tight")
print("Dynamic memory cross point plot saved to: seq_len_dynamic_memory_crosspoint.png")
