# LLM Systems Tutorial

> A long-term, evolving collection of tutorials documenting my journey into the systems-level engineering behind large language models — from foundational theory to practical optimization.

This repository serves as a personal learning log and open reference. Each tutorial is developed incrementally: some are fully fleshed out with slides, code, and notes, while others are placeholders for topics I plan to explore next. The goal is to build a comprehensive, self-contained curriculum over time.

## Roadmap

The table below tracks the current state of each tutorial. Tutorials are added as I study new topics — checked items have substantive content, unchecked items are planned.

| Status | Topic | Slides | Code | Description |
|:------:|-------|--------|------|-------------|
| :black_square_button: | **Transformer Architecture** | — | — | Core architecture walkthrough: self-attention, feed-forward networks, residual connections, layer normalization. |
| :black_square_button: | **Position Encoding** | — | — | Positional encoding schemes: sinusoidal, RoPE, ALiBi, and beyond. |
| :black_square_button: | **Parameters** | — | — | Parameter counting, weight shapes, and scaling laws for Transformer models. |
| :black_square_button: | **FLOPS** | — | — | Floating-point operation analysis for training and inference workloads. |
| :black_square_button: | **Mixture of Experts (MoE)** | — | — | Sparse MoE architectures, routing strategies, load balancing, and systems implications. |
| :white_check_mark: | **LLM Memory Analysis** | [PDF](tutorials/llm-memory/slides/main.pdf) | [code](code/llm-memory/) | Training & inference memory breakdown — weights, activations, optimizer states, KV cache, and optimization strategies. |

## Repository Structure

```
.
├── code/                              # Runnable experiment scripts
│   └── llm-memory/
│       ├── profile_memory.py          # GPU memory profiling with PyTorch profiler
│       ├── find_max_batch.py          # Binary search for max training batch size
│       ├── find_max_seq_len.py        # Binary search for max inference sequence length
│       └── plot_seq_len_memory.py     # Plot sequence length vs. peak GPU memory
│
└── tutorials/                         # Lecture materials and notes
    ├── llm-memory/                    # ✅ Complete
    │   ├── slides/                    #    LaTeX source, figures, and compiled PDF
    │   ├── notes.md                   #    Supplementary notes
    │   └── README.md                  #    Tutorial-specific guide
    ├── transformer/                   # 🚧 Planned
    ├── parameters/                    # 🚧 Planned
    ├── FLOPS/                         # 🚧 Planned
    ├── Position Encoding/             # 🚧 Planned
    └── MoE/                           # 🚧 Planned
```

## Contributing

This is primarily a personal learning project, but suggestions, corrections, and discussions are welcome — feel free to open an issue.

## License

[MIT](LICENSE)
