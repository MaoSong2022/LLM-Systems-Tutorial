# LLM Systems Tutorial

A structured, ground-up tutorial series on the systems engineering behind large language models — spanning architecture fundamentals, cost modeling, hardware analysis, distributed parallelism, and production-grade training & inference systems.

## Curriculum

| Module | Topic | Subtopics |
|:------:|-------|-----------|
| 0 | **Foundations** | Tokenizer · Transformer Architecture |
| 1 | **Cost Modeling** | Parameter Count · FLOPs · Memory · Numerical Precision |
| 2 | **Core Bottleneck: Attention** | Attention Overview · Flash Attention |
| 3 | **Hardware & Roofline** | Roofline Model · GPU / TPU Architecture |
| 4 | **Parallelism & Distribution** | Matrix Partitioning · Communication Primitives |
| 5 | **Parallelism Strategies** | DP · ZeRO · PP · TP · EP · SP · MoE |
| 6 | **Training Systems** | Megatron-LM · FSDP · DeepSpeed |
| 7 | **Inference Systems** | nano-vLLM · mini-SGLang |
| 8 | **Advanced Topics** | MoE · KV Cache · Custom Kernels |

## Progress

| Status | Tutorial | Materials |
|:------:|----------|-----------|
| :white_check_mark: | LLM Memory Analysis | [Slides (PDF)](tutorials/llm-memory/slides/main.pdf) · [Code](code/llm-memory/) · [Notes](tutorials/llm-memory/notes.md) |

> Tutorials are developed incrementally. Checked items have substantive content; the rest are in progress.

## Repository Structure

```
.
├── code/                       # Runnable experiments & profiling scripts
│   └── llm-memory/
└── tutorials/                  # Lecture slides, notes, and references
    └── llm-memory/
        ├── slides/             # LaTeX source + compiled PDF
        └── notes.md            # Supplementary notes
```

## License

[MIT](LICENSE)
