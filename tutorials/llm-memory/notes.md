# LLM 显存占用分析-训练与推理过程中的内存需求与优化

## Motivation

我们从一个简单的问题开始

> 假如我有一张 80GB 的显卡，我想训练/推理一个 4B 的模型，我应该设置多大的 batch size 和 sequence length?

在这个 tutorial 中，我们将基于这个问题来进行思考和分析。我们将考虑更一般的问题形式：

**Motivation:** 在训练和推理时 LLM 所需要的内存是多少？如何进行优化内存占用？

为了回答以上问题，我们先介绍训练/推理阶段的内存计算，再针对可优化部分进行分析并介绍相应优化算法。

## Background

### Transformer Architecture

以 Qwen3 为例，现代 LLM 的架构包含多层 Transformer Block，其中具体的模块不同的模型可能有改动。下图是对应的模型架构

![Architecture of Qwen3](slides/LM_architecture.png)

### Notation

与参数量、FLOPs 计算所用记号一致；参数量 \(P\) 的推导见 [LLM parameter analysis](https://maosong.website/p/llm-parameter-computation/)。

| 变量 | 含义 |
|:---:|:---|
| \(P\) | number of parameters |
| \(L\) | layers |
| \(V\) | vocabulary size |
| \(d\) | hidden size |
| \(d_{\text{ff}}\) | FFN hidden size |
| \(s\) | sequence length |
| \(b\) | batch size |
| \(h\) | number of attention heads |
| \(d_h\) | attention head dimension |

### Assumptions

1. 若无特别说明，使用 **BF16/FP16**，每个参数 **2** byte。
2. 不使用 dropout（与现代大模型设定一致）。
3. Attention 基于原始 multi-head attention.
4. FFN 基于 SwiGLU.

## Training Memory Analysis

### Training Memory Components

训练部分的内存占用由四部分组成：

$$ \text{training\_memory} = \text{weight} + \text{activation} + \text{optimizer} + \text{gradient} $$

- **Weights**: \(\boxed{2P}\)
- **Gradients**（与权重同精度）: \(\boxed{2P}\)

### Optimizer States

**AdamW** 优化器需要维护两个动量状态：

- 一阶动量 \(m_t\)：\(2P\)
- 二阶动量 \(v_t\)：\(2P\)
- 合计：\(2 \times 2P = \boxed{4P}\)

AdamW [[2]](#references) 的更新规则如下：

$$
\begin{aligned}
m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\
v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
\hat{m}_t &\leftarrow \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t \leftarrow \frac{v_t}{1 - \beta_2^t} \\
\theta_t &\leftarrow \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
\end{aligned}
$$

### Activation

激活值是前向传播过程中计算得到的中间结果，用于在反向传播时计算梯度。

我们仅针对 linear layer 进行推导：

$$
\begin{aligned}
\text{forward:} \quad & \mathbf{z}_\ell = W_\ell \mathbf{a}_{\ell-1} + b_\ell, \quad \mathbf{a}_\ell = \phi(\mathbf{z}_\ell) \\
\text{backward:} \quad & \frac{\partial \mathcal{L}}{\partial W_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \cdot \frac{\partial \mathbf{z}_\ell}{\partial W_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_\ell} \cdot \boxed{\mathbf{a}_{\ell-1}}
\end{aligned}
$$

可以看到，计算第 \(\ell\) 层关于 \(W_\ell\) 的梯度时需要其输入 \(\mathbf{a}_{\ell-1}\)，因此训练时需保存每个模块对应的输入，也就是激活值 (activation)。

#### Activation — Attention

按计算图（无优化）可得需保存的激活：

- Q/K/V 投影：共享输入 → \(2bsd\)
- \(Q^\top K\)：Q, K 均需保存 → \(2 \times 2bsd = 4bsd\)
- softmax 输入：\(2bhs^2\)
- weighted sum 输入：\(2bhs^2 + 2bsd\)
- output projection 输入：\(2bsd\)

**Attention 合计：** \(\boxed{10bsd + 4bhs^2}\)

#### Activation — FFN & LayerNorm

**FFN**（SwiGLU，assume \(d_{\text{ff}} = 4d\)）：

- 第一层输入：\(2bsd\)
- SwiGLU 输入：\(2 \times d_{\text{ff}} \times d = 8bsd\)
- 第二层输入：\(2 \times d_{\text{ff}} \times d = 8bsd\)
- 合计：\(\boxed{18bsd}\)

**LayerNorm**：保存输入 → \(\boxed{2bsd}\)

#### Activation — Output

**Output** 包含以下组成部分：

- **FinalNorm** 输入：\(2bsd\)
- **lm_head** 输入：\(2bsd\)
- **Loss** 输入：\(2bsV\)

合计：\(\boxed{4bsd + 2bsV}\)

#### Activation — Total

将上面的结果汇总在一起，得到：

$$
\begin{aligned}
\text{activation} &= L \cdot \text{transformer\_block} + \text{output} \\
&= L \cdot (\text{Pre\_Norm} + \textcolor{red}{\text{Attention}} + \text{Post\_Norm} + \text{FFN}) + \text{output} \\
&= \boxed{bs(32dL + \textcolor{red}{4hsL} + 4d + 2V)} \\
&\approx bsL(32d + \textcolor{red}{4hs}) \\
&\approx \textcolor{red}{4bs^2hL}
\end{aligned}
$$

> 注：在 Qwen3 中，\(2V / 32dL \approx 6\%\)，\(32dL / 4hsL \approx 2.5\%\)。

可以看到，未优化的情况下，\(\text{activation} \propto bs^2\)。这里 \(s^2\) 主要由 attention 部分产生，后续 Flash Attention 就针对这一点进行了优化。

### Total Training Memory

将上面的结果进行汇总：

$$
\begin{aligned}
\text{training\_memory} &= \text{weight} + \text{activation} + \text{optimizer} + \text{gradient} \\
&= 2P + bs(32dL + 4hsL + 4d + 2V) + 4P + 2P \\
&= 8P + bs(32dL + 4hsL + 4d + 2V) \quad \text{(exact)} \\
&\approx 8P + 4bs^2hL
\end{aligned}
$$

可以看到，训练阶段的内存占用分为**固定部分** (\(8P\)) 和**动态部分** (\(4bs^2hL\))，动态部分主要是 attention 的缓存。

### Experiments

我们分别针对 80GB 的显卡计算 Qwen3 系列模型的最高配置：

| Model | \(P\) | \(L\) | \(h\) | \(s\) | predicted \(b\) | actual \(b\) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-0.6B | 0.6 | 28 | 16 | 512 | 68 | 34 |
| Qwen3-1.7B | 1.7 | 28 | 16 | 512 | 42 | 28 |
| Qwen3-4B | 4 | 36 | 32 | 512 | 16 | 12 |
| Qwen3-8B | 8.1 | 36 | 32 | 512 | 4 | 2 |

其中 predicted \(b\) 基于前面的准确公式计算得到；actual \(b\) 通过实验验证得到。注意我们这里的 prediction 没有考虑任何优化手段与其他内存开销，因此与实际值有出入。

### Case Study

我们分别使用 Qwen3-4B 和 Qwen3-8B 来进行实验（\(b=1\), \(s=512\)）。参考 [PyTorch 显存可视化与 Snapshot 数据分析](https://zhuanlan.zhihu.com/p/677203832)。

## Inference Memory Analysis

### Inference Components

Inference 阶段内存占用主要与模型参数、KV cache 两部分相关：

$$
\text{Inference\_Memory} = \text{weight} + \text{activation} + \text{KV cache}
$$

- **Weights**: \(\boxed{2P}\)
- **Activations**（经验值，batch size=1）: \(\approx 0.4P\)（参见 [transformer-math](https://blog.eleuther.ai/transformer-math/)）
- **KV cache**：与序列长度相关，见后续分析。

### KV Cache Mechanism

LLM 推理中为避免重复计算历史 token 的 key/value 而使用的**空间换时间**的缓存机制。

自回归时逐 token 生成，每步 attention 形式为（\(\mathbf{q}_t\) 当前 query，\(\mathbf{k}_{:,t}\) / \(\mathbf{v}_{:,t}\) 历史 K/V）：

$$
\mathbf{q}_t = W_Q \mathbf{x}_t, \quad \mathbf{k}_{:,t} = W_K[\mathbf{x}_1, \ldots, \mathbf{x}_t], \quad \mathbf{v}_{:,t} = W_V[\mathbf{x}_1, \ldots, \mathbf{x}_t]
$$

处理下一 token \(\mathbf{x}_{t+1}\) 时只需在已有结果后追加当前步：

$$
\mathbf{k}_{:,t+1} = [\mathbf{k}_{:,t},\, W_K \mathbf{x}_{t+1}], \quad \mathbf{v}_{:,t+1} = [\mathbf{v}_{:,t},\, W_V \mathbf{x}_{t+1}]
$$

- **缓存前**：每生成一个 token 都重新计算 → 总计算量 \(\sum_{t=1}^{s} \mathcal{O}(t) = \mathcal{O}(s^2)\)
- **缓存后**：每步只算当前 token \(W_K \mathbf{x}_{t+1}\)、\(W_V \mathbf{x}_{t+1}\) → 计算量 \(\mathcal{O}(s)\)、空间占用 \(\mathcal{O}(s)\) —— **以空间换时间**

### KV Cache Memory

对于 multi-head attention，KV cache 的显存占用为：

$$
\text{Memory}(\text{KV cache}) = s \times 2 \times 2 \times L \times h \times d_h = \boxed{4sLhd_h}
$$

**因子含义：** \(s\) 序列长，第一个 \(2\) 为 K+V，第二个 \(2\) 为 BF16 的 2 bytes，\(L\) 层、\(h\) 头、\(d_h\) 头维度。

**Remark:**

- KV 占用与**模型配置**（\(L, h, d_h\)）和**序列长度** \(s\) 都有关，token 越多占用越高。
- 实际中因 *page granularity*、*padding*、*fragmentation* 往往略高于理论值。
- 长输出时 KV 占比会超过权重，成为推理瓶颈 → 见后续「KV Cache 优化」。

### Total Inference Memory

综合前面分析，推理阶段的总内存为：

$$
\boxed{\text{Inference\_Memory} \approx 2.4P + 4sLhd_h}
$$

可以看到，推理阶段也由固定部分（参数量，activation）以及动态部分（KV cache）组成。

### Dynamic vs. Static

由于 Qwen3 的 KV cache 计算为 \(4sLh_{kv}d_h\)，而不同模型只有 \(L\) 不一样，因此对于更大的模型，KV cache 显存占用超过模型权重的上下文长度更高。

## Optimization

### Overview

| 阶段 | 核心方法 | 典型技术 |
|:---|:---|:---|
| Training | 显存与效率提升 | Activation Checkpointing, Mixed Precision Training, Flash Attention, ZeRO, Pipeline/Model/Data Parallelism |
| Inference | 长序列与速度优化 | KV Cache Optimization, Paged/Radix Attention, Faster Attention, Quantization |

### Mixed Precision Training

计算量大的部分用低精度，计算量小的部分用高精度。低精度参与运算，高精度避免 Overflow/Underflow。

下表是 DeepSeek-V3 [[3]](#references) 使用的混合精度训练框架的显存分析：

| Precision | BF16 | FP32 | BF16 |
|:---|:---:|:---:|:---:|
| **AMP** | No | Yes | Yes |
| Weights | BF16 (2) | FP32 (4) | BF16 (2) |
| Master weights | - | - | FP32 (4) |
| Gradients | BF16 (2) | FP32 (4) | BF16 (2) |
| Adam m | BF16 (2) | FP32 (4) | FP32 (4) |
| Adam v | BF16 (2) | FP32 (4) | FP32 (4) |
| **Static total (bytes/param)** | **8** | **16** | **16** |

**Remark:**

1. BF16 (w/ AMP) 与 FP32 (w/ AMP) 的静态显存占用相同，但 BF16 (w/ AMP) 的动态显存占用更低。
2. 主流框架基本都使用了 BF16/FP8 (w/ AMP) 的训练方式。

### ZeRO

将 optimizer states / gradients / weights 按不同 GPU 切片存储，需要参与计算时再 all-gather 整合成完整参数。这样每张卡只需维护自己负责的一部分，大幅降低单卡显存需求 [[4]](#references)。

**ZeRO Stages:**

- **ZeRO-1**：shard optimizer states

$$
\text{Training\_Memory} = \text{weight} + \text{activation} + \frac{\text{optimizer}}{\#\text{GPUs}} + \text{gradient}
$$

- **ZeRO-2**：shard optimizer states + gradients

$$
\text{Training\_Memory} = \text{weight} + \text{activation} + \frac{\text{optimizer} + \text{gradient}}{\#\text{GPUs}}
$$

- **ZeRO-3**：shard all

$$
\text{Training\_Memory} = \text{activation} + \frac{\text{weight} + \text{optimizer} + \text{gradient}}{\#\text{GPUs}}
$$

> ZeRO-3 可极大降低单卡显存上限，但通信量也会提高。

### Model Parallelism

将模型切分到不同的 GPU 上，计算时，先 dispatch，再执行计算，最后通过 all-gather 等操作得到最终结果 [[5]](#references)。切分方式包括 PP (Pipeline Parallelism)、TP (Tensor Parallelism)、EP (Expert Parallelism) 等。

$$
\text{Training\_Memory} = \frac{\text{Memory}(\text{weight})}{\text{PP degree} \times \text{TP degree}}
$$

结合 ZeRO-1 与 Model Parallelism 时（activation 中与 TP 相关的部分按 TP degree 缩减）：

$$
\text{Memory}_{\text{train}} \approx \frac{\text{weight}}{\text{PP} \times \text{TP}} + \frac{\text{activation}}{\text{TP}} + \frac{\text{optimizer}}{\#\text{GPUs}} + \frac{\text{gradient}}{\text{PP}}
$$

### Activation Checkpointing

在反向传播时，重新计算所需的输入，来达到以时间换空间的目的 [[6]](#references)。

| | No ckpt | Selective ckpt | Full ckpt |
|:---|:---:|:---:|:---:|
| memory | 很高 | 中等 | 很低 \(\sim 2bsd\) |
| extra compute | 无 | 中等 | 很高 \(\sim 2Pbs\) |

> 实践中常结合 model parallelism 与 selective checkpointing 来实现 trade-off。

### Flash Attention

通过将 Attention 的计算进行分块，来提高内存访问效率以及降低反向传播时所需要的 activation 大小 [[7]](#references)。

**Flash Attention** 通过 tiling 与 online-softmax 降低该部分显存并提升效率（详见 [notes on Flash Attention](https://maosong.website/p/notes-on-flashattention/)）。这样 attention 部分的显存就由 \(\text{activation} \propto bs^2\) 降低到了 \(\text{activation} \propto bs\)。

**Theorem:** Flash Attention 输出 \(O = \text{softmax}(QK^T)V\)（correctness）。其时间复杂度为 \(\mathcal{O}(s^2 d)\)，空间复杂度为 \(\mathcal{O}(s)\)（memory savings）。

### KV Cache Optimization

$$
\text{Memory}(\text{KV cache}) = s \times 2 \times 2 \times L \times h \times d_h
$$

针对公式中各因子的优化方向 [[8]](#references)：

1. **\(s\)**: KV cache compression, eviction, selection
2. **\(2\) (bytes)**: KV cache quantization
3. **\(2\) (K+V)**: key-value sharing, MLA [[9]](#references)
4. **\(h \times d_h\)**: MQA [[10]](#references), GQA [[11]](#references), MLA

### Weight Quantization

使用低精度来表示高精度数值的方法，来减少内存占用/提高计算效率。

| 量化时机 | 代表性工作 |
|:---|:---|
| 训练后量化 (PTQ) | GPTQ [[12]](#references), AWQ [[13]](#references), SmoothQuant [[14]](#references), GGUF [[15]](#references) |
| 量化感知训练 (QAT) | LLM-QAT [[16]](#references), PEQA [[17]](#references) |

### Activation Offloading

将一部分参数/优化器状态/激活值等存储到 CPU 上，需要的时候再加载到 GPU 上。

| Offloading 场景 | 代表性工作 |
|:---|:---|
| 训练阶段 Offloading | ZeRO-Offload [[18]](#references) / ZeRO-Infinity, FSDP [[19]](#references) CPU Offload |
| 推理阶段 Offloading | FlexGen [[20]](#references), vLLM [[21]](#references) KV Cache Offload |
| MoE Offloading | KTransformers [[22]](#references), DeepSpeed-MoE [[23]](#references) |

## Real Systems

### Training Frameworks

| Framework | Memory Optimizations |
|:---|:---|
| Megatron-LM [[5]](#references) | TP, SP, Activation Checkpointing |
| DeepSpeed [[24]](#references) | ZeRO-1/2/3, CPU/NVMe Offload, Activation Checkpointing |
| FSDP [[19]](#references) | Full parameter sharding, Gradient sharding, CPU Offload |
| Colossal-AI [[25]](#references) | ZeRO, TP, PP, Activation Checkpointing |

### Inference Frameworks

| Framework | Key Techniques | Memory Optimizations |
|:---|:---|:---|
| vLLM [[21]](#references) | Paged Attention | KV cache paging, Continuous batching |
| SGLang [[26]](#references) | Radix Attention | KV cache reuse, Efficient scheduling |
| TensorRT-LLM [[27]](#references) | Kernel fusion | Weight quantization, KV cache optimization |

## Conclusion

### Takeaway

| Components | Training | Inference | Optimization |
|:---|:---:|:---:|:---|
| weights | \(2P\) | \(2P\) | quantization |
| optimizer states | \(4P\) | 0 | ZeRO, offloading |
| gradients | \(2P\) | 0 | ZeRO |
| activations | \(\sim 4Lhs^2b\) | \(\sim 0.4P\) | ckpt, offloading, flash attention |
| KV cache | 0 | \(4sLhd_h\) | KV optimization, attention |
| **TOTAL** | \(8P + 4Lhs^2b\) | \(2.4P + 4sLhd_h\) | |

- **训练**：瓶颈主要在激活值（随 batch size / 序列长度线性增长）。
- **推理**：瓶颈主要在 KV Cache（随序列长度增长）。

### Future Directions

1. More efficient architecture (attention, MoE).
2. Scalable training/inference framework.
3. Software-hardware co-design algorithms.

## References

1. An Yang et al., "Qwen3 Technical Report," arXiv:2505.09388, 2025.
2. Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay Regularization," arXiv:1711.05101, 2019.
3. DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, 2025.
4. Samyam Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," arXiv:1910.02054, 2020.
5. Mohammad Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053, 2020.
6. Vijay Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," arXiv:2205.05198, 2022.
7. Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135, 2022.
8. Haoyang Li et al., "A Survey on Large Language Model Acceleration based on KV Cache Management," arXiv:2412.19442, 2025.
9. DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434, 2024.
10. Noam Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv:1911.02150, 2019.
11. Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245, 2023.
12. Elias Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," arXiv:2210.17323, 2023.
13. Ji Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv:2306.00978, 2024.
14. Guangxuan Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," arXiv:2211.10438, 2024.
15. Georgi Gerganov, "ggml: Tensor library for machine learning," [GitHub](https://github.com/ggerganov/ggml), 2023.
16. Zechun Liu et al., "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models," arXiv:2305.17888, 2023.
17. Jeonghoon Kim et al., "Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization," arXiv:2305.14152, 2023.
18. Jie Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training," arXiv:2101.06840, 2021.
19. Yanli Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel," arXiv:2304.11277, 2023.
20. Ying Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," arXiv:2303.06865, 2023.
21. Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP, 2023.
22. Hongtao Chen et al., "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models," SOSP, 2025.
23. Samyam Rajbhandari et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale," arXiv:2201.05596, 2022.
24. Jeff Rasley et al., "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters," KDD, 2020.
25. Shenggui Li et al., "Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training," ICPP, 2023.
26. Lianmin Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS, 2024.
27. NVIDIA Corporation, "TensorRT-LLM: A TensorRT Toolset for Optimizing LLM Inference," [GitHub](https://github.com/NVIDIA/TensorRT-LLM), 2023.
