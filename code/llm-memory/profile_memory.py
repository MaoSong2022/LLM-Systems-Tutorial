import torch
from torch import nn
from datetime import datetime
from torch.autograd.profiler import record_function
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def train(num_iter=5, device="cuda:0"):
    model_name = "Qwen/Qwen3-4B"
    bsz = 40
    max_length = 512

    def trace_handler(prof: torch.profiler.profile):
        # 获取时间用于文件命名
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"visual_mem_{timestamp}"

        # 导出tracing格式的profiling
        prof.export_chrome_trace(f"{model_name.replace('/', '_')}_{bsz}_{max_length}_{file_name}.json")

        # 导出mem消耗可视化数据
        prof.export_memory_timeline(
            f"{model_name.replace('/', '_')}_{bsz}_{max_length}_{file_name}.html", device="cuda:0"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

    dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")
    sample = dataset[:bsz]

    inputs = tokenizer(
        sample["horoscope"],
        truncation=True,
        max_length=max_length,
        padding=True,  # 补齐到 batch 内最长序列
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)  # [1, seq_len]
    attention_mask = inputs["attention_mask"].to(device)  # [1, seq_len]
    x = input_ids[:, :-1]  # [1, seq_len-1]
    labels = input_ids[:, 1:]  # [1, seq_len-1]
    mask = attention_mask[:, :-1]  # [1, seq_len-1]

    criterion = torch.nn.CrossEntropyLoss()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iter):
            prof.step()
            with record_function("## forward ##"):
                outputs = model(input_ids=x, attention_mask=mask)
                logits = outputs.logits

            with record_function("## backward ##"):
                loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
                loss.backward()
                print(loss.item())

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    print("Training completed!")


if __name__ == "__main__":
    # warm-up:
    train(1)
    # run:
    train(5)
