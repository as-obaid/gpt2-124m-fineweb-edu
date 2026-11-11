---
language:
- en
license: odc-by
library_name: transformers
pipeline_tag: text-generation
tags:
- gpt2
- llm.c
datasets:
- HuggingFaceFW/fineweb-edu
model-index:
- name: gpt2-124m-fineweb-edu
  results:
  - task:
      type: text-generation
    dataset:
      name: HellaSwag
      type: Rowan/hellaswag
      split: validation
    metrics:
    - name: acc_norm
      type: acc_norm
      value: 32.49
---

# GPT-2 124M, 3 epochs of FineWeb-Edu

GPT-2 124M trained from scratch on three epochs of [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) with Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). The scripts, logs and full write-up are on GitHub: [as-obaid/gpt2-124m-fineweb-edu](https://github.com/as-obaid/gpt2-124m-fineweb-edu).

| Model | Val loss ↓ | HellaSwag ↑ |
|:--|:-:|:-:|
| OpenAI GPT-2 124M | – | 29.55% |
| **This model** | 2.969 | 32.49% |

Val loss is the mean cross-entropy on the first 10.5M tokens of the held-out val shard (the first 100M tokens of sample-10BT, never trained on). HellaSwag is acc_norm on all 10,042 validation examples, scored by llm.c.

![Train and validation loss, and HellaSwag accuracy, over the 56,352 steps](training_curves.png)

## Use

```python
from transformers import pipeline

generate = pipeline("text-generation", model="as-obaid/gpt2-124m-fineweb-edu")
out = generate("During photosynthesis in green plants", max_new_tokens=60, do_sample=True, top_k=50)
print(out[0]["generated_text"])
```

It loads as a standard `GPT2LMHeadModel` with OpenAI's GPT-2 tokenizer. The weights are bfloat16, as llm.c saves them.

## Training

| | |
|:--|:--|
| **Architecture** | GPT-2 124M: 12 layers, 12 heads, 768 dim, 1,024 context |
| **Data** | FineWeb-Edu sample-10BT: 9.85B training tokens, reshuffled every epoch |
| **Schedule** | 3 epochs, 56,352 steps of 524,288 tokens |
| **Optimizer** | AdamW, LR 6e-4 with 715 warmup steps, cosine decay to 6e-5, weight decay 0.1 |
| **Precision** | bf16, cuDNN flash attention |
| **Compute** | One GPU at a time: an NVIDIA H200 for 48,000 steps and an H100 for 8,352, 18.2 GPU-hours in all |

## Files

- `model.safetensors`, `config.json` and the tokenizer files: the Transformers model, converted with llm.c's [`dev/eval/export_hf.py`](https://github.com/karpathy/llm.c/blob/master/dev/eval/export_hf.py).
- `llmc/model_00056352.bin`: the original llm.c checkpoint from the last step. llm.c loads it with `-e llmc/model_00056352.bin`.

## Limitations

This is a small base model, with no instruction tuning or safety training. It writes fluent English that is often wrong, so don't rely on it for facts. It learned from English educational web pages and carries their biases.

## License and credits

ODC-By 1.0, the license of FineWeb-Edu. Training code: [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy (MIT). Data: FineWeb-Edu, Penedo et al., 2024. Architecture: GPT-2, Radford et al., OpenAI, 2019.

Muhammad Obaidullah
