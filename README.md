# llm

A decoder-only transformer language model in PyTorch, with the full pipeline
around it: tokenizer training, dataset preparation, training, and sampling.

| Stage | Module | What it does |
| --- | --- | --- |
| Tokenization | `llm/tokenizer.py` | Byte-level BPE, trained from your corpus. Lossless for any UTF-8 text. |
| Data | `llm/prepare.py`, `llm/data.py` | Encodes text to memory-mapped token files and samples random windows. |
| Embedding | `llm/model.py` | Token + learned position embeddings; output head tied to the token embedding. |
| Attention | `llm/model.py` | Multi-head causal self-attention, with a key/value cache for generation. |
| Feed-forward | `llm/model.py` | 4x-wide GELU MLP in pre-norm residual blocks. |
| Training loop | `llm/train.py` | AdamW, warmup + cosine schedule, gradient accumulation and clipping, mixed precision, periodic eval, checkpoint and resume. |
| Sampling | `llm/generate.py` | Temperature, top-k and top-p sampling. |

## Setup

Requires Python 3.10+.

```sh
pip install -e ".[dev]"
```

For GPU training, install a CUDA build of PyTorch first (see pytorch.org).

## Quick start

Train a small model on Tiny Shakespeare (about 15-20 minutes on a laptop CPU):

```sh
curl -L -o data/shakespeare.txt --create-dirs https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

llm-prepare --input data/shakespeare.txt --out-dir data/tiny --vocab-size 2048
llm-train --config configs/tiny.yaml
llm-generate --run runs/tiny --prompt "ROMEO:" --max-new-tokens 200
```

Each command is also available as a module, e.g. `python -m llm.train`.

## Preparing data

```sh
llm-prepare --input a.txt b.txt --out-dir data/mine --vocab-size 8192
```

This trains the tokenizer on the input files and writes `train.bin`, `val.bin`,
`tokenizer.json` and `meta.json`. Each input file is treated as one document and
is followed by an `<|endoftext|>` token. The last `--val-fraction` (default 0.1)
of the token stream is held out for validation. Pass `--tokenizer path.json` to
reuse an existing tokenizer instead of training a new one.

Tokenizer training is pure Python: fine for corpora up to a few hundred MB,
slow beyond that. The encoded tokens are held in memory before being written.

## Training

```sh
llm-train --config configs/tiny.yaml
llm-train --config configs/tiny.yaml --set train.max_steps=5000 --set model.n_layer=6
llm-train --config configs/tiny.yaml --resume
```

All options and their defaults are in `llm/config.py`. Unknown keys are rejected.
`model.vocab_size` is read from the dataset unless you set it.

The run directory (`train.out_dir`) receives:

- `last.pt` - most recent checkpoint, written at every evaluation and on Ctrl+C
- `best.pt` - checkpoint with the lowest validation loss
- `metrics.jsonl` - one JSON line per log and eval step
- `config.yaml` - the resolved config
- `tokenizer.json` - copied from the dataset, so the directory is self-contained

`--resume` continues from `last.pt`. The model section of the config must match
the checkpoint; training options (for example a larger `max_steps`) may change.

Mixed precision is used automatically on CUDA (bfloat16 where supported,
otherwise float16 with gradient scaling). `configs/base.yaml` is a
GPT-2-small-sized configuration for a single GPU.

## Using the model from Python

```python
import torch
from llm.checkpoint import load_model
from llm.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load("runs/tiny/tokenizer.json")
model, config = load_model("runs/tiny/best.pt", device="cpu")

idx = torch.tensor([tokenizer.encode("ROMEO:")])
out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_p=0.95, eos_id=tokenizer.eot_id)
print(tokenizer.decode(out[0].tolist(), skip_special=True))
```

`tokenizer.encode` treats special tokens in its input as plain text unless
`allow_special=True`, so user-supplied text cannot inject `<|endoftext|>`.

## Tests

```sh
pytest
```

## Not included

- Multi-GPU / distributed training
- Padding and attention masks for batched variable-length inputs (training uses
  fixed-length windows; batched generation needs equal-length prompts)
- An inference server
