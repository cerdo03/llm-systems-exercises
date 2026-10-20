"""Configurable Tiny Shakespeare GPT for GPU experiments (Kaggle / Colab).

Same architecture as GPTLanguageModel / GQALanguageModel in tiny_shakespeare.ipynb:
pre-norm blocks, residuals, grouped-query attention with output projection,
FFN (GELU or ReLU), dropout, char or BPE tokens.

Every run appends one row to results.csv, so runs from different sessions can be compared.

    python gpt_experiments.py --name M-base --n_embd 384 --n_head 6 --n_layer 6
    python gpt_experiments.py --show            # print results.csv as a table
    python gpt_experiments.py --smoke           # tiny CPU check, a few seconds
"""

import argparse
import collections
import csv
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

p = argparse.ArgumentParser()
p.add_argument("--name", default="run")
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--kv_groups", type=int, default=0, help="0 = n_head (MHA); 1 = MQA")
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max_iters", type=int, default=5000)
p.add_argument("--eval_interval", type=int, default=500)
p.add_argument("--eval_iters", type=int, default=50, help="batches per split at intermediate evals")
p.add_argument("--final_eval_iters", type=int, default=200)
p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
p.add_argument("--bpe_vocab", type=int, default=1000)
p.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--results", default="results.csv")
p.add_argument("--show", action="store_true", help="print results table and exit")
p.add_argument("--smoke", action="store_true", help="tiny config for a quick CPU check")
args = p.parse_args()


def show_results(path):
    rows = []
    if Path(path).exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        print("no results yet")
        return
    cols = ["name", "params", "tokenizer", "kv_groups", "block_size", "dropout",
            "train_loss", "test_loss", "test_bpc", "gap", "kv_floats_per_token", "minutes"]
    widths = {c: max(len(c), *(len(r.get(c, "")) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(r.get(c, "").ljust(widths[c]) for c in cols))


if args.show:
    show_results(args.results)
    raise SystemExit

if args.smoke:
    args.n_embd, args.n_head, args.n_layer, args.block_size, args.batch_size = 32, 4, 2, 16, 4
    args.max_iters, args.eval_interval, args.eval_iters, args.final_eval_iters = 20, 10, 2, 2
    args.bpe_vocab = min(args.bpe_vocab, 120)
kv_groups = args.kv_groups or args.n_head
assert args.n_embd % args.n_head == 0 and args.n_head % kv_groups == 0

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
torch.manual_seed(args.seed)
print(f"[{args.name}] device={device} "
      f"gpu={torch.cuda.get_device_name(0) if use_amp else '-'}")

# ---------------- data ----------------
text_path = Path("input.txt")
if not text_path.exists():
    urllib.request.urlretrieve(DATA_URL, text_path)
text = text_path.read_text(encoding="utf-8")
split_at = int(0.9 * len(text))
train_text, test_text = text[:split_at], text[split_at:]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}

if args.tokenizer == "char":
    vocab = {i: c for i, c in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]
else:
    # BPE trained on the training split only; cached so later runs reuse it.
    CHUNK = re.compile(r" ?[A-Za-z']+| ?[^A-Za-z'\s]|\s")
    cache = Path(f"bpe_{args.bpe_vocab}.json")
    if cache.exists():
        merge_list = json.loads(cache.read_text())
    else:
        t0 = time.time()
        counts = collections.Counter(CHUNK.findall(train_text))
        pieces = {w: [stoi[c] for c in w] for w in counts}
        merge_list, next_id = [], len(chars)
        while next_id < args.bpe_vocab:
            pairs = collections.Counter()
            for w, ids in pieces.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += counts[w]
            (a, b), _ = pairs.most_common(1)[0]
            merge_list.append([a, b, next_id])
            for w, ids in pieces.items():
                out, i = [], 0
                while i < len(ids):
                    if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                pieces[w] = out
            next_id += 1
        cache.write_text(json.dumps(merge_list))
        print(f"trained BPE-{args.bpe_vocab} in {time.time() - t0:.0f}s")
    merges = {(a, b): n for a, b, n in merge_list}
    vocab = {i: c for i, c in enumerate(chars)}
    for a, b, n in merge_list:
        vocab[n] = vocab[a] + vocab[b]
    chunk_cache = {}

    def encode_chunk(chunk):
        if chunk not in chunk_cache:
            ids = [stoi[c] for c in chunk]
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:]), key=lambda pr: merges.get(pr, math.inf))
                if pair not in merges:
                    break
                out, i = [], 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                        out.append(merges[pair])
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                ids = out
            chunk_cache[chunk] = ids
        return chunk_cache[chunk]

    def encode(s):
        return [t for chunk in CHUNK.findall(s) for t in encode_chunk(chunk)]


def decode(tokens):
    return "".join(vocab[t] for t in tokens)


train_data = torch.tensor(encode(train_text), dtype=torch.long)
test_data = torch.tensor(encode(test_text), dtype=torch.long)
assert decode(test_data.tolist()) == test_text
vocab_size = len(vocab)
chars_per_token = len(test_text) / len(test_data)
print(f"tokenizer={args.tokenizer} vocab={vocab_size} chars/token={chars_per_token:.2f}")


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i + args.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------- model ----------------
class GroupedQueryAttention(nn.Module):
    # kv_groups == n_head -> multi-head; kv_groups == 1 -> multi-query.
    def __init__(self):
        super().__init__()
        self.hs = args.n_embd // args.n_head
        self.query = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.key = nn.Linear(args.n_embd, kv_groups * self.hs, bias=False)
        self.value = nn.Linear(args.n_embd, kv_groups * self.hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(args.block_size, args.block_size)))
        self.attn_dropout = nn.Dropout(args.dropout)
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, args.n_head, self.hs).transpose(1, 2)  # (B, H, T, hs)
        k = self.key(x).view(B, T, kv_groups, self.hs).transpose(1, 2)  # (B, G, T, hs)
        v = self.value(x).view(B, T, kv_groups, self.hs).transpose(1, 2)
        k = k.repeat_interleave(args.n_head // kv_groups, dim=1)  # (B, H, T, hs)
        v = v.repeat_interleave(args.n_head // kv_groups, dim=1)
        scores = q @ k.transpose(-2, -1) / self.hs**0.5  # (B, H, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(q.dtype))
        out = (wei @ v).transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.GELU() if args.activation == "gelu" else nn.ReLU()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.sa_heads = GroupedQueryAttention()
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd), act,
            nn.Linear(4 * args.n_embd, args.n_embd), nn.Dropout(args.dropout),
        )

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(args.n_layer)], nn.LayerNorm(args.n_embd)
        )
        self.lm_head = nn.Linear(args.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        logits = self.lm_head(self.blocks(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1).float(), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -args.block_size:])
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


m = GPTLanguageModel().to(device)
n_params = sum(p.numel() for p in m.parameters())
kv_floats = 2 * args.n_layer * kv_groups * (args.n_embd // args.n_head)
print(f"params={n_params:,} heads={args.n_head} kv_groups={kv_groups} "
      f"KV-cache floats/token={kv_floats}")
optimizer = torch.optim.AdamW(m.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def estimate_loss(iters):
    out = {}
    m.eval()
    for split in ("train", "test"):
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


def to_bpc(loss):
    # nats per token -> bits per character
    return loss / chars_per_token / math.log(2)


start = time.time()
for step in range(args.max_iters):
    if step % args.eval_interval == 0 and step > 0:
        ls = estimate_loss(args.eval_iters)
        print(f"step {step}: train {ls['train']:.4f} test {ls['test']:.4f} "
              f"(test bpc {to_bpc(ls['test']):.3f}) {(time.time() - start) / 60:.1f} min")
    xb, yb = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if step == 49:
        per_iter = (time.time() - start) / 50
        print(f"~{per_iter * 1000:.0f} ms/iter -> about "
              f"{per_iter * args.max_iters / 60:.0f} min for {args.max_iters} steps (+ evals)")

ls = estimate_loss(args.final_eval_iters)
minutes = (time.time() - start) / 60
print(f"FINAL [{args.name}] train {ls['train']:.4f} test {ls['test']:.4f} "
      f"test bpc {to_bpc(ls['test']):.4f} gap {ls['test'] - ls['train']:.4f} {minutes:.1f} min")

m.eval()
sample = decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 600)[0].tolist())
Path(f"sample_{args.name}.txt").write_text(sample, encoding="utf-8")
print("\n" + sample[:400])

row = {
    "name": args.name, "params": n_params, "tokenizer": f"{args.tokenizer}"
    + (f"-{args.bpe_vocab}" if args.tokenizer == "bpe" else ""),
    "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
    "kv_groups": kv_groups, "block_size": args.block_size, "batch_size": args.batch_size,
    "dropout": args.dropout, "lr": args.lr, "max_iters": args.max_iters,
    "activation": args.activation, "seed": args.seed,
    "train_loss": f"{ls['train']:.4f}", "test_loss": f"{ls['test']:.4f}",
    "test_bpc": f"{to_bpc(ls['test']):.4f}", "gap": f"{ls['test'] - ls['train']:.4f}",
    "kv_floats_per_token": kv_floats, "minutes": f"{minutes:.1f}",
    "gpu": torch.cuda.get_device_name(0) if use_amp else "cpu",
}
new_file = not Path(args.results).exists()
with open(args.results, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new_file:
        w.writeheader()
    w.writerow(row)
print(f"\nappended to {args.results}")
