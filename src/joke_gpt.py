"""Joke GPT: the gpt_colab.py architecture, plus a document mask and per-joke positions.

Changes from gpt_colab.py:
- attention mask comes from the batch: causal AND same joke ([B, T, T])
- position ids come from the batch: restart at 0 at every <|joke|>
- --no-doc-mask: plain causal mask + positions 0..T-1 (GPT-2 style baseline)
- generation starts from <|joke|> and stops at the next <|joke|>

Usage:
    python -m src.joke_gpt --smoke
    python -m src.joke_gpt --out data/run_masked
    python -m src.joke_gpt --no-doc-mask --out data/run_unmasked
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src import bpe
from src import joke_dataset as jd


@dataclass
class Config:
    vocab_size: int = jd.VOCAB_SIZE
    block_size: int = 256
    n_embedd: int = 192
    num_heads: int = 4
    n_layer: int = 4
    dropout: float = 0.1


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embedd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embedd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embedd, head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        k, q, v = self.key(x), self.query(x), self.value(x)  # (B, T, head_size)
        scores = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5  # (B, T, T)
        # mask: (B, T, T) bool, True = may attend. The diagonal is always True, so no row is all -inf.
        scores = scores.masked_fill(~mask, float("-inf"))
        wei = self.dropout(F.softmax(scores, dim=-1))
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embedd // cfg.num_heads
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.num_heads)])
        self.proj = nn.Linear(cfg.n_embedd, cfg.n_embedd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embedd, 4 * cfg.n_embedd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embedd, cfg.n_embedd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embedd)
        self.sa_heads = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embedd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x, mask):
        x = x + self.sa_heads(self.ln1(x), mask)
        return x + self.ffwd(self.ln2(x))


class JokeGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embedd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embedd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embedd)
        self.lm_head = nn.Linear(cfg.n_embedd, cfg.vocab_size)

    def forward(self, idx, pos, mask, targets=None):
        """idx, pos: (B, T) long; mask: (B, T, T) bool; targets: (B, T) or None."""
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, max_new_tokens=255, temperature=1.0, top_k=None, generator=None, prompt_ids=()):
        """Sample one joke: start at <|joke|> (+ optional prompt), stop at the next <|joke|>.

        Returns ids without SEP; the prompt ids are included at the start.
        temperature < 1 sharpens the distribution, > 1 flattens it.
        top_k keeps only the k most likely tokens before sampling.
        """
        device = self.lm_head.weight.device
        idx = torch.tensor([[jd.SEP, *prompt_ids]], device=device)
        for _ in range(min(max_new_tokens, self.cfg.block_size - idx.shape[1])):
            T = idx.shape[1]
            pos = torch.arange(T, device=device)[None]
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))[None]
            logits, _ = self(idx, pos, mask)
            last = logits[0, -1] / temperature
            if top_k is not None:
                kth = torch.topk(last, top_k).values[-1]
                last = last.masked_fill(last < kth, float("-inf"))
            probs = F.softmax(last, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(device)
            if nxt.item() == jd.SEP:
                break
            idx = torch.cat([idx, nxt[None]], dim=1)
        return idx[0, 1:].tolist()


def batch_inputs(x, y, pos, doc, doc_mask, device):
    """NumPy batch -> tensors. Without doc_mask: plain causal mask and positions 0..T-1.

    The [B, T, T] mask is built on the device (same rule as jd.document_mask) so only
    [B, T] arrays cross from CPU; on a fast GPU the copy would otherwise dominate.
    """
    to = lambda a: torch.from_numpy(np.array(a)).to(device)
    x, y, pos, doc = to(x), to(y), to(pos), to(doc)
    T = x.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    if doc_mask:
        mask = causal & (doc[:, :, None] == doc[:, None, :])
    else:
        mask = causal.expand(len(x), T, T)
        pos = torch.arange(T, device=x.device).expand_as(x)
    return x, pos, mask, y


def split_jokes(jokes, test_frac=0.1, seed=0):
    """Split by whole jokes (never mid-joke), random order so time-of-posting doesn't bias test."""
    order = np.random.default_rng(seed).permutation(len(jokes))
    n_test = int(test_frac * len(jokes))
    return [jokes[i] for i in order[n_test:]], [jokes[i] for i in order[:n_test]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-doc-mask", action="store_true")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-data", default="data/jokes_clean.jsonl",
                   help="training jokes; the test set always comes from data/jokes_clean.jsonl")
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=0, help="linear LR warmup steps before cosine decay")
    p.add_argument("--out", default="data/run")
    args = p.parse_args(argv)

    cfg = Config(n_embedd=args.n_embd, num_heads=args.heads, n_layer=args.layers, dropout=args.dropout)
    if args.smoke:
        cfg = Config(n_embedd=32, num_heads=2, n_layer=1)
        args.iters, args.eval_every, args.eval_iters, args.batch = 20, 10, 2, 4
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp = device == "cuda"  # fp16 mixed precision: T4 tensor cores
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    doc_mask = not args.no_doc_mask

    merges = bpe.load("data/bpe_1024.json")
    with open("data/jokes_clean.jsonl") as f:
        jokes = [json.loads(line) for line in f]
    train_jokes, test_jokes = split_jokes(jokes)
    if args.train_data != "data/jokes_clean.jsonl":
        # Same test jokes across runs so test losses are comparable; keep them out of training.
        held_out = set(test_jokes)
        with open(args.train_data) as f:
            train_jokes = [j for j in map(json.loads, f) if j not in held_out]
    streams = {}
    for name, js in (("train", train_jokes), ("test", test_jokes)):
        streams[name], dropped = jd.build_stream(js, merges, cfg.block_size)
        print(f"{name}: {len(js)} jokes, {len(streams[name][0])} tokens, dropped {dropped}")

    model = JokeGPT(cfg).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"device={device} amp={use_amp} doc_mask={doc_mask} params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    decay_steps = max(1, args.iters - args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / args.warmup if s < args.warmup
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / decay_steps)))
    ))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def batch(split):
        return batch_inputs(*jd.get_batch(streams[split], cfg.block_size, args.batch, rng), doc_mask, device)

    def loss_on(split):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(*batch(split))[1]

    @torch.no_grad()
    def estimate():
        model.eval()
        out = {}
        for split in ("train", "test"):
            out[split] = sum(loss_on(split).item() for _ in range(args.eval_iters)) / args.eval_iters
        model.train()
        return out

    history, best, start = [], float("inf"), time.time()
    for step in range(args.iters + 1):
        if step % args.eval_every == 0:
            losses = estimate()
            history.append({"step": step, **losses})
            print(f"step {step}: train {losses['train']:.4f} test {losses['test']:.4f} {time.time() - start:.0f}s")
            if losses["test"] < best:
                best = losses["test"]
                torch.save({"cfg": asdict(cfg), "model": model.state_dict()}, f"{args.out}.pt")
        if step == args.iters:
            break
        loss = loss_on("train")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

    summary = {"args": vars(args), "cfg": asdict(cfg), "params": n_params, "history": history,
               "best_test": best, "seconds": round(time.time() - start)}
    with open(f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=1)
    model.load_state_dict(torch.load(f"{args.out}.pt", map_location=device)["model"])
    model.eval()
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        print("---\n" + bpe.decode(model.generate(generator=g), merges))
    return summary


if __name__ == "__main__":
    main()
