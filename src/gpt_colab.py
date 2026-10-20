"""Full-size Tiny Shakespeare GPT, same architecture as tiny_shakespeare.ipynb.

Colab: Runtime -> Change runtime type -> T4 GPU, upload this file, then run
    !python gpt_colab.py
Local smoke test (tiny config, CPU, ~seconds):
    python3 src/gpt_colab.py --smoke
"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="tiny config for a quick check")
args = parser.parse_args()

# nanoGPT "Let's build GPT" sizes; --smoke shrinks everything.
if args.smoke:
    batch_size, block_size, max_iters, eval_interval, eval_iters = 4, 16, 20, 10, 2
    n_embedd, num_heads, n_layer, dropout, learning_rate = 32, 4, 2, 0.2, 1e-3
else:
    batch_size, block_size, max_iters, eval_interval, eval_iters = 64, 256, 5000, 500, 200
    n_embedd, num_heads, n_layer, dropout, learning_rate = 384, 6, 6, 0.2, 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"  # fp16 mixed precision: ~2x faster on a T4
torch.manual_seed(1337)
print(f"device={device}, amp={use_amp}")

# ---- data: same character tokenizer and 90/10 split as the notebook ----
text_path = Path(__file__).with_name("input.txt")
if not text_path.exists():
    urllib.request.urlretrieve(DATA_URL, text_path)
text = text_path.read_text(encoding="utf-8")

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}


def encode(string):
    return [stoi[char] for char in string]


def decode(tokens):
    return "".join(itos[token] for token in tokens)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, test_data = data[:n], data[n:]


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---- model: identical structure to GPTLanguageModel in the notebook, GELU in FFN ----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedd, head_size, bias=False)
        self.query = nn.Linear(n_embedd, head_size, bias=False)
        self.value = nn.Linear(n_embedd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)  # (B, T, head_size)
        d_k = k.shape[-1]
        scores = q @ k.transpose(-2, -1) / d_k**0.5  # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(scores, dim=-1))
        return wei @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embedd // num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedd, n_embedd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embedd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedd, 4 * n_embedd),
            nn.GELU(),
            nn.Linear(4 * n_embedd, n_embedd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embedd)
        self.sa_heads = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(n_embedd)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd)
        self.position_embedding_table = nn.Embedding(block_size, n_embedd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)], nn.LayerNorm(n_embedd))
        self.lm_head = nn.Linear(n_embedd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        logits = self.lm_head(self.blocks(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            C = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(B * T, C).float(), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


m = GPTLanguageModel().to(device)
print(f"Parameters: {sum(p.numel() for p in m.parameters()):,}")
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ("train", "test"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


start = time.time()
for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train {losses['train']:.4f}, test {losses['test']:.4f}, "
            f"{time.time() - start:.0f}s"
        )
    if step == max_iters:
        break
    xb, yb = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

m.eval()
sample = decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 1000)[0].tolist())
print("\n" + sample)
Path("sample.txt").write_text(sample, encoding="utf-8")
torch.save(m.state_dict(), "gpt_tiny_shakespeare.pt")
print("\nSaved sample.txt and gpt_tiny_shakespeare.pt")
