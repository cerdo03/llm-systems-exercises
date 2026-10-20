"""Full fine-tune of pretrained SmolLM2-135M (base) on the cleaned jokes. Own training loop, no Trainer.

Format per joke: <|endoftext|> joke <|endoftext|>   (EOS plays the role of our <|joke|>)
Batches pad each joke separately (attention_mask 0, labels -100 on padding), so attention
never crosses jokes. Jokes are grouped by length so little compute goes to padding.
Same data as run C: score>=10 training jokes, fixed score>=50 test jokes.
Loss is reported as bits per character (bpc) so it is comparable across tokenizers.

Usage: python -m src.smol_finetune [--smoke] [--epochs 1] [--out data/smol_jokes]
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from src import joke_gpt as jg

MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_TOKENS = 256


def encode_jokes(jokes, tok):
    """Return [(ids, n_chars)] with EOS on both sides; drop jokes longer than MAX_TOKENS."""
    out = []
    for joke in jokes:
        ids = [tok.eos_token_id, *tok(joke).input_ids, tok.eos_token_id]
        if len(ids) <= MAX_TOKENS:
            out.append((ids, len(joke)))
    return out


def length_batches(items, batch_size, rng):
    """Shuffle, sort within chunks of 50 batches by length, then shuffle the batches."""
    order = rng.permutation(len(items))
    chunk = 50 * batch_size
    batches = []
    for s in range(0, len(order), chunk):
        part = sorted(order[s:s + chunk], key=lambda i: len(items[i][0]))
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def collate(seqs, pad_id):
    """Right-pad to the longest sequence. Labels -100 on padding so it adds no loss."""
    T = max(map(len, seqs))
    input_ids = torch.full((len(seqs), T), pad_id)
    attention = torch.zeros((len(seqs), T), dtype=torch.long)
    labels = torch.full((len(seqs), T), -100)
    for r, s in enumerate(seqs):
        input_ids[r, :len(s)] = torch.tensor(s)
        attention[r, :len(s)] = 1
        labels[r, :len(s)] = torch.tensor(s)
    return input_ids, attention, labels


@torch.no_grad()
def bits_per_char(model, items, pad_id, device, batch_size=32):
    """Total next-token loss (nats) over all jokes / ln 2 / total joke characters."""
    model.eval()
    nats, chars = 0.0, 0
    for b in range(0, len(items), batch_size):
        part = items[b:b + batch_size]
        ids, att, lab = (t.to(device) for t in collate([s for s, _ in part], pad_id))
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        nats += torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), lab[:, 1:].flatten(), ignore_index=-100, reduction="sum").item()
        chars += sum(n for _, n in part)
    model.train()
    return nats / math.log(2) / chars


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", default="data/smol_jokes")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp = device == "cuda"  # fp16 autocast on T4; weights stay fp32
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

    with open("data/jokes_clean.jsonl") as f:
        _, test_jokes = jg.split_jokes([json.loads(line) for line in f])
    held_out = set(test_jokes)
    with open("data/jokes_clean_s10.jsonl") as f:
        train_jokes = [j for j in map(json.loads, f) if j not in held_out]
    train, test = encode_jokes(train_jokes, tok), encode_jokes(test_jokes, tok)
    if args.smoke:
        train, test, args.epochs, args.eval_every = train[:64], test[:32], 1, 2
    print(f"train {len(train)} jokes / {sum(len(s) for s, _ in train):,} tokens, test {len(test)} jokes")

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = int(args.epochs * steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / args.warmup if s < args.warmup
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, total - args.warmup))))
    ))

    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"device={device} amp={use_amp} steps={total}")
    history = [{"step": 0, "test_bpc": bits_per_char(model, test, tok.eos_token_id, device)}]
    print(f"step 0 (pretrained, no fine-tune): test bpc {history[0]['test_bpc']:.4f}", flush=True)
    best, step, start = history[0]["test_bpc"], 0, time.time()
    while step < total:
        for batch in length_batches(train, args.batch, rng):
            ids, att, lab = (t.to(device) for t in collate([train[i][0] for i in batch], tok.eos_token_id))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # clip real gradients, not fp16-scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % args.eval_every == 0 or step == total:
                bpc = bits_per_char(model, test, tok.eos_token_id, device)
                history.append({"step": step, "train_loss": loss.item(), "test_bpc": bpc})
                print(f"step {step}/{total}: train loss {loss.item():.3f} test bpc {bpc:.4f} "
                      f"{time.time() - start:.0f}s", flush=True)
                if bpc < best:
                    best = bpc
                    model.save_pretrained(args.out)
                    tok.save_pretrained(args.out)
            if step == total:
                break
    with open(f"{args.out}.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "best_test_bpc": best}, f, indent=1)
    print(f"best test bpc {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
