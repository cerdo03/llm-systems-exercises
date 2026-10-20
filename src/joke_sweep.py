"""Train three model sizes on the same data/test set; keep the one with the best test loss.

All runs: doc mask, dropout 0.3, score>=10 training jokes, fixed >=50 test jokes,
batch 64 x 256 tokens, 5000 steps, 200 warmup steps, cosine decay, best-test checkpoint.
Larger models get smaller learning rates (standard practice; a confound to keep in mind).

Usage (Kaggle or local):  python -m src.joke_sweep [--smoke]
Outputs: data/sweep_<name>.{pt,json}, data/sweep_results.json, data/best.pt
"""

import json
import shutil
import sys

from src import joke_gpt

SIZES = {
    #        n_embd, heads, layers, lr
    "S": (192, 4, 4, 1e-3),
    "M": (384, 6, 6, 6e-4),
    "L": (512, 8, 8, 3e-4),
}
COMMON = ["--dropout", "0.3", "--train-data", "data/jokes_clean_s10.jsonl", "--batch", "64",
          "--iters", "5000", "--warmup", "200", "--eval-every", "250", "--eval-iters", "50"]


def main(smoke=False):
    results = {}
    for name, (emb, heads, layers, lr) in SIZES.items():
        out = f"data/sweep_{name}"
        argv = ["--smoke"] if smoke else [*COMMON, "--n-embd", str(emb), "--heads", str(heads),
                                          "--layers", str(layers), "--lr", str(lr)]
        print(f"\n===== {name}: n_embd {emb}, heads {heads}, layers {layers}, lr {lr}", flush=True)
        s = joke_gpt.main([*argv, "--out", out])
        final = s["history"][-1]
        results[name] = {"params": s["params"], "best_test": s["best_test"], "final_train": final["train"],
                         "final_test": final["test"], "seconds": s["seconds"], "out": out}
        with open("data/sweep_results.json", "w") as f:
            json.dump(results, f, indent=1)

    print("\nsize  params      best_test  final_train  final_test  minutes")
    for name, r in results.items():
        print(f"{name:4}  {r['params']:>10,}  {r['best_test']:.4f}     {r['final_train']:.4f}       "
              f"{r['final_test']:.4f}      {r['seconds'] / 60:.1f}")
    best = min(results, key=lambda n: results[n]["best_test"])
    shutil.copy(f"{results[best]['out']}.pt", "data/best.pt")
    print(f"best: {best} -> data/best.pt")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
