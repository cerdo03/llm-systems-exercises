# Experiment results

Results below are transcribed from saved notebook outputs or local experiment JSON files. They are useful for understanding the experiments, not controlled benchmark claims. Most notebook runs used one fixed seed (1337); small test-loss differences may be evaluation noise. Generated samples are omitted.

## Multi-head vs grouped-query vs multi-query

Saved Tiny Shakespeare notebook runs use the same compact model: 4 query heads, head width 8, context block 32, 5,000 training steps, seed 1337, and no dropout. MHA is the baseline (D); GQA uses two KV groups (H); MQA uses one KV group (G).

| Run | KV groups | Parameters | Train loss | Test loss | Train-test gap | KV floats per token per block |
|---|---:|---:|---:|---:|---:|---:|
| D — MHA | 4 | 43,137 | 1.8410 | 1.9651 | 0.1240 | 64 |
| H — GQA | 2 | 40,065 | 1.8560 | 1.9819 | 0.1258 | 32 |
| G — MQA | 1 | 38,529 | 1.8814 | 1.9902 | 0.1088 | 16 |

GQA maps query heads 0 and 1 to KV group 0, and query heads 2 and 3 to KV group 1. The cache figures count key and value elements only: `2 × KV groups × head width`. They show a 2× reduction per halving of KV groups; they do not measure latency or imply proportional speedup. This notebook implementation repeats KV tensors across query heads for computation and does not implement an incremental KV cache.

The measured test losses rise slightly as KV groups decrease. With one seed and a small evaluation set, these differences do not establish a robust quality ranking.

## Tiny Shakespeare architecture progression

Saved test losses from the notebook's incremental model builds:

| Model | Test loss |
|---|---:|
| Token-only baseline | 2.4937 |
| Token + position embeddings | 2.4880 |
| Causal average | 2.8140 |
| One self-attention head | 2.4100 |
| Four-head MHA | 2.2589 |
| One block: MHA + feed-forward network | 2.1888 |
| Three blocks, no residual connections | 2.2215 |
| Three blocks + residual connections | 2.0954 |
| Three blocks + residuals + LayerNorm | 2.0719 |

The runs show that learned attention improved on uniform causal averaging in this setup. Adding model components changed both capacity and optimization, so this is an instructional progression rather than a one-factor ablation.

## Context, dropout, projection, and activation

Follow-on experiments compare against the LayerNorm model at block 8, 5,000 steps (test loss 2.0719), then mostly against model D at block 32.

| Run | Change | Test loss | Notes |
|---|---|---:|---|
| A | Block 8, 10,000 steps | 2.0521 | More steps; train-test gap increased from 0.0731 to 0.1420. |
| B | Block 32, 5,000 steps | 1.9682 | Four times the context length. |
| C | B + dropout 0.2 | 2.0573 | Smaller gap (0.0547); higher test loss in this run. |
| D | B + output projection; ReLU FFN | 1.9651 | 43,137 parameters. |
| E | D without FFN activation | 2.0871 | Linear FFN layers performed worse here. |
| F | D with GELU FFN | 1.9537 | Lowest notebook test loss among these variants. |

The block-size comparison changes context and positional-embedding capacity. D/E/F use matched initialization seed and training steps, though single-seed results still limit conclusions.

## Character vs BPE

BPE-1000 has a larger vocabulary and about 103k parameters versus 43k for the character-token D baseline. Results use bits per character (BPC), not raw token loss.

| Setup | Context | Test loss/token | Test BPC |
|---|---:|---:|---:|
| Character baseline D | 32 characters | 1.9651 | 2.8350 |
| BPE-1000 | 13 tokens (~32 characters) | 4.0788 | 2.5273 |
| BPE-1000 | 32 tokens (~81 characters) | 4.0126 | 2.4863 |

The BPE runs also have more embedding/output parameters. The longer-context BPE run changes tokenization and character span together, so these results do not isolate one cause.

## Joke GPT model-size sweep

Separate Kaggle T4 runs trained on cleaned Reddit jokes with document masking, dropout 0.3, batch size 64, 5,000 steps, and score-threshold-10 training data. Values come from saved `sweep_*.json` files; raw data and checkpoints are not published here.

| Size | Parameters | Best test loss | Final train loss | Final test loss | Runtime |
|---|---:|---:|---:|---:|---:|
| S | 2,221,313 | 2.8286 | 2.5403 | 2.8286 | 9.4 min |
| M | 11,527,169 | 2.6383 | 1.5567 | 2.6466 | 23.9 min |
| L | 26,389,505 | 2.6359 | 1.3277 | 2.6688 | 47.8 min |

M and L are effectively tied at best test loss for this single-seed run; L's final train-test gap indicates stronger memorization. The evaluation used 50 batches at each checkpoint.

## References

- [Repository source and tests](https://github.com/cerdo03/llm-systems-exercises)
- [Notebook implementation and saved outputs](https://github.com/cerdo03/llm-systems-exercises/blob/main/src/tiny_shakespeare.ipynb)
- [Experiment configurations](https://github.com/cerdo03/llm-systems-exercises/blob/main/src/kaggle_plan.md)
- [Karpathy's Tiny Shakespeare source corpus](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
