# Kaggle experiment plan

Script: `gpt_experiments.py` (same architecture as the notebook's GQA model). Every run appends a row to `results.csv`; `python gpt_experiments.py --show` prints the table.

Defaults unless a run says otherwise: char tokens, MHA, n_embd 384, 6 heads, 6 layers, block 256, batch 64, dropout 0.2, lr 3e-4, GELU, 5000 steps.

Run times are estimates. Each run prints its own projection after 50 steps (`~N ms/iter -> about M min`); if a session will overrun, skip its last run.

## Setup (once per session)

1. kaggle.com → Create → New Notebook.
2. Right panel → Session options: Accelerator **GPU T4 x2** (or P100), Internet **On** (needs phone verification once).
3. Cell 1: `%%writefile gpt_experiments.py` followed by the script.
4. If you have `results.csv` from earlier sessions: upload it (Add Input → Upload) and copy it in: `!cp /kaggle/input/*/results.csv .`
5. Run the session's commands, each in its own cell: `!python gpt_experiments.py ...`
6. End: `!python gpt_experiments.py --show`, then download `results.csv` and the `sample_*.txt` files from the Output panel.

Tip: **Save Version → Save & Run All** runs the whole notebook in the background (no browser needed) and keeps all outputs.

## Session 1: model size (prediction first: where does test loss stop improving, and where does the gap grow?)

```
!python gpt_experiments.py --name S-small --n_embd 192 --n_head 6 --n_layer 4
!python gpt_experiments.py --name M-base
!python gpt_experiments.py --name L-large --n_embd 512 --n_head 8 --n_layer 8
```
About 1.9M, 10.8M and 25M parameters. L-large is the longest run.

## Session 2: MHA vs GQA vs MQA at 10.8M (M-base from session 1 is the MHA baseline)

```
!python gpt_experiments.py --name M-gqa3 --kv_groups 3
!python gpt_experiments.py --name M-gqa2 --kv_groups 2
!python gpt_experiments.py --name M-mqa --kv_groups 1
```
Compare `test_bpc` against `kv_floats_per_token` (MHA 4608, GQA-3 2304, GQA-2 1536, MQA 768).

## Session 3: char vs BPE at 10.8M (compare test_bpc, not test_loss)

```
!python gpt_experiments.py --name M-bpe512 --tokenizer bpe --bpe_vocab 512
!python gpt_experiments.py --name M-bpe1000 --tokenizer bpe --bpe_vocab 1000
!python gpt_experiments.py --name M-bpe2000 --tokenizer bpe --bpe_vocab 2000
!python gpt_experiments.py --name M-bpe1000-span --tokenizer bpe --bpe_vocab 1000 --block_size 110
```
The last run covers about 256 characters, the same span as M-base, so it separates the tokenizer effect from the longer-context effect. BPE training takes about 30 s to 2 min the first time per vocab size.

## Session 4: dropout and context length at 10.8M

```
!python gpt_experiments.py --name M-drop0 --dropout 0.0
!python gpt_experiments.py --name M-drop0.1 --dropout 0.1
!python gpt_experiments.py --name M-block128 --block_size 128
!python gpt_experiments.py --name M-block512 --block_size 512 --batch_size 32
```
M-block512 halves the batch so tokens per step stay the same and memory fits.

## Caveats

- One seed per run. Differences under about 0.02 in test_bpc can be noise; repeat with `--seed 1` and `--seed 2` before drawing conclusions from them.
- BPE runs have more parameters (bigger embedding and lm_head).

## References

- Completed notebook MHA/GQA/MQA results: [experiment summary](https://github.com/cerdo03/llm-systems-exercises/blob/main/docs/experiment-results.md#multi-head-vs-grouped-query-vs-multi-query).
- Implementation and saved notebook outputs: [Tiny Shakespeare notebook](https://github.com/cerdo03/llm-systems-exercises/blob/main/src/tiny_shakespeare.ipynb).
- These Kaggle GQA-3/GQA-2/MQA commands are a proposed plan; no result files were found for them.
