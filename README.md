# LLM Systems Exercises

Small, inspectable experiments for understanding transformer inference and model training. The repository includes a NumPy attention reference, a Tiny Shakespeare GPT notebook, grouped-query attention experiments, and a small joke-model pipeline.

## Experiment results

The notebook compares several architecture choices. The compact result index includes configuration notes and caveats:

- [MHA, GQA, and MQA comparison](docs/experiment-results.md#multi-head-vs-grouped-query-vs-multi-query)
- [Tiny Shakespeare model progression](docs/experiment-results.md#tiny-shakespeare-architecture-progression)
- [Tokenizer, context, and activation runs](docs/experiment-results.md#context-dropout-projection-and-activation)
- [Joke GPT size sweep](docs/experiment-results.md#joke-gpt-model-size-sweep)

The saved notebook outputs are historical results. Training differences use one seed and small evaluation samples; treat close scores as inconclusive.

## Contents

- `src/attention.py`: single-head scaled dot-product attention in NumPy.
- `src/tiny_shakespeare.ipynb`: GPT-from-scratch learning notebook, with saved experiment outputs.
- `src/gpt_colab.py`: standalone Tiny Shakespeare GPT training script.
- `src/gpt_experiments.py`: configurable MHA/GQA/MQA and tokenizer/context experiments.
- `src/jokes_data.py`, `src/bpe.py`, `src/joke_dataset.py`, `src/joke_gpt.py`: data preparation and joke-model exercises.
- `src/smol_finetune.py`: optional full fine-tuning exercise for SmolLM2.
- `tests/`: unit and behavior tests.

Raw joke datasets, trained checkpoints, cloud-run outputs, credentials, and personal learning notes are not included. Obtain and review any dataset under its own terms before use. Tiny Shakespeare text is also not bundled; notebook users can download the corpus from [Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) into `src/input.txt`. The standalone GPT scripts download it when needed.

## Setup

Python 3.9 or newer. Install PyTorch for your platform, then install the remaining dependencies:

```sh
python -m pip install -r requirements.txt
```

Run the test suite with `python -m pytest`. Run lint with `ruff check src tests`.

The NumPy attention example is CPU-only. PyTorch scripts use CUDA when available; their local CPU timing does not predict GPU throughput. Joke-model runs require local dataset files under `data/`; see each script's usage text for expected inputs.

## References

- [MHA/GQA/MQA implementation and result discussion](src/tiny_shakespeare.ipynb)
- [Experiment result summary](docs/experiment-results.md)
- [Kaggle experiment plan](src/kaggle_plan.md)
- [Karpathy's char-rnn repository and Tiny Shakespeare source](https://github.com/karpathy/char-rnn)
