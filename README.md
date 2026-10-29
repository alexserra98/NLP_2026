# NLP Transformers Labs

This repo containts the tutorial notebooks for the NLP course as part of the NLP course at University of Trieste. These notebooks cover:
- Lab 1 Building transformers from scratch (opens with a 15-minute text-generation warm-up)
- Lab 2 Using pretrained models

These notebooks are adapted from the original lab material in [cltl/nlp_transformer_tutorials](https://github.com/cltl/nlp_transformer_tutorials), which you can also browse on its website: [cltl.github.io/vunlp_tutorials](https://cltl.github.io/vunlp_tutorials).

## Setup

This repo uses [**uv**](https://docs.astral.sh/uv/) to manage the environment. Install it first if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Use **Python 3.11 or 3.12** for the full lab environment. The pinned PyTorch and NumPy versions do not support Python 3.13/3.14. Create the virtual environment with an explicit Python version and install the dependencies:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

This creates a `.venv/` directory. Activate it with:

```bash
source .venv/bin/activate
```

Then launch Jupyter (with the environment activated):

```bash
python -m notebook
```

Or, without activating, run commands through uv directly:

```bash
uv run python -m notebook
```

In your notebook editor, select `.venv/bin/python` as the interpreter/kernel. If you already created the environment with a different Python version, recreate it using Python 3.11 or 3.12 (`uv venv --python 3.12` will overwrite the existing `.venv`).
