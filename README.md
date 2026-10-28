# NLP Transformers Labs

This repo containts the tutorial notebooks for the NLP course as part of the Master AI at the Vu. These notebooks cover:
- Lab 1 Building transformers from scratch (opens with a 15-minute text-generation warm-up)
- Lab 2 Using pretrained models

These notebooks are adapted from the original lab material in [cltl/nlp_transformer_tutorials](https://github.com/cltl/nlp_transformer_tutorials), which you can also browse on its website: [cltl.github.io/vunlp_tutorials](https://cltl.github.io/vunlp_tutorials).

## Setup

Use **Python 3.11 or 3.12** for the full lab environment. The pinned PyTorch and NumPy versions do not support Python 3.13/3.14. Create a virtual environment with an explicit Python version:

```bash
python3.12 -m venv nlp_transformers
source nlp_transformers/bin/activate
python -m pip install -r requirements.txt
```

Then launch Jupyter:

```bash
python -m notebook
```

In your notebook editor, select `nlp_transformers/bin/python` as the interpreter/kernel. If you already created the environment with a different Python version, recreate it using Python 3.11 or 3.12; upgrading pip alone does not change its Python version.
