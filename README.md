# 🔍 Vision Transformer (ViT) from Scratch

This project implements the Vision Transformer (ViT) architecture from the paper  
**[“An Image is Worth 16x16 Words”](https://arxiv.org/abs/2010.11929)** using **PyTorch**, and trains it from scratch on the **CIFAR-10** dataset.

Built from the ground up — patch embedding, transformer encoder, class token, MLP head.

## ⚙️ Setup (Using `uv`)

Install `uv` (super fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Install all dependencies:

```bash
uv sync
```

Run the project
```bash
cd src && uv run main.py
```