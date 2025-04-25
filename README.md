# 🔍 Vision Transformer (ViT) from Scratch

This project implements the Vision Transformer (ViT) architecture from the paper  
**[“An Image is Worth 16x16 Words”](https://arxiv.org/abs/2010.11929)** using **PyTorch**, and trains it from scratch on the **CIFAR-10** dataset.

Built from the ground up — patch embedding, transformer encoder, class token, MLP head.

## ⚙️ Setup (Using `uv`)

Install `uv` (super fast Python package manager):

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


**Install all dependencies:**

```bash
uv sync
```

**Run the project:**
```bash
cd uv run src/main.py
```

### Command-Line Arguments

You can customize the training process and model hyperparameters using command-line arguments:

*   `--epochs`: Number of training epochs (default: 10)
*   `--batch-size`: Input batch size (default: 32)
*   `--lr`: Learning rate (default: 3e-4)
*   `--weight-decay`: Weight decay (default: 1e-4)
*   `--patch-size`: Size of image patches (default: 4)
*   `--embed-dim`: Embedding dimension (default: 128)
*   `--num-heads`: Number of attention heads (default: 4)
*   `--mlp-dim`: Dimension of the MLP layer (default: 512)
*   `--num-layers`: Number of transformer encoder layers (default: 6)
*   `--dropout`: Dropout rate (default: 0.1)
*   `--data-dir`: Directory for storing dataset (default: './data')
*   `--save-path`: Path to save the trained model state dict (default: 'vit_cifar10_state.pth')
*   `--num-workers`: Number of data loading workers (default: 2)
*   `--device`: Device to use ('cuda', 'mps', 'cpu') (default: 'cuda')
*   `--use-compile`: Use `torch.compile` for potential speedup (flag, default: False)

**Example:**

```bash
cd src && uv run main.py --epochs 20 --batch-size 64 --lr 1e-4 --embed-dim 256 --mlp-dim 1024 --device mps --use-compile
```