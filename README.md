# Linear Fitting with Custom CUDA Kernel

This project demonstrates how to implement a simple linear regression model using a **custom CUDA kernel** and integrate it with **PyTorch's autograd system** via `torch.autograd.Function`.

## Installation

This project uses `pyproject.toml` to manage dependencies.

Clone the repository:
```bash
git clone git@github.com:sh1027/pytorch-custom-cuda-tutorial.git
cd pytorch-custom-cuda-tutorial
```
Install Python dependencies using [uv](https://github.com/astral-sh/uv):
```bash
# Make sure you have uv installed. Check (https://docs.astral.sh/uv/#installation) for the details
uv sync
```
Or, with `pip`:
```bash
pip install .
```
### Additional system dependencies
- CUDA Toolkit
- ffmpeg (for saving videos)