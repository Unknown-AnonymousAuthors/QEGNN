# Add this in a Google Colab cell to install the correct version of Pytorch Geometric.
import torch
import os

def _format_pytorch_version(version):
  return version.split('+')[0]

def _format_cuda_version(version):
  return 'cu' + version.replace('.', '')


TORCH_version = torch.__version__
TORCH = _format_pytorch_version(TORCH_version)
CUDA_version = torch.version.cuda
CUDA = _format_cuda_version(CUDA_version)

# !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
# !pip install torch-geometric