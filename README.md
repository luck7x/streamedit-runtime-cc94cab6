# StreamEdit Runtime

Deployment snapshot for a single NVIDIA RTX 5090 (32 GB).

## Setup

Follow [DEPLOYMENT.md](DEPLOYMENT.md) to install Python 3.10, PyTorch CUDA 12.8,
SageAttention with the bundled patch, and the in-tree `streamedit_ops` extension.
Download the model weights separately using the Hugging Face instructions in that guide.
Weights, development history, recordings and caches are not included.

## Start

From the repository root, after completing setup:

```bash
conda activate streamedit
STREAMEDIT_SAGE_ATTN=1 STREAMEDIT_FP8_FAST_ACCUM=1 STREAMEDIT_LOW_VRAM=1 \
STREAMEDIT_HOST=127.0.0.1 bash deploy/run_server.sh
```

Open http://localhost:8080. Defaults are 840 x 480 at a configured 24 FPS;
actual performance and memory use must be verified on the target machine.

The supplied launcher targets Linux. Windows/WSL2 requires separate host setup
and GPU validation. Do not expose this service directly to the public Internet.