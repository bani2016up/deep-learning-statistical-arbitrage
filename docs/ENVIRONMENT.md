# Environment Report

Detected on 2026-09-22 before project changes:

| Item | Detected value |
|---|---|
| OS | macOS 26.5.1, Darwin 25.5.0, arm64 |
| RAM | 16 GiB |
| Initially detected Python | 3.12.0 at `/usr/local/bin/python3` |
| Project Python | 3.13, selected by `.python-version` and managed by `uv` |
| Package/environment manager | `uv`; dependencies locked in `uv.lock` |
| Git | Existing parent repository, branch `main`; remotes preserved |
| Existing unrelated change | `../../../.obsidian/workspace.json` (untouched) |
| Initially detected PyTorch | 2.5.1 |
| Locked project PyTorch | 2.14.0 on the validated macOS arm64 environment |
| NVIDIA/CUDA | No NVIDIA device, `nvidia-smi`, or CUDA runtime available |
| Apple acceleration | MPS available and used |
| CPU fallback | Supported |

The project requires Python 3.13 or newer. Environment creation and command execution use
`uv`; manually created `venv` and direct `pip` workflows are unsupported. The target RTX
3080 could not be exercised on this host; CUDA selection and peak-memory reporting are
implemented for a CUDA host.
