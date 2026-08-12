"""veRL padding compatibility namespace.

vLLM 0.12 supplies its own FA2 CUDA kernels on A800. This namespace only
provides the padding helpers required by veRL when the optional upstream
flash-attn package is unavailable or incompatible with Cutlass.
"""

__version__ = "0.0.0-airline-padding-fallback"

