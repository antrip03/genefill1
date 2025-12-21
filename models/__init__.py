# models/__init__.py

from .encoder import CNNBiLSTMEncoder
from .decoder import GapDecoder
from .transformer_encoder import DNAMaskedEncoder  # NEW

__all__ = [
    "CNNBiLSTMEncoder",
    "GapDecoder",
    "DNAMaskedEncoder",   # NEW
]
