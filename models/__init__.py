# models/__init__.py

from .encoder import CNNBiLSTMEncoder  # CHANGED: use new encoder
from .decoder import GapDecoder

__all__ = ["CNNBiLSTMEncoder", "GapDecoder"]
