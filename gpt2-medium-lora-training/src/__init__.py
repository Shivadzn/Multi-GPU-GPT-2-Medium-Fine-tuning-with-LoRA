"""
GPT-2 LoRA Fine-tuning Package

This package provides tools for fine-tuning GPT-2 models using LoRA (Low-Rank Adaptation).
"""

__version__ = "0.1.0"

# Import main components to make them available at the package level
from .config import load_config, update_config  # noqa: F401
from .models.gpt2_lora import get_model  # noqa: F401
from .data.dataset import get_dataset  # noqa: F401
from .training.trainer import train  # noqa: F401
