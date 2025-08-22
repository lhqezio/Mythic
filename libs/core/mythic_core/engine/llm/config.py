"""
Configuration classes and constants for the LLM system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

# Default configuration values
DEFAULT_CONFIG = {
    'max_tokens': 500,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'repeat_penalty': 1.1,
    'stop_sequences': ["\n\n", "###", "END", "STOP"]
}


class BackendType(Enum):
    """Supported LLM backend types"""
    LLAMA_CPP = "llama.cpp"
    TRANSFORMERS = "transformers"


class DeviceType(Enum):
    """Supported device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    model_path: str
    backend: BackendType
    device: DeviceType
    max_tokens: int = DEFAULT_CONFIG['max_tokens']
    temperature: float = DEFAULT_CONFIG['temperature']
    top_p: float = DEFAULT_CONFIG['top_p']
    top_k: int = DEFAULT_CONFIG['top_k']
    repeat_penalty: float = DEFAULT_CONFIG['repeat_penalty']
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = DEFAULT_CONFIG['stop_sequences']
        
        # Validate configuration
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.repeat_penalty < 0:
            raise ValueError("repeat_penalty must be non-negative")
