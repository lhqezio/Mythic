"""
LLM Runner Package

This package provides a modular interface for running Large Language Models
with support for multiple backends including llama.cpp and transformers.
"""

from .base import BaseLLMRunner
from .llama_cpp_runner import LlamaCppRunner
from .transformers import TransformersRunner
from .factory import LLMRunnerFactory
from .config import ModelConfig, BackendType, DeviceType
from .exceptions import LLMError, LLMInitializationError, LLMInferenceError

# Backward compatibility
LLMRunner = LLMRunnerFactory

__all__ = [
    'BaseLLMRunner',
    'LlamaCppRunner', 
    'TransformersRunner',
    'LLMRunnerFactory',
    'LLMRunner',  # Backward compatibility
    'ModelConfig',
    'BackendType',
    'DeviceType',
    'LLMError',
    'LLMInitializationError',
    'LLMInferenceError'
]
