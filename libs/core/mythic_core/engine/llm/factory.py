"""
Factory class for creating LLM runners.
"""

from typing import Union

from .base import BaseLLMRunner
from .config import ModelConfig, BackendType
from .llama_cpp import LlamaCppRunner
from .transformers import TransformersRunner


class LLMRunnerFactory:
    """Factory class to create the appropriate LLM runner based on backend configuration"""
    
    @staticmethod
    def create_runner(config: Union[dict, ModelConfig]) -> BaseLLMRunner:
        """
        Create an LLM runner based on the backend configuration
        
        Args:
            config: Configuration dictionary or ModelConfig object containing LLM settings
            
        Returns:
            An instance of the appropriate LLM runner class
            
        Raises:
            ValueError: If the backend is not supported
            LLMInitializationError: If model initialization fails
        """
        if isinstance(config, dict):
            backend_str = config.get('llm', {}).get('backend', 'llama.cpp')
        else:
            backend_str = config.backend.value
        
        try:
            backend = BackendType(backend_str)
        except ValueError:
            raise ValueError(f"Unsupported backend: {backend_str}. Supported backends: {[b.value for b in BackendType]}")
        
        if backend == BackendType.LLAMA_CPP:
            return LlamaCppRunner(config)
        elif backend == BackendType.TRANSFORMERS:
            return TransformersRunner(config)
        else:
            raise ValueError(f"Backend {backend} is not supported. Supported backends: {[b.value for b in BackendType]}")
