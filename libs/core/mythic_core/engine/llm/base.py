"""
Abstract base class for LLM runners.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any, Union, List

from .config import ModelConfig
from .exceptions import LLMInitializationError, LLMInferenceError

logger = logging.getLogger(__name__)


class BaseLLMRunner(ABC):
    """Base class for LLM runners"""
    
    def __init__(self, config: Union[dict, ModelConfig]):
        """
        Initialize the LLM runner
        
        Args:
            config: Configuration dictionary or ModelConfig object
        """
        if isinstance(config, dict):
            self.config = self._parse_config(config)
        else:
            self.config = config
            
        self.model_path = self.config.model_path
        self.device = self.config.device
        self.llm = None
        self._initialized = False
        
        try:
            self._initialize_model()
            self._initialized = True
            logger.info(f"Successfully initialized {self.config.backend.value} model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.config.backend.value} model: {e}")
            raise LLMInitializationError(f"Model initialization failed: {e}") from e
    
    def _parse_config(self, config_dict: dict) -> ModelConfig:
        """Parse configuration dictionary into ModelConfig object"""
        from .config import BackendType, DeviceType, DEFAULT_CONFIG
        
        llm_config = config_dict.get('llm', {})
        
        # Extract backend with validation
        backend_str = llm_config.get('backend', 'llama.cpp')
        try:
            backend = BackendType(backend_str)
        except ValueError:
            raise ValueError(f"Unsupported backend: {backend_str}. Supported: {[b.value for b in BackendType]}")
        
        # Extract device with validation
        device_str = llm_config.get('device', 'auto')
        try:
            device = DeviceType(device_str)
        except ValueError:
            raise ValueError(f"Unsupported device: {device_str}. Supported: {[d.value for d in DeviceType]}")
        
        return ModelConfig(
            model_path=llm_config['model_path'],
            backend=backend,
            device=device,
            max_tokens=llm_config.get('max_tokens', DEFAULT_CONFIG['max_tokens']),
            temperature=llm_config.get('temperature', DEFAULT_CONFIG['temperature']),
            top_p=llm_config.get('top_p', DEFAULT_CONFIG['top_p']),
            top_k=llm_config.get('top_k', DEFAULT_CONFIG['top_k']),
            repeat_penalty=llm_config.get('repeat_penalty', DEFAULT_CONFIG['repeat_penalty']),
            stop_sequences=llm_config.get('stop_sequences', DEFAULT_CONFIG['stop_sequences'])
        )
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the specific model implementation"""
        pass
    
    @abstractmethod
    def _run_streaming(self, prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
        """Run LLM with streaming output"""
        pass
    
    @abstractmethod
    def _run_non_streaming(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run LLM with non-streaming output"""
        pass
    
    @abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Create a chat completion"""
        pass
    
    def run(self, prompt: str, max_tokens: Optional[int] = None, 
            temperature: Optional[float] = None, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Run the LLM with the given prompt
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated text or streaming generator
            
        Raises:
            LLMInferenceError: If inference fails
        """
        if not self._initialized:
            raise RuntimeError("LLM not initialized")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Validate parameters
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        try:
            if stream:
                return self._run_streaming(prompt, max_tokens, temperature)
            else:
                return self._run_non_streaming(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            raise LLMInferenceError(f"Inference failed: {e}") from e

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using the LLM (non-streaming version)"""
        return self.run(prompt, max_tokens=max_tokens, stream=False)

    def generate_text_streaming(self, prompt: str, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Generate text using the LLM (streaming version)"""
        return self.run(prompt, max_tokens=max_tokens, stream=True)

    def is_initialized(self) -> bool:
        """Check if the LLM is properly initialized"""
        return self._initialized and self.llm is not None

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        pass
    
    @abstractmethod
    def set_device(self, device: str) -> None:
        """Change the device for the model"""
        pass
    
    @abstractmethod
    def optimize_for_inference(self) -> None:
        """Optimize the model for inference"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        pass
    
    def get_config(self) -> ModelConfig:
        """Get the current configuration"""
        return self.config
    
    def reload_model(self) -> None:
        """Reload the model with current configuration"""
        logger.info("Reloading model...")
        self.llm = None
        self._initialized = False
        try:
            self._initialize_model()
            self._initialized = True
            logger.info("Model reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            raise LLMInitializationError(f"Model reload failed: {e}") from e
