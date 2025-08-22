"""
LLM runner implementation for llama.cpp backend.
"""

import logging
from typing import Generator, Dict, Any, Union, List

from .base import BaseLLMRunner
from .exceptions import LLMInitializationError, LLMInferenceError

logger = logging.getLogger(__name__)


class LlamaCppRunner(BaseLLMRunner):
    """LLM runner implementation for llama.cpp backend"""
    
    def _initialize_model(self) -> None:
        try:
            from llama_cpp import Llama
            
            # Filter out unsupported parameters for llama.cpp
            llama_config = {
                k: v for k, v in self.config.__dict__.items() 
                if k not in ['model_path', 'backend', 'device', 'stop_sequences']
            }
            
            self.llm = Llama(
                model_path=self.model_path,
                **llama_config
            )
            
        except ImportError:
            raise ImportError("llama.cpp is not installed. Please install it with `pip install llama-cpp-python`")
        except Exception as e:
            raise LLMInitializationError(f"Failed to initialize llama.cpp model: {e}") from e
    
    def _run_streaming(self, prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
        """Run LLM with streaming output"""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=self.config.stop_sequences,
                stream=True,
            )
            
            for chunk in response:
                if chunk and 'choices' in chunk and chunk['choices']:
                    choice = chunk['choices'][0]
                    if choice.get('finish_reason') is not None:
                        # Final chunk
                        content = choice.get('delta', {}).get('content', '')
                        if content:
                            yield content
                        break
                    else:
                        # Regular content chunk
                        content = choice.get('delta', {}).get('content', '')
                        if content:
                            yield content
                            
        except Exception as e:
            logger.error(f"Streaming LLM inference failed: {e}")
            raise LLMInferenceError(f"Streaming inference failed: {e}") from e

    def _run_non_streaming(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run LLM with non-streaming output"""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=self.config.stop_sequences,
                stream=False,
            )
            
            if response and 'choices' in response and response['choices']:
                return response['choices'][0]['message']['content']
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Non-streaming LLM inference failed: {e}")
            raise LLMInferenceError(f"Non-streaming inference failed: {e}") from e
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Create a chat completion using llama.cpp"""
        if not self.is_initialized():
            raise RuntimeError("LLM not initialized")
        
        # Get generation parameters
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        stream = kwargs.get('stream', False)
        
        # llama.cpp has native chat completion support
        return self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_initialized():
            return {}
        
        try:
            # Try to get model information if available
            if hasattr(self.llm, 'model'):
                return {
                    'model_path': self.model_path,
                    'backend': self.config.backend.value,
                    'device': self.device.value,
                    'context_size': getattr(self.llm.model, 'n_ctx', 'unknown'),
                    'vocab_size': getattr(self.llm.model, 'n_vocab', 'unknown'),
                    'model_type': 'llama.cpp',
                    'initialized': self._initialized
                }
            else:
                return {
                    'model_path': self.model_path,
                    'backend': self.config.backend.value,
                    'device': self.device.value,
                    'model_type': 'llama.cpp',
                    'initialized': self._initialized
                }
                
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {
                'model_path': self.model_path,
                'backend': self.config.backend.value,
                'device': self.device.value,
                'model_type': 'llama.cpp',
                'initialized': self._initialized
            }
    
    def set_device(self, device: str) -> None:
        """Change the device for the model (llama.cpp doesn't support device changes)"""
        logger.warning("Device change not supported for llama.cpp backend")
    
    def optimize_for_inference(self) -> None:
        """Optimize the model for inference (llama.cpp is already optimized)"""
        logger.info("llama.cpp models are already optimized for inference")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information (limited for llama.cpp)"""
        logger.warning("Detailed memory usage info not available for llama.cpp backend")
        return {
            'backend': self.config.backend.value,
            'note': 'Detailed memory usage not available for llama.cpp'
        }
