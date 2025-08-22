"""
LLM runner implementation for transformers backend.
"""

import logging
import time
from typing import Generator, Dict, Any, Union, List

from .base import BaseLLMRunner
from .config import DeviceType
from .exceptions import LLMInitializationError, LLMInferenceError

logger = logging.getLogger(__name__)


class TransformersRunner(BaseLLMRunner):
    """LLM runner implementation for transformers backend"""
    
    def _initialize_model(self) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # Get device configuration
            if self.device == DeviceType.AUTO:
                if torch.cuda.is_available():
                    self.device = DeviceType.CUDA
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = DeviceType.MPS
                else:
                    self.device = DeviceType.CPU
            
            device_str = self.device.value
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
                device_map=device_str if device_str in ["cuda", "mps"] else None,
                **{k: v for k, v in self.config.__dict__.items() 
                if k not in ['model_path', 'backend', 'device', 'stop_sequences']}
            )
            
            # Create pipeline for easier text generation
            self.llm = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_str,
                torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
            )
            
        except ImportError:
            raise ImportError("transformers is not installed. Please install it with `pip install transformers torch`")
        except Exception as e:
            raise LLMInitializationError(f"Failed to initialize transformers model: {e}") from e
    
    def _run_streaming(self, prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
        """Run LLM with streaming output"""
        try:
            # For transformers, we need to simulate streaming since pipeline doesn't support it natively
            # We'll generate the full response and yield it character by character
            full_response = self._run_non_streaming(prompt, max_tokens, temperature)
            
            # Yield character by character to simulate streaming
            for char in full_response:
                yield char
                # Small delay to simulate real streaming
                time.sleep(0.01)
                        
        except Exception as e:
            logger.error(f"Streaming LLM inference failed: {e}")
            raise LLMInferenceError(f"Streaming inference failed: {e}") from e

    def _run_non_streaming(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run LLM with non-streaming output"""
        try:
            # Use the transformers pipeline for text generation
            generation_config = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id,
            }
            
            # Add stop sequences if specified
            if self.config.stop_sequences:
                generation_config['eos_token_id'] = self.tokenizer.encode(self.config.stop_sequences[0])[0] if self.config.stop_sequences else None
            
            # Generate text
            outputs = self.llm(
                prompt,
                **generation_config
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Remove the input prompt from the output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Non-streaming LLM inference failed: {e}")
            raise LLMInferenceError(f"Non-streaming inference failed: {e}") from e
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Create a chat completion using transformers"""
        if not self.is_initialized():
            raise RuntimeError("LLM not initialized")
        
        # Get generation parameters
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        stream = kwargs.get('stream', False)
        
        if stream:
            # Return a generator that yields chunks
            return self._transformers_chat_stream(messages, max_tokens, temperature, **kwargs)
        else:
            # Return a single response
            return self._transformers_chat_completion(messages, max_tokens, temperature, **kwargs)
    
    def _transformers_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, **kwargs) -> Dict[str, Any]:
        """Handle chat completion for transformers backend"""
        # Format conversation for the model
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        response_text = self._run_non_streaming(prompt, max_tokens, temperature)
        
        # Return in OpenAI-compatible format
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': response_text
                },
                'finish_reason': 'stop',
                'index': 0
            }],
            'usage': {
                'prompt_tokens': len(self.tokenizer.encode(prompt)) if hasattr(self, 'tokenizer') else 0,
                'completion_tokens': len(self.tokenizer.encode(response_text)) if hasattr(self, 'tokenizer') else 0,
                'total_tokens': len(self.tokenizer.encode(prompt + response_text)) if hasattr(self, 'tokenizer') else 0
            }
        }
    
    def _transformers_chat_stream(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Handle streaming chat completion for transformers backend"""
        # Format conversation for the model
        prompt = self._format_chat_prompt(messages)
        
        # Generate response
        response_text = self._run_non_streaming(prompt, max_tokens, temperature)
        
        # Yield chunks in OpenAI-compatible streaming format
        for i, char in enumerate(response_text):
            yield {
                'choices': [{
                    'delta': {
                        'content': char
                    },
                    'finish_reason': None if i < len(response_text) - 1 else 'stop',
                    'index': 0
                }]
            }
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt string for transformers"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        # Join with newlines and add assistant prefix for the response
        prompt = '\n'.join(prompt_parts) + '\nAssistant: '
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_initialized():
            return {}
        
        try:
            if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                return {
                    'model_path': self.model_path,
                    'backend': self.config.backend.value,
                    'device': self.device.value,
                    'model_type': type(self.model).__name__,
                    'vocab_size': getattr(self.model.config, 'vocab_size', 'unknown'),
                    'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', 'unknown'),
                    'hidden_size': getattr(self.model.config, 'hidden_size', 'unknown'),
                    'num_attention_heads': getattr(self.model.config, 'num_attention_heads', 'unknown'),
                    'num_layers': getattr(self.model.config, 'num_hidden_layers', 'unknown'),
                    'initialized': self._initialized
                }
            else:
                return {
                    'model_path': self.model_path,
                    'backend': self.config.backend.value,
                    'device': self.device.value,
                    'initialized': self._initialized
                }
                
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {
                'model_path': self.model_path,
                'backend': self.config.backend.value,
                'device': self.device.value,
                'initialized': self._initialized
            }
    
    def set_device(self, device: str) -> None:
        """Change the device for the model (transformers backend only)"""
        if hasattr(self, 'model'):
            import torch
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            if device != self.device.value:
                from .config import DeviceType
                self.device = DeviceType(device)
                self.model = self.model.to(device)
                logger.info(f"Moved transformers model to {device}")
        else:
            logger.warning("Device change only supported for transformers backend")
    
    def optimize_for_inference(self) -> None:
        """Optimize the model for inference (transformers backend only)"""
        if hasattr(self, 'model'):
            try:
                import torch
                
                # Enable evaluation mode
                self.model.eval()
                
                # Enable torch.compile if available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Applied torch.compile optimization")
                
                # Enable half precision for CUDA
                if self.device.value == "cuda" and torch.cuda.is_available():
                    self.model = self.model.half()
                    logger.info("Applied half precision optimization")
                
                logger.info("Model optimized for inference")
                
            except Exception as e:
                logger.warning(f"Could not optimize model: {e}")
        else:
            logger.warning("Model optimization only supported for transformers backend")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information (transformers backend only)"""
        if hasattr(self, 'model'):
            try:
                import torch
                import psutil
                
                # Get model parameters count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                # Get GPU memory if available
                gpu_memory = {}
                if self.device.value == "cuda" and torch.cuda.is_available():
                    gpu_memory = {
                        'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                        'cached': torch.cuda.memory_reserved() / (1024**3),      # GB
                        'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    }
                
                # Get system memory
                system_memory = psutil.virtual_memory()
                
                return {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'gpu_memory_gb': gpu_memory,
                    'system_memory_gb': {
                        'total': system_memory.total / (1024**3),
                        'available': system_memory.available / (1024**3),
                        'used': system_memory.used / (1024**3)
                    }
                }
                
            except Exception as e:
                logger.warning(f"Could not get memory usage: {e}")
                return {}
        else:
            logger.warning("Memory usage info only available for transformers backend")
            return {}
