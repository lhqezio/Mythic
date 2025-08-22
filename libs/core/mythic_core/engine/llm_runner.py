import logging
from typing import Generator, Optional, Dict, Any

logger = logging.getLogger(__name__)

class LLMRunner:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config['llm']['model_path']
        self.backend = config.get('llm', {}).get('backend', 'llama.cpp')
        self.device = config.get('llm', {}).get('device', 'auto')
        self.llm = None

        if self.backend == "llama.cpp":
            try:
                from llama_cpp import Llama
                # Initialize with basic parameters first
                self.llm = Llama(
                    model_path=self.model_path,
                    **{k: v for k, v in config['llm'].items() 
                    if k not in ['model_path', 'model_filename', 'backend', 'device']}
                )
                logger.info(f"Initialized llama.cpp model from {self.model_path}")
            except ImportError:
                raise ImportError("llama.cpp is not installed. Please install it with `pip install llama-cpp-python`")
        
        elif self.backend == "transformers":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch
                
                # Get device configuration
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device if self.device == "cuda" else None,
                    **{k: v for k, v in config['llm'].items() 
                    if k not in ['model_path', 'model_filename', 'backend', 'device']}
                )
                
                # Create pipeline for easier text generation
                self.llm = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                
                logger.info(f"Initialized transformers model from {self.model_path} on {self.device}")
                
            except ImportError:
                raise ImportError("transformers is not installed. Please install it with `pip install transformers torch")
        
        else:
            raise ValueError(f"Backend {self.backend} is not supported")

    def run(self, prompt: str, max_tokens: Optional[int] = None, 
            temperature: Optional[float] = None, stream: bool = False) -> Any:
        """
        Run the LLM with the given prompt
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated text or streaming generator
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        # Use config defaults if not specified
        if max_tokens is None:
            max_tokens = self.config['llm'].get('max_tokens', 500)
        if temperature is None:
            temperature = self.config['llm'].get('temperature', 0.7)
        
        try:
            if stream:
                return self._run_streaming(prompt, max_tokens, temperature)
            else:
                return self._run_non_streaming(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            raise

    def _run_streaming(self, prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
        """Run LLM with streaming output"""
        try:
            if self.backend == "llama.cpp":
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config['llm'].get('top_p', 0.9),
                    top_k=self.config['llm'].get('top_k', 40),
                    repeat_penalty=self.config['llm'].get('repeat_penalty', 1.1),
                    stop=self.config['llm'].get('stop_sequences', ["\n\n", "###", "END", "STOP"]),
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
            
            elif self.backend == "transformers":
                # For transformers, we need to simulate streaming since pipeline doesn't support it natively
                # We'll generate the full response and yield it character by character
                full_response = self._run_non_streaming(prompt, max_tokens, temperature)
                
                # Yield character by character to simulate streaming
                for char in full_response:
                    yield char
                    # Small delay to simulate real streaming
                    import time
                    time.sleep(0.01)
                            
        except Exception as e:
            logger.error(f"Streaming LLM inference failed: {e}")
            raise

    def _run_non_streaming(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run LLM with non-streaming output"""
        try:
            if self.backend == "llama.cpp":
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config['llm'].get('top_p', 0.9),
                    top_k=self.config['llm'].get('top_k', 40),
                    repeat_penalty=self.config['llm'].get('repeat_penalty', 1.1),
                    stop=self.config['llm'].get('stop_sequences', ["\n\n", "###", "END", "STOP"]),
                    stream=False,
                )
                
                if response and 'choices' in response and response['choices']:
                    return response['choices'][0]['message']['content']
                else:
                    return ""
            
            elif self.backend == "transformers":
                # Use the transformers pipeline for text generation
                generation_config = {
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': self.config['llm'].get('top_p', 0.9),
                    'top_k': self.config['llm'].get('top_k', 40),
                    'do_sample': temperature > 0,
                    'pad_token_id': self.tokenizer.eos_token_id,
                }
                
                # Add stop sequences if specified
                stop_sequences = self.config['llm'].get('stop_sequences', ["\n\n", "###", "END", "STOP"])
                if stop_sequences:
                    generation_config['eos_token_id'] = self.tokenizer.encode(stop_sequences[0])[0] if stop_sequences else None
                
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
            raise

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using the LLM (non-streaming version)"""
        return self.run(prompt, max_tokens=max_tokens, stream=False)

    def generate_text_streaming(self, prompt: str, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Generate text using the LLM (streaming version)"""
        return self.run(prompt, max_tokens=max_tokens, stream=True)

    def is_initialized(self) -> bool:
        """Check if the LLM is properly initialized"""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.llm:
            return {}
        
        try:
            if self.backend == "llama.cpp":
                # Try to get model information if available
                if hasattr(self.llm, 'model'):
                    return {
                        'model_path': self.model_path,
                        'backend': self.backend,
                        'device': self.device,
                        'context_size': getattr(self.llm.model, 'n_ctx', 'unknown'),
                        'vocab_size': getattr(self.llm.model, 'n_vocab', 'unknown'),
                    }
                else:
                    return {
                        'model_path': self.model_path,
                        'backend': self.backend,
                        'device': self.device,
                    }
            
            elif self.backend == "transformers":
                if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                    return {
                        'model_path': self.model_path,
                        'backend': self.backend,
                        'device': self.device,
                        'model_type': type(self.model).__name__,
                        'vocab_size': getattr(self.model.config, 'vocab_size', 'unknown'),
                        'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', 'unknown'),
                        'hidden_size': getattr(self.model.config, 'hidden_size', 'unknown'),
                        'num_attention_heads': getattr(self.model.config, 'num_attention_heads', 'unknown'),
                        'num_layers': getattr(self.model.config, 'num_hidden_layers', 'unknown'),
                    }
                else:
                    return {
                        'model_path': self.model_path,
                        'backend': self.backend,
                        'device': self.device,
                    }
            
            else:
                return {
                    'model_path': self.model_path,
                    'backend': self.backend,
                    'device': self.device,
                }
                
        except Exception as e:
            logger.warning(f"Could not retrieve model info: {e}")
            return {
                'model_path': self.model_path,
                'backend': self.backend,
                'device': self.device,
            }
    
    def set_device(self, device: str):
        """Change the device for the model (transformers backend only)"""
        if self.backend == "transformers" and hasattr(self, 'model'):
            import torch
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if device != self.device:
                self.device = device
                self.model = self.model.to(device)
                logger.info(f"Moved transformers model to {device}")
        else:
            logger.warning("Device change only supported for transformers backend")
    
    def optimize_for_inference(self):
        """Optimize the model for inference (transformers backend only)"""
        if self.backend == "transformers" and hasattr(self, 'model'):
            try:
                import torch
                
                # Enable evaluation mode
                self.model.eval()
                
                # Enable torch.compile if available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Applied torch.compile optimization")
                
                # Enable half precision for CUDA
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.half()
                    logger.info("Applied half precision optimization")
                
                logger.info("Model optimized for inference")
                
            except Exception as e:
                logger.warning(f"Could not optimize model: {e}")
        else:
            logger.warning("Model optimization only supported for transformers backend")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information (transformers backend only)"""
        if self.backend == "transformers" and hasattr(self, 'model'):
            try:
                import torch
                import psutil
                
                # Get model parameters count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                # Get GPU memory if available
                gpu_memory = {}
                if self.device == "cuda" and torch.cuda.is_available():
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
    
    def create_chat_completion(self, messages: list, **kwargs):
        """
        Create a chat completion (unified interface for both backends)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Chat completion response in OpenAI-compatible format
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        if not user_message:
            raise ValueError("No user message found in messages")
        
        # Get generation parameters
        max_tokens = kwargs.get('max_tokens', self.config['llm'].get('max_tokens', 500))
        temperature = kwargs.get('temperature', self.config['llm'].get('temperature', 0.7))
        stream = kwargs.get('stream', False)
        
        if self.backend == "llama.cpp":
            # llama.cpp has native chat completion support
            return self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        
        elif self.backend == "transformers":
            # For transformers, we need to format the conversation and generate
            if stream:
                # Return a generator that yields chunks
                return self._transformers_chat_stream(messages, max_tokens, temperature, **kwargs)
            else:
                # Return a single response
                return self._transformers_chat_completion(messages, max_tokens, temperature, **kwargs)
        
        else:
            raise ValueError(f"Chat completion not supported for backend {self.backend}")
    
    def _transformers_chat_completion(self, messages: list, max_tokens: int, temperature: float, **kwargs) -> dict:
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
    
    def _transformers_chat_stream(self, messages: list, max_tokens: int, temperature: float, **kwargs):
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
    
    def _format_chat_prompt(self, messages: list) -> str:
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
        