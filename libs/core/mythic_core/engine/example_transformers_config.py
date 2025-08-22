"""
Example configuration for using the transformers backend with LLMRunner
"""

# Example configuration for transformers backend
TRANSFORMERS_CONFIG = {
    'llm': {
        'backend': 'transformers',
        'model_path': 'microsoft/DialoGPT-medium',  # Example model
        'device': 'auto',  # Will auto-detect CUDA/CPU
        
        # Generation parameters
        'max_tokens': 1000,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'stop_sequences': ["\n\n", "###", "END", "STOP"],
        
        # Model loading parameters
        'torch_dtype': 'auto',  # Will use float16 for CUDA, float32 for CPU
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
    },
    
    'logging': {
        'level': 'INFO',
        'enable_colors': True,
        'log_dir': 'logs',
    }
}

# Alternative configuration for larger models
LARGE_MODEL_CONFIG = {
    'llm': {
        'backend': 'transformers',
        'model_path': 'gpt2-large',  # Larger model
        'device': 'auto',
        
        # Generation parameters
        'max_tokens': 500,
        'temperature': 0.8,
        'top_p': 0.95,
        'top_k': 40,
        
        # Memory optimization
        'low_cpu_mem_usage': True,
        'torch_dtype': 'auto',
    },
    
    'logging': {
        'level': 'DEBUG',
        'enable_colors': True,
    }
}

# Configuration for chat models
CHAT_MODEL_CONFIG = {
    'llm': {
        'backend': 'transformers',
        'model_path': 'microsoft/DialoGPT-large',  # Chat-optimized model
        'device': 'auto',
        
        # Chat-specific parameters
        'max_tokens': 150,
        'temperature': 0.6,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        
        # Model parameters
        'trust_remote_code': True,
        'low_cpu_mem_usage': True,
    },
    
    'logging': {
        'level': 'INFO',
        'enable_colors': True,
    }
}

# Example usage:
"""
from mythic_core.engine.llm_runner import LLMRunner
from mythic_core.engine.example_transformers_config import TRANSFORMERS_CONFIG

# Initialize the runner
runner = LLMRunner(TRANSFORMERS_CONFIG)

# Basic text generation
response = runner.generate_text("Hello, how are you?")
print(response)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is artificial intelligence?"}
]

chat_response = runner.create_chat_completion(messages)
print(chat_response['choices'][0]['message']['content'])

# Get model information
model_info = runner.get_model_info()
print(f"Model: {model_info['model_type']}")
print(f"Parameters: {model_info['total_parameters']:,}")

# Optimize for inference
runner.optimize_for_inference()

# Get memory usage
memory_info = runner.get_memory_usage()
print(f"GPU Memory: {memory_info['gpu_memory_gb']}")
"""

