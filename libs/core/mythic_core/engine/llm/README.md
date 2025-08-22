# LLM Runner System

A modular, extensible system for running Large Language Models with support for multiple backends.

## Architecture

The system is organized into several focused modules:

```
llm/
├── __init__.py          # Package initialization and exports
├── base.py              # Abstract base class for all runners
├── config.py            # Configuration classes and constants
├── exceptions.py        # Custom exception hierarchy
├── factory.py           # Factory for creating appropriate runners
├── llama_cpp.py         # llama.cpp backend implementation
├── transformers.py      # Transformers backend implementation
├── example.py           # Usage examples
└── README.md            # This file
```

## Key Components

### 1. BaseLLMRunner (base.py)
Abstract base class that defines the interface all LLM runners must implement:
- Common functionality like `run()`, `generate_text()`, etc.
- Configuration parsing and validation
- State management and error handling

### 2. Configuration (config.py)
- `ModelConfig`: Typed configuration with validation
- `BackendType`: Enum for supported backends (llama.cpp, transformers)
- `DeviceType`: Enum for device types (auto, cpu, cuda, mps)
- Default configuration values

### 3. Exceptions (exceptions.py)
Custom exception hierarchy:
- `LLMError`: Base exception
- `LLMInitializationError`: Model initialization failures
- `LLMInferenceError`: Inference failures

### 4. Factory (factory.py)
`LLMRunnerFactory` creates the appropriate runner based on configuration:
- Automatically selects backend implementation
- Validates configuration
- Provides clean interface for creating runners

### 5. Backend Implementations
- **LlamaCppRunner**: Optimized for llama.cpp models
- **TransformersRunner**: Full-featured PyTorch implementation

## Usage

### Basic Usage

```python
from mythic_core.engine.llm import LLMRunnerFactory

# Dictionary configuration
config = {
    'llm': {
        'backend': 'llama.cpp',
        'model_path': '/path/to/model.gguf',
        'temperature': 0.8,
        'max_tokens': 1000,
    }
}

# Create runner
runner = LLMRunnerFactory.create_runner(config)

# Generate text
response = runner.generate_text("Hello, world!")
```

### Typed Configuration

```python
from mythic_core.engine.llm import ModelConfig, BackendType, DeviceType

config = ModelConfig(
    model_path='/path/to/model',
    backend=BackendType.TRANSFORMERS,
    device=DeviceType.AUTO,
    temperature=0.7,
    max_tokens=500
)

runner = LLMRunnerFactory.create_runner(config)
```

### Streaming Generation

```python
# Streaming text generation
for chunk in runner.generate_text_streaming("Tell me a story:"):
    print(chunk, end='', flush=True)
```

### Chat Completion

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
]

response = runner.create_chat_completion(messages)
print(response['choices'][0]['message']['content'])
```

## Features

### ✅ Supported Backends
- **llama.cpp**: Fast, optimized inference
- **transformers**: Full PyTorch ecosystem support

### ✅ Device Support
- **CPU**: Universal compatibility
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon optimization
- **Auto**: Automatic device selection

### ✅ Advanced Features
- Streaming text generation
- Chat completion (OpenAI-compatible format)
- Model optimization and device management
- Memory usage monitoring
- Configuration validation
- Model reloading
- Comprehensive error handling

## Adding New Backends

To add a new backend:

1. Create a new file (e.g., `new_backend.py`)
2. Inherit from `BaseLLMRunner`
3. Implement all abstract methods
4. Add the backend type to `BackendType` enum
5. Update the factory to handle the new backend

Example:
```python
from .base import BaseLLMRunner

class NewBackendRunner(BaseLLMRunner):
    def _initialize_model(self) -> None:
        # Initialize your backend
        pass
    
    def _run_streaming(self, prompt: str, max_tokens: int, temperature: float):
        # Implement streaming
        pass
    
    # ... implement other abstract methods
```

## Backward Compatibility

The original `llm_runner.py` file maintains backward compatibility by importing from the new modular structure. Existing code should continue to work without changes.

## Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a single responsibility
2. **Maintainability**: Easier to modify and debug specific components
3. **Extensibility**: Simple to add new backends and features
4. **Testing**: Each component can be tested independently
5. **Code Reuse**: Common functionality shared through base class
6. **Type Safety**: Better IDE support and error catching
7. **Documentation**: Clear structure makes code easier to understand

## Error Handling

The system provides comprehensive error handling:
- Configuration validation at startup
- Graceful fallbacks for missing dependencies
- Detailed error messages for debugging
- Custom exception types for different failure modes

## Performance Features

- **llama.cpp**: Optimized C++ implementation
- **transformers**: PyTorch optimizations (torch.compile, half precision)
- **Device Management**: Automatic GPU/CPU selection
- **Memory Optimization**: Efficient memory usage and monitoring
