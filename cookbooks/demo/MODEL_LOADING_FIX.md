# Fix for GGUF Model Loading Error

## Problem
The error `ValueError: Failed to load model from file: ..././mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf` is caused by a bug in llama-cpp-python version 0.3.16 where the `from_pretrained()` method incorrectly constructs file paths with double dots (`./`).

## Solution 1: Update llama-cpp-python (Recommended)
Run the updated CUDA fix script:
```bash
./fix_cuda_llama_cpp.sh
```

This will update to the latest version which should fix the path issue.

## Solution 2: Manual Download and Direct Loading
If the update doesn't work, use this alternative approach:

### Step 1: Download the model manually
```bash
# Install huggingface-hub if you haven't already
pip install huggingface-hub

# Download the model to a local directory
huggingface-cli download bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf --local-dir ./models
```

### Step 2: Load the model using direct path
```python
from llama_cpp import Llama

# Load using direct path instead of from_pretrained
llm = Llama(
    model_path="./models/mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,  # Use all GPU layers
    verbose=False,
)

# Test the model
out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Give me two sentences about Voxtral Mini."}],
    max_tokens=200,
    temperature=0.2,
    top_p=0.95,
)
print(out["choices"][0]["message"]["content"])
```

## Solution 3: Use the Automated Fix Script
Run the provided Python script:
```bash
python load_model_fix.py
```

This script will:
1. Download the model automatically
2. Load it using the direct path method
3. Test it with a simple prompt

## Why This Happens
The `from_pretrained()` method in llama-cpp-python 0.3.16 has a bug where it incorrectly joins paths, resulting in malformed file paths like:
```
/path/to/model/./filename.gguf
```

The double dot (`./`) causes the file loading to fail.

## Prevention
- Always use the latest version of llama-cpp-python
- Consider using direct path loading for critical applications
- Test model loading in a separate script before running your main application

## Alternative Models
If you continue having issues with this specific model, you can try other GGUF models:
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- `TheBloke/Llama-2-7B-Chat-GGUF`
- `TheBloke/CodeLlama-7B-Python-GGUF`
