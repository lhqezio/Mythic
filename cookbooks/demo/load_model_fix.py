#!/usr/bin/env python3
"""
Alternative model loading script to bypass the llama-cpp-python from_pretrained bug.
This script downloads the model manually and loads it using the direct path method.
"""

import os
import subprocess
import sys
from pathlib import Path

def download_model():
    """Download the model using huggingface-cli"""
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return str(model_path)
    
    print("Downloading model using huggingface-cli...")
    try:
        cmd = [
            "huggingface-cli", "download",
            "bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF",
            "mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf",
            "--local-dir", str(model_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Model downloaded successfully!")
        return str(model_path)
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("huggingface-cli not found. Please install it first:")
        print("pip install huggingface-hub")
        return None

def load_model_direct(model_path):
    """Load the model using direct path method"""
    try:
        from llama_cpp import Llama
        
        print(f"Loading model from: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,  # Use all GPU layers
            verbose=False,
        )
        
        print("Model loaded successfully!")
        return llm
        
    except ImportError:
        print("llama-cpp-python not found. Please install it first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_model(llm):
    """Test the loaded model with a simple prompt"""
    if llm is None:
        return
    
    try:
        print("\nTesting model with a simple prompt...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Give me two sentences about Voxtral Mini."}],
            max_tokens=200,
            temperature=0.2,
            top_p=0.95,
        )
        
        content = response["choices"][0]["message"]["content"]
        print(f"Model response: {content}")
        
    except Exception as e:
        print(f"Error testing model: {e}")

def main():
    print("=== GGUF Model Loading Fix ===")
    print("This script bypasses the llama-cpp-python from_pretrained bug\n")
    
    # Download model if needed
    model_path = download_model()
    if not model_path:
        print("Failed to download model. Exiting.")
        sys.exit(1)
    
    # Load model using direct path
    llm = load_model_direct(model_path)
    if not llm:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Test the model
    test_model(llm)
    
    print("\n=== Model loading successful! ===")
    print("You can now use this model in your code with:")
    print(f"llm = Llama(model_path='{model_path}', n_gpu_layers=-1)")

if __name__ == "__main__":
    main()
