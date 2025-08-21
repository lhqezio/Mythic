"""
Configuration file for LLM to TTS Demo
Customize these settings to optimize performance for your environment
"""

# LLM Configuration
LLM_CONFIG = {
    # Model settings
    "model_path": "bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF",
    "model_filename": "mistralai_Voxtral-Mini-3B-2507-Q4_K_S.gguf",
    
    # Performance optimizations
    "n_ctx": 8192,                    # Context window size
    "n_gpu_layers": -1,               # GPU layers (-1 for all, 0 for CPU only)
    "n_batch": 512,                   # Batch size for processing
    "n_threads": None,                 # CPU threads (None = auto-detect)
    "low_vram": True,                  # Enable low VRAM mode
    
    # Generation parameters
    "temperature": 0.7,                # Creativity (0.0-1.0)
    "top_p": 0.9,                     # Nucleus sampling
    "top_k": 40,                      # Top-k sampling
    "repeat_penalty": 1.1,            # Repetition penalty
    "max_tokens": 500,                # Maximum tokens to generate
    
    # Stop sequences
    "stop_sequences": ["\n\n", "###", "END", "STOP"],
    
    # Verbosity
    "verbose": False,
}

# TTS Configuration
TTS_CONFIG = {
    # Voice model
    "voice_name": "en_US-amy-low",
    "voice_path": "runtimes/en_US-amy-low.onnx",
    
    # Audio quality settings
    "sample_rate": 22050,             # Audio sample rate
    "channels": 1,                    # Mono audio
    "bit_depth": 16,                  # 16-bit audio
    
    # TTS parameters
    "length_scale": 1.0,              # Speech speed (0.5-2.0)
    "noise_scale": 0.667,             # Voice quality (0.1-1.0)
    "noise_w": 0.8,                   # Voice stability (0.1-1.0)
    "speaker_id": 0,                  # Speaker ID
    
    # GPU acceleration
    "use_cuda": False,                 # Set to True if CUDA available
}

# Demo Configuration
DEMO_CONFIG = {
    # Output settings
    "output_dir": "output",
    "audio_format": "wav",
    
    # Demo prompts
    "prompts": [
        "Explain the concept of artificial intelligence in simple terms.",
        "Tell me a short story about a robot learning to paint.",
        "Describe the benefits of renewable energy sources.",
        "Give me three tips for improving productivity at work.",
        "Explain how machine learning algorithms work.",
        "What are the key principles of sustainable development?",
        "Describe a futuristic city powered by renewable energy.",
        "Explain the concept of neural networks in simple terms.",
        "Tell me about the importance of biodiversity conservation.",
        "What are the benefits of learning a new language?"
    ],
    
    # Performance monitoring
    "enable_metrics": True,
    "log_level": "INFO",
    
    # Error handling
    "max_retries": 3,
    "retry_delay": 1.0,
}

# Environment Configuration
ENV_CONFIG = {
    # Paths
    "runtime_dir": "runtimes",
    "cache_dir": ".cache",
    
    # Downloads
    "auto_download_models": True,
    "download_timeout": 300,           # 5 minutes
    
    # System resources
    "max_memory_usage": "4GB",        # Maximum memory usage
    "cpu_affinity": None,             # CPU affinity (None = auto)
    
    # Logging
    "log_file": "demo.log",
    "log_rotation": "1 day",
    "log_backup_count": 7,
}

# Performance Profiles
PERFORMANCE_PROFILES = {
    "fast": {
        "description": "Fastest generation, lower quality",
        "llm": {
            "temperature": 0.3,
            "max_tokens": 100,
            "n_batch": 256,
        },
        "tts": {
            "length_scale": 1.2,
            "noise_scale": 0.5,
        }
    },
    
    "balanced": {
        "description": "Balanced speed and quality",
        "llm": {
            "temperature": 0.7,
            "max_tokens": 500,
            "n_batch": 512,
        },
        "tts": {
            "length_scale": 1.0,
            "noise_scale": 0.667,
        }
    },
    
    "quality": {
        "description": "Highest quality, slower generation",
        "llm": {
            "temperature": 0.8,
            "max_tokens": 800,
            "n_batch": 1024,
        },
        "tts": {
            "length_scale": 0.9,
            "noise_scale": 0.8,
        }
    }
}

# Default profile
DEFAULT_PROFILE = "balanced"

# Utility functions
def get_config(profile_name: str = None) -> dict:
    """Get configuration with optional performance profile"""
    config = {
        "llm": LLM_CONFIG.copy(),
        "tts": TTS_CONFIG.copy(),
        "demo": DEMO_CONFIG.copy(),
        "env": ENV_CONFIG.copy(),
    }
    
    if profile_name and profile_name in PERFORMANCE_PROFILES:
        profile = PERFORMANCE_PROFILES[profile_name]
        
        # Apply LLM profile settings
        if "llm" in profile:
            config["llm"].update(profile["llm"])
        
        # Apply TTS profile settings
        if "tts" in profile:
            config["tts"].update(profile["tts"])
    
    return config

def get_profile_names() -> list:
    """Get available performance profile names"""
    return list(PERFORMANCE_PROFILES.keys())

def print_config_summary(config: dict):
    """Print a summary of the current configuration"""
    print("Configuration Summary:")
    print("=" * 50)
    
    print(f"LLM Model: {config['llm']['model_path']}")
    print(f"TTS Voice: {config['tts']['voice_name']}")
    print(f"Output Directory: {config['demo']['output_dir']}")
    print(f"Prompts: {len(config['demo']['prompts'])}")
    
    if config['llm']['n_gpu_layers'] > 0:
        print("GPU Acceleration: Enabled")
    else:
        print("GPU Acceleration: Disabled")
    
    print("=" * 50)
