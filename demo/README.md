# Mythic AI Engine - LLM to TTS Demo

This directory contains comprehensive demo scripts that demonstrate the integration of Large Language Models (LLMs) with Text-to-Speech (TTS) systems using the Mythic AI Engine.

## 🚀 Features

### Core Capabilities
- **LLM Integration**: Uses Voxtral Mini 3B model for text generation
- **High-Quality TTS**: Piper TTS engine with optimized voice models
- **Performance Optimization**: Multiple performance profiles for different use cases
- **Streaming Support**: Real-time text and audio generation
- **Error Handling**: Robust error handling with retry mechanisms
- **Performance Monitoring**: Comprehensive metrics and benchmarking

### Performance Profiles
- **Fast**: Optimized for speed, lower quality
- **Balanced**: Balanced speed and quality (default)
- **Quality**: Optimized for quality, slower generation

## 📁 Files

- `llm_to_tts_demo_enhanced.py` - Enhanced demo with advanced features
- `config.py` - Configuration management
- `run_demo.sh` - Enhanced demo launcher script
- `demo_setup.sh` - Environment setup script
- `requirements.txt` - Python dependencies

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Linux/macOS/Windows (WSL2 supported)
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, for acceleration)

### Quick Start
1. **Setup Environment**:
   ```bash
   cd demo
   chmod +x run_demo.sh demo_setup.sh
   ./demo_setup.sh
   ```

2. **Run Basic Demo**:
   ```bash
   ./run_demo.sh basic
   ```

3. **Run Enhanced Demo**:
   ```bash
   ./run_demo.sh enhanced
   ```

## 🎯 Usage

### Command Line Options

#### Enhanced Demo (Direct Python)
```bash
# Run with default balanced profile
python llm_to_tts_demo_enhanced.py

# Run with specific profile
python llm_to_tts_demo_enhanced.py --profile fast
python llm_to_tts_demo_enhanced.py --profile quality

# Interactive mode
python llm_to_tts_demo_enhanced.py --interactive

# Custom output directory
python llm_to_tts_demo_enhanced.py --output-dir /path/to/output

# Custom configuration file
python llm_to_tts_demo_enhanced.py --custom-config my_config.json
```

#### Launcher Script
```bash
# Run enhanced demo (balanced profile - default)
./run_demo.sh

# Run with specific profile
./run_demo.sh fast
./run_demo.sh balanced
./run_demo.sh quality

# Interactive mode
./run_demo.sh interactive

# Show help
./run_demo.sh help
```

### Performance Profiles

#### Fast Profile
- **Use Case**: Quick demos, testing, low-resource environments
- **Settings**: Lower temperature, fewer tokens, smaller batch size
- **Trade-off**: Faster generation, lower quality

#### Balanced Profile (Default)
- **Use Case**: General purpose, production use
- **Settings**: Moderate temperature, balanced parameters
- **Trade-off**: Good balance of speed and quality

#### Quality Profile
- **Use Case**: High-quality output, presentations, final content
- **Settings**: Higher temperature, more tokens, larger batch size
- **Trade-off**: Higher quality, slower generation

## ⚙️ Configuration

### Customizing Settings

Edit `config.py` to modify:
- LLM model parameters
- TTS voice settings
- Performance profiles
- Demo prompts
- System resource limits

### Custom Configuration File

Create a JSON file with your settings:

```json
{
  "llm": {
    "temperature": 0.8,
    "max_tokens": 150
  },
  "tts": {
    "length_scale": 1.2,
    "noise_scale": 0.5
  },
  "demo": {
    "output_dir": "my_output"
  }
}
```

Use it with:
```bash
python llm_to_tts_demo_enhanced.py --custom-config my_config.json
```

## 📊 Performance Monitoring

The demo scripts provide comprehensive performance metrics:

- **LLM Inference Time**: Text generation performance
- **TTS Generation Time**: Audio synthesis performance
- **Total Processing Time**: End-to-end performance
- **Success Rate**: Reliability metrics
- **Memory Usage**: Resource utilization

### Viewing Metrics

#### During Interactive Mode
```
Type 'metrics' to view performance metrics
Type 'status' to view current status
```

#### In Logs
Check `demo.log` for detailed performance information.

## 🔧 Troubleshooting

### Common Issues

#### Memory Issues
- Use the "fast" profile
- Reduce `n_ctx` in LLM config
- Enable `low_vram` mode

#### GPU Issues
- Set `n_gpu_layers: 0` for CPU-only mode
- Check CUDA installation
- Verify GPU memory availability

#### TTS Voice Issues
- Ensure voice model is downloaded
- Check file permissions
- Verify ONNX and JSON files exist

### Debug Mode

Enable verbose logging:
```python
# In config.py
LLM_CONFIG["verbose"] = True
```

### Error Recovery

The enhanced demo includes:
- Automatic retries with exponential backoff
- Graceful degradation
- Comprehensive error logging
- Resource cleanup

## 🎨 Customization Examples

### Adding New Prompts
```python
# In config.py
DEMO_CONFIG["prompts"].extend([
    "Explain quantum computing in simple terms.",
    "Describe the future of renewable energy."
])
```

### Custom Voice Models
```python
# In config.py
TTS_CONFIG["voice_name"] = "en_US-joe-low"
TTS_CONFIG["voice_path"] = "runtimes/en_US-joe-low.onnx"
```

### Performance Tuning
```python
# In config.py
PERFORMANCE_PROFILES["ultra_fast"] = {
    "description": "Ultra-fast generation",
    "llm": {
        "temperature": 0.1,
        "max_tokens": 50,
        "n_batch": 128
    },
    "tts": {
        "length_scale": 1.5,
        "noise_scale": 0.3
    }
}
```

## 📈 Best Practices

### For Production Use
1. **Use Balanced Profile**: Good balance of speed and quality
2. **Monitor Resources**: Watch memory and GPU usage
3. **Error Handling**: Implement proper error handling
4. **Logging**: Enable comprehensive logging
5. **Testing**: Test with various input types

### For Development
1. **Use Fast Profile**: Quick iteration cycles
2. **Interactive Mode**: Real-time testing
3. **Custom Configs**: Easy parameter experimentation
4. **Metrics**: Performance optimization

### For Demos
1. **Quality Profile**: Best output quality
2. **Prepared Prompts**: Test with known inputs
3. **Audio Quality**: Ensure good audio output
4. **Performance**: Monitor generation times

## 🤝 Contributing

To contribute to the demo scripts:

1. Follow the existing code style
2. Add comprehensive error handling
3. Include performance metrics
4. Update configuration options
5. Test with different environments

## 📄 License

This demo is part of the Mythic AI Engine project. See the main project LICENSE for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `demo.log`
3. Verify your configuration
4. Check system requirements

---

**Happy Demo-ing! 🎉**
