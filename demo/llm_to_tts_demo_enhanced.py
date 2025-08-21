#!/usr/bin/env python3
"""
Mythic AI Engine - Enhanced LLM to TTS Demo Script
Advanced demo with configuration management, performance profiles, and streaming capabilities

Features:
- Configuration-driven setup
- Multiple performance profiles (fast, balanced, quality)
- Streaming audio generation
- Advanced error handling and retries
- Performance benchmarking
- Interactive mode
"""

import os
import sys
import time
import wave
import asyncio
import threading
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
import logging
from datetime import datetime

# Import configuration
try:
    from config import get_config, get_profile_names, print_config_summary
except ImportError:
    print("Error: config.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from piper import PiperVoice
    from llama_cpp import Llama
    import numpy as np
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please run: pip install piper-tts llama-cpp-python numpy")
    sys.exit(1)


class EnhancedLLMToTTSDemo:
    """Enhanced demo class with advanced features"""
    
    def __init__(self, profile: str = "balanced", custom_config: dict = None):
        """
        Initialize the enhanced demo
        
        Args:
            profile: Performance profile name
            custom_config: Custom configuration overrides
        """
        self.profile = profile
        self.config = get_config(profile)
        
        # Apply custom configuration overrides
        if custom_config:
            self._merge_config(custom_config)
        
        # Initialize paths
        self.output_dir = Path(self.config['demo']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.llm: Optional[Llama] = None
        self.tts_voice: Optional[PiperVoice] = None
        
        # Performance metrics
        self.metrics = {
            'llm_inference_time': [],
            'tts_generation_time': [],
            'total_processing_time': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        
        # Status tracking
        self.is_initialized = False
        self.current_prompt_index = 0
        
        logger.info(f"Initialized demo with '{profile}' profile")
        print_config_summary(self.config)
    
    def _merge_config(self, custom_config: dict):
        """Merge custom configuration with default config"""
        for section, settings in custom_config.items():
            if section in self.config:
                self.config[section].update(settings)
            else:
                self.config[section] = settings
    
    def setup_environment(self) -> bool:
        """Setup the environment with enhanced error handling"""
        try:
            logger.info("Setting up environment...")
            
            # Create necessary directories
            for dir_path in [self.output_dir, Path(self.config['env']['runtime_dir'])]:
                dir_path.mkdir(exist_ok=True)
            
            # Check and download voice model if needed
            if self.config['env']['auto_download_models']:
                if not self._check_voice_model():
                    self._download_voice_model()
            
            # Check system resources
            self._check_system_resources()
            
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _check_voice_model(self) -> bool:
        """Check if voice model exists and is valid"""
        voice_path = Path(self.config['tts']['voice_path'])
        
        # Check for different possible config file extensions
        possible_configs = [
            voice_path.with_suffix('.onnx.json'),  # .onnx.json extension
            voice_path.with_suffix('.json'),       # .json extension
            voice_path.with_suffix('').with_suffix('.json')  # append .json
        ]
        
        config_exists = any(config.exists() for config in possible_configs)
        
        return voice_path.exists() and config_exists
    
    def _download_voice_model(self):
        """Download TTS voice model with progress tracking"""
        try:
            import subprocess
            from tqdm import tqdm
            
            logger.info("Downloading TTS voice model...")
            
            cmd = [
                "python3", "-m", "piper.download_voices", 
                self.config['tts']['voice_name']
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.config['env']['runtime_dir'],
                timeout=self.config['env']['download_timeout']
            )
            
            if result.returncode == 0:
                logger.info("Voice model downloaded successfully")
            else:
                logger.error(f"Voice download failed: {result.stderr}")
                raise RuntimeError("Voice model download failed")
                
        except subprocess.TimeoutExpired:
            logger.error("Voice model download timed out")
            raise
        except Exception as e:
            logger.error(f"Voice download error: {e}")
            raise
    
    def _check_system_resources(self):
        """Check system resources and provide recommendations"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
            
            if memory.available < 2 * (1024**3):  # Less than 2GB
                logger.warning("Low memory detected. Consider using 'fast' profile.")
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            logger.info(f"CPU cores: {cpu_count}")
            
            # Check GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    logger.info(f"GPU memory: {gpu_memory / (1024**3):.1f} GB")
                    
                    if gpu_memory < 4 * (1024**3):  # Less than 4GB
                        logger.warning("Low GPU memory. Consider using CPU-only mode.")
                        self.config['llm']['n_gpu_layers'] = 0
                else:
                    logger.info("CUDA not available, using CPU mode")
                    self.config['llm']['n_gpu_layers'] = 0
            except ImportError:
                logger.info("PyTorch not available, GPU detection skipped")
                
        except ImportError:
            logger.warning("psutil not available, system resource check skipped")
    
    def initialize_llm(self) -> bool:
        """Initialize LLM with enhanced error handling and retries"""
        max_retries = self.config['demo']['max_retries']
        retry_delay = self.config['demo']['retry_delay']
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing LLM (attempt {attempt + 1}/{max_retries})...")
                
                # Set CPU threads if not specified
                if self.config['llm']['n_threads'] is None:
                    self.config['llm']['n_threads'] = os.cpu_count()
                
                self.llm = Llama.from_pretrained(
                    repo_id=self.config['llm']['model_path'],
                    filename=self.config['llm']['model_filename'],
                    **{k: v for k, v in self.config['llm'].items() 
                       if k not in ['model_path', 'model_filename']}
                )
                
                logger.info("LLM initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"LLM initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All LLM initialization attempts failed")
                    return False
        
        return False
    
    def initialize_tts(self) -> bool:
        """Initialize TTS with enhanced error handling"""
        try:
            logger.info("Initializing TTS...")
            
            voice_path = self.config['tts']['voice_path']
            
            # Check if voice model and config exist
            if not os.path.exists(voice_path):
                raise FileNotFoundError(f"Voice model not found: {voice_path}")
            
            # Try different config file extensions
            config_path = None
            possible_configs = [
                voice_path.replace('.onnx', '.onnx.json'),  # .onnx.json extension
                voice_path.replace('.onnx', '.json'),       # .json extension
                voice_path + '.json'                        # append .json
            ]
            
            for config in possible_configs:
                if os.path.exists(config):
                    config_path = config
                    break
            
            if not config_path:
                raise FileNotFoundError(f"TTS config file not found. Tried: {possible_configs}")
            
            logger.info(f"Using TTS config: {config_path}")
            
            # Load TTS voice with optimized settings
            self.tts_voice = PiperVoice.load(
                voice_path,
                config_path=config_path,
                use_cuda=self.config['tts']['use_cuda']
            )
            
            logger.info("TTS initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return False
    
    def generate_text_streaming(self, prompt: str, max_tokens: int = None) -> Generator[str, None, None]:
        """
        Generate text using streaming for real-time output
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (uses config default if None)
            
        Yields:
            Generated text chunks
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        if max_tokens is None:
            max_tokens = self.config['llm']['max_tokens']
        
        start_time = time.time()
        
        try:
            # Streaming chat completion
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.config['llm']['temperature'],
                top_p=self.config['llm']['top_p'],
                top_k=self.config['llm']['top_k'],
                repeat_penalty=self.config['llm']['repeat_penalty'],
                stop=self.config['llm']['stop_sequences'],
                stream=True,  # Enable streaming
            )
            
            full_text = ""
            for chunk in response:
                if chunk and 'choices' in chunk and chunk['choices']:
                    choice = chunk['choices'][0]
                    # Check if this is the final chunk
                    if choice.get('finish_reason') is not None:
                        # This is the final chunk, ensure we get any remaining content
                        content = choice.get('delta', {}).get('content', '')
                        if content:
                            full_text += content
                            yield content
                        break
                    else:
                        # Regular content chunk
                        content = choice.get('delta', {}).get('content', '')
                        if content:
                            full_text += content
                            yield content
            
            # Record metrics
            inference_time = time.time() - start_time
            self.metrics['llm_inference_time'].append(inference_time)
            
            logger.info(f"Streaming text generation completed in {inference_time:.2f}s")
            logger.info(f"Final generated text length: {len(full_text)} characters")
            
            # Check if text was truncated
            if len(full_text) < max_tokens * 2:  # Rough estimate: 2 chars per token
                logger.info(f"Text generation completed. Final text: {full_text[-100:]}...")
            else:
                logger.info(f"Text generation may have been truncated at {max_tokens} tokens")
            
        except Exception as e:
            logger.error(f"Streaming text generation failed: {e}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text using the LLM (non-streaming version)"""
        if max_tokens is None:
            max_tokens = self.config['llm']['max_tokens']
        
        # Collect streaming output
        text_chunks = list(self.generate_text_streaming(prompt, max_tokens))
        final_text = ''.join(text_chunks)
        
        # Log the final text for debugging
        logger.debug(f"Generated text length: {len(final_text)} characters")
        logger.debug(f"Generated text preview: {final_text[:200]}...")
        
        return final_text
    
    def text_to_speech_streaming(self, text: str):
        """Convert text to speech with streaming audio chunks"""
        if not self.tts_voice:
            raise RuntimeError("TTS not initialized")
        
        start_time = time.time()
        
        try:
            # Clean text for TTS
            cleaned_text = self._prepare_text_for_tts(text)
            
            # Create synthesis configuration
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                length_scale=self.config['tts']['length_scale'],
                noise_scale=self.config['tts']['noise_scale'],
                noise_w_scale=self.config['tts']['noise_w'],
            )
            
            # Generate audio
            audio_chunks = self.tts_voice.synthesize(cleaned_text, syn_config)
            
            # Split audio into chunks for streaming effect
            chunk_size = int(self.config['tts']['sample_rate'] * 0.1)  # 100ms chunks
            for chunk in audio_chunks:
                audio_data = chunk.audio_int16_array
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]
            
            # Record metrics
            tts_time = time.time() - start_time
            self.metrics['tts_generation_time'].append(tts_time)
            
            logger.info(f"Streaming TTS generation completed in {tts_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Streaming TTS generation failed: {e}")
            raise
    
    def text_to_speech(self, text: str, output_filename: str) -> str:
        """Convert text to speech and save to file"""
        if not self.tts_voice:
            raise RuntimeError("TTS not initialized")
        
        start_time = time.time()
        
        try:
            # Clean text for TTS
            cleaned_text = self._prepare_text_for_tts(text)
            
            # Create synthesis configuration
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                length_scale=self.config['tts']['length_scale'],
                noise_scale=self.config['tts']['noise_scale'],
                noise_w_scale=self.config['tts']['noise_w'],
            )
            
            # Generate audio
            audio_chunks = self.tts_voice.synthesize(cleaned_text, syn_config)
            
            # Convert chunks to single audio array
            audio_data = []
            for chunk in audio_chunks:
                audio_data.extend(chunk.audio_int16_array)
            audio_data = np.array(audio_data)
            
            # Save audio to file
            output_path = self.output_dir / output_filename
            self._save_audio(audio_data, output_path)
            
            # Record metrics
            tts_time = time.time() - start_time
            self.metrics['tts_generation_time'].append(tts_time)
            
            logger.info(f"TTS generation completed in {tts_time:.2f}s")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for optimal TTS generation"""
        import re
        
        # Remove markdown and special characters
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'```[\s\S]*?```', '', text)            # Code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)              # Inline code
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)        # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)            # Italic
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1 \2', text)
        text = text.strip()
        
        return text
    
    def _save_audio(self, audio_data: np.ndarray, output_path: Path):
        """Save audio data to WAV file with enhanced error handling"""
        try:
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(self.config['tts']['channels'])
                wav_file.setsampwidth(self.config['tts']['bit_depth'] // 8)
                wav_file.setframerate(self.config['tts']['sample_rate'])
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"Audio saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def run_interactive_demo(self):
        """Run interactive demo mode"""
        print("\n" + "="*60)
        print("INTERACTIVE LLM TO TTS DEMO")
        print("="*60)
        print("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\nEnter your prompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                elif user_input.lower() == 'metrics':
                    self._show_metrics()
                    continue
                elif not user_input:
                    continue
                
                # Process the prompt
                print(f"\nGenerating text for: {user_input[:50]}...")
                
                # Generate text
                generated_text = self.generate_text(user_input)
                print(f"Generated text: {generated_text}")
                print(f"Text length: {len(generated_text)} characters")
                
                # Convert to speech
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"interactive_{timestamp}.wav"
                audio_path = self.text_to_speech(generated_text, output_filename)
                
                print(f"Audio saved to: {audio_path}")
                
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
                break
            except Exception as e:
                logger.error(f"Interactive demo error: {e}")
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show available commands"""
        print("\nAvailable commands:")
        print("  help     - Show this help")
        print("  status   - Show current status")
        print("  metrics  - Show performance metrics")
        print("  quit     - Exit the demo")
        print("\nOr just type your prompt to generate text and speech!")
    
    def _show_status(self):
        """Show current demo status"""
        print(f"\nDemo Status:")
        print(f"  Profile: {self.profile}")
        print(f"  LLM Initialized: {self.llm is not None}")
        print(f"  TTS Initialized: {self.tts_voice is not None}")
        print(f"  Prompts Processed: {len(self.metrics['total_processing_time'])}")
        print(f"  Output Directory: {self.output_dir}")
    
    def _show_metrics(self):
        """Show performance metrics"""
        if not self.metrics['total_processing_time']:
            print("\nNo metrics available yet. Run some prompts first!")
            return
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Prompts: {len(self.metrics['total_processing_time'])}")
        print(f"  Average LLM Time: {np.mean(self.metrics['llm_inference_time']):.2f}s")
        print(f"  Average TTS Time: {np.mean(self.metrics['tts_generation_time']):.2f}s")
        print(f"  Average Total Time: {np.mean(self.metrics['total_processing_time']):.2f}s")
        print(f"  Total Demo Time: {sum(self.metrics['total_processing_time']):.2f}s")
    
    def run_demo(self, prompts: List[str] = None) -> Dict[str, Any]:
        """Run the complete demo with enhanced features"""
        if prompts is None:
            prompts = self.config['demo']['prompts']
        
        logger.info(f"Starting enhanced LLM to TTS demo with '{self.profile}' profile...")
        
        results = []
        total_start_time = time.time()
        
        try:
            # Setup and initialize
            if not self.setup_environment():
                raise RuntimeError("Environment setup failed")
            
            if not self.initialize_llm():
                raise RuntimeError("LLM initialization failed")
            
            if not self.initialize_tts():
                raise RuntimeError("TTS initialization failed")
            
            self.is_initialized = True
            
            # Process prompts
            for i, prompt in enumerate(prompts):
                logger.info(f"\n--- Processing Prompt {i+1}/{len(prompts)} ---")
                logger.info(f"Prompt: {prompt[:100]}...")
                
                prompt_start_time = time.time()
                
                try:
                    # Generate text
                    generated_text = self.generate_text(prompt)
                    logger.info(f"Generated text: {generated_text[:100]}...")
                    
                    # Convert to speech
                    output_filename = f"demo_output_{i+1}.wav"
                    audio_path = self.text_to_speech(generated_text, output_filename)
                    
                    # Record metrics
                    prompt_time = time.time() - prompt_start_time
                    self.metrics['total_processing_time'].append(prompt_time)
                    
                    results.append({
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'audio_path': audio_path,
                        'processing_time': prompt_time,
                        'profile': self.profile
                    })
                    
                    logger.info(f"Prompt {i+1} completed in {prompt_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to process prompt {i+1}: {e}")
                    results.append({
                        'prompt': prompt,
                        'error': str(e),
                        'profile': self.profile
                    })
            
            # Generate summary
            demo_summary = self._generate_demo_summary(total_start_time, results)
            
            return demo_summary
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def _generate_demo_summary(self, total_start_time: float, results: List[dict]) -> Dict[str, Any]:
        """Generate comprehensive demo summary"""
        total_time = time.time() - total_start_time
        
        # Calculate metrics
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if successful_results:
            avg_llm_time = np.mean(self.metrics['llm_inference_time'])
            avg_tts_time = np.mean(self.metrics['tts_generation_time'])
            avg_total_time = np.mean(self.metrics['total_processing_time'])
        else:
            avg_llm_time = avg_tts_time = avg_total_time = 0
        
        summary = {
            'profile': self.profile,
            'total_demo_time': total_time,
            'prompts_total': len(results),
            'prompts_successful': len(successful_results),
            'prompts_failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'average_llm_time': avg_llm_time,
            'average_tts_time': avg_tts_time,
            'average_total_time': avg_total_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log summary
        logger.info("\n--- Demo Summary ---")
        logger.info(f"Profile: {self.profile}")
        logger.info(f"Total demo time: {total_time:.2f}s")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Average LLM time: {avg_llm_time:.2f}s")
        logger.info(f"Average TTS time: {avg_tts_time:.2f}s")
        
        return summary
    
    def cleanup(self):
        """Clean up resources with enhanced error handling"""
        try:
            if self.llm:
                del self.llm
                self.llm = None
            
            if self.tts_voice:
                del self.tts_voice
                self.tts_voice = None
            
            self.is_initialized = False
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main demo execution with command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced LLM to TTS Demo")
    parser.add_argument(
        "--profile", 
        choices=get_profile_names(),
        default="balanced",
        help="Performance profile to use"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--custom-config", 
        type=str,
        help="Path to custom configuration JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Custom output directory"
    )
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    custom_config = {}
    if args.custom_config:
        try:
            with open(args.custom_config, 'r') as f:
                custom_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load custom config: {e}")
            sys.exit(1)
    
    if args.output_dir:
        custom_config['demo'] = custom_config.get('demo', {})
        custom_config['demo']['output_dir'] = args.output_dir
    
    # Initialize demo
    demo = EnhancedLLMToTTSDemo(args.profile, custom_config)
    
    try:
        if args.interactive:
            # Run interactive mode
            demo.run_demo()  # Initialize components
            demo.run_interactive_demo()
        else:
            # Run standard demo
            results = demo.run_demo()
            
            # Display results
            print("\n" + "="*60)
            print("ENHANCED LLM TO TTS DEMO COMPLETED!")
            print("="*60)
            
            print(f"Profile: {results['profile']}")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Total Time: {results['total_demo_time']:.2f}s")
            
            for i, result in enumerate(results['results']):
                if 'error' in result:
                    print(f"\nPrompt {i+1}: ERROR - {result['error']}")
                else:
                    print(f"\nPrompt {i+1}:")
                    print(f"  Text: {result['generated_text'][:80]}...")
                    print(f"  Audio: {result['audio_path']}")
                    print(f"  Time: {result['processing_time']:.2f}s")
            
            print(f"\nAudio files saved in: {demo.output_dir}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        demo.cleanup()


if __name__ == "__main__":
    main()
