"""
Whisper manager for hyprwhspr
Handles Whisper model loading and speech-to-text processing
"""

import subprocess
import tempfile
import os
import wave
import json
import time
import urllib.request
import urllib.error
import mimetypes
import uuid
import numpy as np
from pathlib import Path
from typing import Optional
try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager


class WhisperManager:
    """Manages whisper.cpp integration for audio transcription"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager
            
        # Whisper configuration
        self.current_model = self.config.get_setting('model', 'base')
        self.whisper_binary = None
        self.model_path = None
        self.temp_dir = None

        # Server configuration
        self.use_server = bool(self.config.get_setting('use_server', True))
        self.server_url = str(self.config.get_setting('server_url', 'http://127.0.0.1:8080')).rstrip('/')
        self.server_threads = int(self.config.get_setting('server_threads', 4) or 4)
        self.server_model = self.config.get_setting('server_model', None)
        self.server_enabled = False
        
        # Whisper process state
        self.current_process = None
        self.ready = False
        
    def initialize(self) -> bool:
        """Initialize the whisper manager and check dependencies"""
        try:
            # Get paths from config manager
            self.whisper_binary = self.config.get_whisper_binary_path()
            self.temp_dir = self.config.get_temp_directory()
            
            # Check if whisper binary exists
            if not self.whisper_binary.exists():
                print(f"ERROR: Whisper binary not found at: {self.whisper_binary}")
                print("  Please build whisper.cpp first by running the build scripts")
                return False
            
            # Set model path based on current model
            self.model_path = self.config.get_whisper_model_path(self.current_model)
            
            # Check if model exists
            if not self.model_path.exists():
                print(f"ERROR: Whisper model not found at: {self.model_path}")
                print(f"  Please download the {self.current_model} model first")
                return False
            
            print(f"Whisper binary found: {self.whisper_binary}")
            print(f"Using model: {self.current_model} at {self.model_path}")

            self.server_enabled = False
            if self.use_server:
                if self._check_server_health():
                    self.server_enabled = True
                    print(f"Using persistent whisper server at {self.server_url}")
                else:
                    print(f"Whisper server not reachable at {self.server_url}, falling back to CLI per-call mode")
            
            self.ready = True
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if whisper is ready for transcription"""
        return self.ready
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using whisper.cpp
        
        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            
        Returns:
            Transcribed text string
        """
        if not self.ready:
            raise RuntimeError("Whisper manager not initialized")
        
        # Check if we have valid audio data
        if audio_data is None:
            print("No audio data provided to transcribe")
            return ""
        
        if len(audio_data) == 0:
            print("Empty audio data provided to transcribe")
            return ""
        
        # Check if audio is too short (less than 0.1 seconds)
        min_samples = int(sample_rate * 0.1)  # 0.1 seconds minimum
        if len(audio_data) < min_samples:
            print(f"Audio too short: {len(audio_data)} samples (minimum {min_samples})")
            return ""
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.temp_dir) as temp_file:
            temp_wav_path = temp_file.name

        try:
            # Save audio data as WAV file
            self._save_audio_as_wav(audio_data, temp_wav_path, sample_rate)

            if self.server_enabled and self._check_server_health():
                transcription = self._run_server(temp_wav_path)
                if transcription:
                    return transcription.strip()

            transcription = self._run_whisper(temp_wav_path)
            return transcription.strip() if transcription else ""
        finally:
            try:
                os.unlink(temp_wav_path)
            except:
                pass
    
    def _save_audio_as_wav(self, audio_data: np.ndarray, filepath: str, sample_rate: int):
        """Save numpy audio data as a WAV file"""
        # Convert float32 to int16 for WAV format
        if audio_data.dtype == np.float32:
            # Scale from [-1, 1] to [-32768, 32767]
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    def _check_server_health(self) -> bool:
        try:
            url = f"{self.server_url}/health"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def _run_server(self, audio_file_path: str) -> str:
        try:
            url = f"{self.server_url}/inference"
            fields = {
                'language': 'en',
                'threads': str(self.server_threads),
                'prompt': self.config.get_setting(
                    'whisper_prompt',
                    'Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules.'
                ),
            }
            if self.server_model:
                fields['model'] = self.server_model
            else:
                fields['model'] = self.current_model

            boundary = '----hyprwhspr-' + uuid.uuid4().hex
            data_parts = []

            for k, v in fields.items():
                data_parts.append(f'--{boundary}\r\n'.encode())
                data_parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
                data_parts.append(f'{v}\r\n'.encode())

            filename = os.path.basename(audio_file_path)
            mime = mimetypes.guess_type(filename)[0] or 'audio/wav'
            data_parts.append(f'--{boundary}\r\n'.encode())
            data_parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
            data_parts.append(f'Content-Type: {mime}\r\n\r\n'.encode())
            with open(audio_file_path, 'rb') as f:
                data_parts.append(f.read())
            data_parts.append(b'\r\n')
            data_parts.append(f'--{boundary}--\r\n'.encode())

            body = b''.join(data_parts)
            req = urllib.request.Request(url, data=body, method='POST')
            req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
            req.add_header('Content-Length', str(len(body)))

            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get('Content-Type', '')
                data = resp.read()
                if 'application/json' in content_type:
                    try:
                        payload = json.loads(data.decode('utf-8', errors='ignore'))
                        return payload.get('text') or payload.get('transcription') or ''
                    except Exception:
                        return data.decode('utf-8', errors='ignore')
                return data.decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            print(f"Whisper server HTTP error: {e.code}")
            return ""
        except urllib.error.URLError as e:
            print(f"Whisper server URL error: {e.reason}")
            return ""
        except Exception as e:
            print(f"Error calling whisper server: {e}")
            return ""

    
    def _run_whisper(self, audio_file_path: str) -> str:
        """Run whisper.cpp on the given audio file"""
        try:
            # Get whisper prompt from config or use default
            whisper_prompt = self.config.get_setting(
                'whisper_prompt', 
                'Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules.'
            )
            
            # Construct whisper.cpp command
            cmd = [
                str(self.whisper_binary),
                '-m', str(self.model_path),
                '-f', audio_file_path,
                '--output-txt',
                '--language', 'en',
                '--threads', '4',
                '--prompt', whisper_prompt
            ]
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Try to read the output txt file
                txt_file = audio_file_path + '.txt'
                if os.path.exists(txt_file):
                    with open(txt_file, 'r') as f:
                        transcription = f.read().strip()
                    # Clean up the txt file
                    os.unlink(txt_file)
                    return transcription
                else:
                    # Fall back to stdout if no txt file
                    return result.stdout.strip()
            else:
                print(f"Whisper command failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("Whisper transcription timed out")
            return ""
        except Exception as e:
            print(f"Error running whisper: {e}")
            return ""
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the whisper model
        
        Args:
            model_name: Name of the model (e.g., 'base', 'small')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the new model exists
            new_model_path = self.config.get_whisper_model_path(model_name)
            
            if not new_model_path.exists():
                print(f"ERROR: Model {model_name} not found at {new_model_path}")
                return False
            
            # Update current model
            self.current_model = model_name
            self.model_path = new_model_path
            
            # Update config
            self.config.set_setting('model', model_name)
            
            print(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to set model {model_name}: {e}")
            return False
    
    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.current_model
    
    def get_available_models(self) -> list:
        """Get list of available whisper models"""
        models_dir = self.config.get_whisper_model_path('').parent
        available_models = []
        
        # Look for the supported model files
        supported_models = ['tiny', 'base', 'small', 'medium', 'large']
        
        for model in supported_models:
            # Check for both English-only and multilingual versions
            model_files = [
                models_dir / f"ggml-{model}.en.bin",  # English-only
                models_dir / f"ggml-{model}.bin"      # Multilingual
            ]
            
            for model_file in model_files:
                if model_file.exists():
                    # Add model name with suffix if it's English-only
                    if model_file.name.endswith('.en.bin'):
                        model_name = f"{model}.en"
                    else:
                        model_name = model
                    
                    if model_name not in available_models:
                        available_models.append(model_name)
                    break  # Don't add both versions of same model
        
        return sorted(available_models)
