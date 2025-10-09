import os
import time
import uuid
import json
import urllib.request
import urllib.error
import mimetypes
import subprocess
from pathlib import Path
from typing import Optional

class WhisperServer:
    def __init__(self, whisper_binary: Path, model_path: Path, threads: int = 0):
        self.whisper_binary = whisper_binary
        self.model_path = model_path
        self.threads = threads
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.url: Optional[str] = None

    def _pick_free_port(self) -> int:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        _, port = s.getsockname()
        s.close()
        return port

    def start(self) -> bool:
        try:
            port = self._pick_free_port()
            threads = self.threads or (os.cpu_count() or 4)
            binary = str(self.whisper_binary).replace('main', 'server') if str(self.whisper_binary).endswith('main') else str(self.whisper_binary)
            args = [binary, '-m', str(self.model_path), '-p', str(port), '-t', str(threads)]
            self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.port = port
            self.url = f"http://127.0.0.1:{port}"
            for _ in range(10):
                if self.healthy():
                    return True
                time.sleep(0.2)
            return False
        except Exception:
            self.proc = None
            return False

    def ensure(self) -> bool:
        try:
            if self.proc is not None and self.proc.poll() is None:
                return self.healthy()
            return self.start()
        except Exception:
            return False

    def stop(self):
        try:
            if self.proc is None:
                return
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
            self.proc = None
        except Exception:
            self.proc = None

    def healthy(self) -> bool:
        try:
            if not self.url:
                return False
            req = urllib.request.Request(f"{self.url}/health", method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def transcribe(self, audio_file_path: str, model_name: str, prompt: str, threads_override: int = 0) -> str:
        try:
            if not self.url:
                return ""
            threads = threads_override or self.threads or (os.cpu_count() or 4)
            url = f"{self.url}/inference"
            fields = {
                'language': 'en',
                'threads': str(threads),
                'prompt': prompt,
                'model': model_name,
            }
            boundary = '----hyprwhspr-' + uuid.uuid4().hex
            parts = []
            for k, v in fields.items():
                parts.append(f'--{boundary}\r\n'.encode())
                parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
                parts.append(f'{v}\r\n'.encode())
            filename = os.path.basename(audio_file_path)
            mime = mimetypes.guess_type(filename)[0] or 'audio/wav'
            parts.append(f'--{boundary}\r\n'.encode())
            parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
            parts.append(f'Content-Type: {mime}\r\n\r\n'.encode())
            with open(audio_file_path, 'rb') as f:
                parts.append(f.read())
            parts.append(b'\r\n')
            parts.append(f'--{boundary}--\r\n'.encode())

            body = b''.join(parts)
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
        except urllib.error.HTTPError:
            return ""
        except urllib.error.URLError:
            return ""
        except Exception:
            return ""
