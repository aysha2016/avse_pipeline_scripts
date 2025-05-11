import torch
import torchaudio
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    n_mels: int = 80
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class AudioProcessor:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize STFT/ISTFT transforms
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=None,
            return_complex=True
        ).to(self.device)
        
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        ).to(self.device)
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels
        ).to(self.device)
        
        # Buffer for overlapping windows
        self.buffer = torch.zeros(self.config.n_fft, device=self.device)
        self.buffer_size = self.config.n_fft

    def process_chunk(self, audio_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a chunk of audio data in real-time
        
        Args:
            audio_chunk: Raw audio tensor of shape [chunk_size]
            
        Returns:
            Tuple of (processed_features, enhanced_audio)
        """
        # Ensure input is on correct device and has correct shape
        audio_chunk = audio_chunk.to(self.device)
        if len(audio_chunk.shape) == 1:
            audio_chunk = audio_chunk.unsqueeze(0)
            
        # Update buffer with overlap
        self.buffer = torch.cat([self.buffer[self.config.hop_length:], audio_chunk])
        
        # Compute STFT
        stft_features = self.stft(self.buffer)
        magnitude = torch.abs(stft_features)
        phase = torch.angle(stft_features)
        
        # Compute mel spectrogram features
        mel_features = self.mel_transform(self.buffer)
        
        # Apply noise reduction (placeholder for actual noise reduction model)
        enhanced_magnitude = self._apply_noise_reduction(magnitude)
        
        # Reconstruct enhanced audio
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = self.istft(enhanced_stft)
        
        return mel_features, enhanced_audio

    def _apply_noise_reduction(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Apply noise reduction to magnitude spectrogram
        This is a placeholder - replace with actual noise reduction model
        """
        # Simple spectral subtraction as placeholder
        noise_floor = torch.mean(magnitude, dim=-1, keepdim=True)
        enhanced = torch.clamp(magnitude - 0.1 * noise_floor, min=0)
        return enhanced

    def reset(self):
        """Reset the audio buffer"""
        self.buffer = torch.zeros(self.config.n_fft, device=self.device)

class AudioStream:
    """Base class for audio stream sources"""
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    async def read(self, size: int) -> Optional[torch.Tensor]:
        """Read a chunk of audio data"""
        raise NotImplementedError

class MicrophoneStream(AudioStream):
    """Stream audio from microphone"""
    def __init__(self, device_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        import sounddevice as sd
        self.stream = sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        self.stream.start()
        
    async def read(self, size: int) -> Optional[torch.Tensor]:
        try:
            data, _ = self.stream.read(size)
            return torch.from_numpy(data).squeeze()
        except Exception as e:
            print(f"Error reading from microphone: {e}")
            return None
            
    def __del__(self):
        self.stream.stop()
        self.stream.close()

class FileStream(AudioStream):
    """Stream audio from file"""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.waveform, self.sr = torchaudio.load(file_path)
        self.position = 0
        
    async def read(self, size: int) -> Optional[torch.Tensor]:
        if self.position >= len(self.waveform[0]):
            return None
            
        end = min(self.position + size, len(self.waveform[0]))
        chunk = self.waveform[0, self.position:end]
        self.position = end
        
        # Pad if necessary
        if len(chunk) < size:
            chunk = torch.nn.functional.pad(chunk, (0, size - len(chunk)))
            
        return chunk