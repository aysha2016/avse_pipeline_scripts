import torch
import torch.nn as nn
import numpy as np
from scipy.signal import spectrogram
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class MEGConfig:
    sample_rate: int = 1000  # Hz
    n_channels: int = 306  # Typical MEG system
    window_size: int = 1000  # 1 second window
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MEGProcessor:
    def __init__(self, config: Optional[MEGConfig] = None):
        self.config = config or MEGConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize attention classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(self.config.n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: attentive vs. inattentive
        ).to(self.device)
        
        # Buffer for temporal data
        self.data_buffer = []
        self.buffer_size = self.config.window_size
        
        # Initialize preprocessing
        self._init_preprocessing()

    def _init_preprocessing(self):
        """Initialize preprocessing parameters"""
        # High-pass filter to remove DC drift
        self.highpass_freq = 0.1  # Hz
        self.highpass_order = 4
        
        # Notch filter to remove power line noise
        self.notch_freq = 60  # Hz (or 50 Hz depending on region)
        self.notch_quality = 30
        
        # Normalization parameters
        self.mean = None
        self.std = None

    def _preprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess a chunk of MEG data
        
        Args:
            chunk: Raw MEG data of shape [n_channels, n_samples]
            
        Returns:
            Preprocessed data of same shape
        """
        # Convert to numpy if needed
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.cpu().numpy()
            
        # Apply high-pass filter
        from scipy.signal import butter, filtfilt
        nyquist = self.config.sample_rate / 2
        b, a = butter(self.highpass_order, self.highpass_freq / nyquist, btype='high')
        filtered = filtfilt(b, a, chunk, axis=1)
        
        # Apply notch filter
        from scipy.signal import iirnotch
        b, a = iirnotch(self.notch_freq / nyquist, self.notch_quality)
        filtered = filtfilt(b, a, filtered, axis=1)
        
        # Update normalization parameters
        if self.mean is None:
            self.mean = np.mean(filtered, axis=1, keepdims=True)
            self.std = np.std(filtered, axis=1, keepdims=True)
        else:
            # Update running statistics
            alpha = 0.1  # Learning rate
            new_mean = np.mean(filtered, axis=1, keepdims=True)
            new_std = np.std(filtered, axis=1, keepdims=True)
            self.mean = (1 - alpha) * self.mean + alpha * new_mean
            self.std = (1 - alpha) * self.std + alpha * new_std
            
        # Normalize
        normalized = (filtered - self.mean) / (self.std + 1e-6)
        
        return normalized

    def process_chunk(self, chunk: np.ndarray) -> Tuple[torch.Tensor, float]:
        """
        Process a chunk of MEG data and classify attention state
        
        Args:
            chunk: Raw MEG data of shape [n_channels, n_samples]
            
        Returns:
            Tuple of (processed_features, attention_probability)
        """
        # Preprocess data
        processed = self._preprocess_chunk(chunk)
        
        # Update buffer
        self.data_buffer.append(processed)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            
        # Only process if we have enough data
        if len(self.data_buffer) < self.buffer_size:
            return torch.zeros(64, device=self.device), 0.5
            
        # Stack buffer and convert to tensor
        window = np.concatenate(self.data_buffer, axis=1)
        window_tensor = torch.from_numpy(window).float().to(self.device)
        
        # Extract features and classify
        with torch.no_grad():
            features = self.classifier[:-1](window_tensor.unsqueeze(0))
            logits = self.classifier[-1](features)
            attention_prob = torch.softmax(logits, dim=1)[0, 1].item()
            
        return features.squeeze(), attention_prob

    def reset(self):
        """Reset the processor state"""
        self.data_buffer = []
        self.mean = None
        self.std = None

class MEGStream:
    """Base class for MEG data streams"""
    def __init__(self, sample_rate: int = 1000, n_channels: int = 306):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        
    async def read(self) -> Optional[np.ndarray]:
        """Read a chunk of MEG data"""
        raise NotImplementedError

class FileMEGStream(MEGStream):
    """Stream MEG data from file"""
    def __init__(self, file_path: str, chunk_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.data = np.load(file_path)  # Assuming .npy format
        self.position = 0
        self.chunk_size = chunk_size
        
    async def read(self) -> Optional[np.ndarray]:
        if self.position >= self.data.shape[1]:
            return None
            
        end = min(self.position + self.chunk_size, self.data.shape[1])
        chunk = self.data[:, self.position:end]
        self.position = end
        
        # Pad if necessary
        if chunk.shape[1] < self.chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, self.chunk_size - chunk.shape[1])))
            
        return chunk

class RealTimeMEGStream(MEGStream):
    """Stream MEG data in real-time (placeholder)"""
    def __init__(self, device_id: str, **kwargs):
        super().__init__(**kwargs)
        # Initialize connection to MEG device
        # This is a placeholder - implement actual device connection
        self.device_id = device_id
        
    async def read(self) -> Optional[np.ndarray]:
        try:
            # Placeholder for actual device reading
            # Replace with actual device communication
            chunk = np.random.randn(self.n_channels, 1000)
            return chunk
        except Exception as e:
            print(f"Error reading from MEG device: {e}")
            return None

def process_meg_signal(meg_data):
    f, t, Sxx = spectrogram(meg_data)
    attention_score = np.mean(Sxx)
    return attention_score > np.median(Sxx)