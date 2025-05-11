import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import mediapipe as mp

@dataclass
class VisualConfig:
    frame_size: Tuple[int, int] = (640, 480)
    lip_region_size: Tuple[int, int] = (96, 96)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LipDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices in MediaPipe Face Mesh
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 191, 80, 81, 82
        ]

    def detect_lips(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract lip region from frame"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract lip landmarks
        lip_points = []
        for idx in self.lip_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            lip_points.append([x, y])
            
        lip_points = np.array(lip_points)
        
        # Get bounding box with padding
        x_min, y_min = np.min(lip_points, axis=0)
        x_max, y_max = np.max(lip_points, axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # Extract and resize lip region
        lip_region = frame[y_min:y_max, x_min:x_max]
        if lip_region.size == 0:
            return None
            
        lip_region = cv2.resize(lip_region, (96, 96))
        return lip_region

class VisualFeatureExtractor:
    def __init__(self, config: Optional[VisualConfig] = None):
        self.config = config or VisualConfig()
        self.device = torch.device(self.config.device)
        self.lip_detector = LipDetector()
        
        # Initialize CNN for lip feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128)
        ).to(self.device)
        
        # Buffer for temporal features
        self.feature_buffer = []
        self.buffer_size = 5  # Number of frames to keep in buffer

    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a single video frame to extract lip features
        
        Args:
            frame: BGR image array of shape [height, width, 3]
            
        Returns:
            Tensor of shape [feature_dim]
        """
        # Detect and extract lip region
        lip_region = self.lip_detector.detect_lips(frame)
        if lip_region is None:
            # Return zero features if no lips detected
            return torch.zeros(128, device=self.device)
            
        # Convert to tensor and normalize
        lip_tensor = torch.from_numpy(lip_region).float()
        lip_tensor = lip_tensor.permute(2, 0, 1) / 255.0  # [3, H, W]
        lip_tensor = lip_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Extract features
        features = self.feature_extractor(lip_tensor).squeeze()
        
        # Update feature buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)
            
        # Return temporal features
        if len(self.feature_buffer) == self.buffer_size:
            return torch.stack(self.feature_buffer).mean(dim=0)
        else:
            return features

    def reset(self):
        """Reset the feature buffer"""
        self.feature_buffer = []

class VideoStream:
    """Base class for video stream sources"""
    def __init__(self, frame_size: Tuple[int, int] = (640, 480)):
        self.frame_size = frame_size
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video stream"""
        raise NotImplementedError

class WebcamStream(VideoStream):
    """Stream video from webcam"""
    def __init__(self, device_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
        
    def __del__(self):
        self.cap.release()

class FileVideoStream(VideoStream):
    """Stream video from file"""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(file_path)
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.resize(frame, self.frame_size)
        
    def __del__(self):
        self.cap.release()