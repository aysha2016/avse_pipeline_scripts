import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class FusionConfig:
    audio_feature_dim: int = 128
    visual_feature_dim: int = 128
    meg_feature_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AVSEFusionModel(nn.Module):
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()
        self.device = torch.device(self.config.device)
        
        # Feature projection layers
        self.audio_proj = nn.Linear(self.config.audio_feature_dim, self.config.hidden_dim)
        self.visual_proj = nn.Linear(self.config.visual_feature_dim, self.config.hidden_dim)
        self.meg_proj = nn.Linear(self.config.meg_feature_dim, self.config.hidden_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, self.config.hidden_dim))
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.dropout
            ) for _ in range(self.config.num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.audio_decoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim * 2, self.config.audio_feature_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize projection layers
        for proj in [self.audio_proj, self.visual_proj, self.meg_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.xavier_uniform_(block.mlp[3].weight)
            
        # Initialize output layers
        nn.init.xavier_uniform_(self.audio_decoder[0].weight)
        nn.init.xavier_uniform_(self.audio_decoder[3].weight)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        meg_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the fusion model
        
        Args:
            audio_features: Audio features of shape [batch_size, audio_feature_dim]
            visual_features: Visual features of shape [batch_size, visual_feature_dim]
            meg_features: MEG features of shape [batch_size, meg_feature_dim]
            
        Returns:
            Enhanced audio features of shape [batch_size, audio_feature_dim]
        """
        # Project features to common dimension
        audio_proj = self.audio_proj(audio_features).unsqueeze(1)  # [B, 1, H]
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)  # [B, 1, H]
        meg_proj = self.meg_proj(meg_features).unsqueeze(1)  # [B, 1, H]
        
        # Concatenate features and add position embeddings
        x = torch.cat([audio_proj, visual_proj, meg_proj], dim=1)  # [B, 3, H]
        x = x + self.pos_embed
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Decode enhanced audio features
        enhanced_audio = self.audio_decoder(x[:, 0])  # Use audio token
        
        return enhanced_audio

    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'AVSEFusionModel':
        """Load model from checkpoint"""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model