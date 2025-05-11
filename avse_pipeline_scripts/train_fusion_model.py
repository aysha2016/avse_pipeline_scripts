#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from typing import Tuple, Dict, Any, List
from tqdm import tqdm
import argparse

from fusion_model import AVSEFusionModel, FusionConfig
from audio_preprocessing import AudioProcessor
from visual_feature_extraction import VisualFeatureExtractor
from meg_processing import MEGProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AVSEDataset(Dataset):
    """Dataset for AVSE training"""
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load metadata
        with open(self.data_dir / f"{split}_metadata.json") as f:
            self.metadata = json.load(f)
            
        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualFeatureExtractor()
        self.meg_processor = MEGProcessor()
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a training sample"""
        sample = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / sample['audio_path']
        audio_features, _ = self.audio_processor.process_chunk(
            torch.load(audio_path)
        )
        
        # Load video features
        video_path = self.data_dir / sample['video_path']
        video_features = torch.load(video_path)
        
        # Load MEG features
        meg_path = self.data_dir / sample['meg_path']
        meg_features, _ = self.meg_processor.process_chunk(
            torch.load(meg_path)
        )
        
        # Load target (clean audio features)
        target_path = self.data_dir / sample['target_path']
        target = torch.load(target_path)
        
        return {
            'audio': audio_features,
            'video': video_features,
            'meg': meg_features
        }, target

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get batch data
        inputs, targets = batch
        audio = inputs['audio'].to(device)
        video = inputs['video'].to(device)
        meg = inputs['meg'].to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio, video, meg)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        # Get batch data
        inputs, targets = batch
        audio = inputs['audio'].to(device)
        video = inputs['video'].to(device)
        meg = inputs['meg'].to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(audio, video, meg)
        
        # Compute loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train AVSE fusion model")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing training data")
    parser.add_argument("--output-dir", type=str, default="models",
                      help="Directory to save model checkpoints")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--device", type=str,
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    config = FusionConfig()
    model = AVSEFusionModel(config).to(args.device)
    
    # Initialize dataset and dataloader
    train_dataset = AVSEDataset(args.data_dir, "train")
    val_dataset = AVSEDataset(args.data_dir, "val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_dir / "best_model.pt")
            logger.info("Saved best model")
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            model.save(output_dir / f"checkpoint_epoch_{epoch+1}.pt")
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

if __name__ == "__main__":
    main() 