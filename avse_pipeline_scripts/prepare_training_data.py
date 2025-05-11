#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import argparse
import random
from concurrent.futures import ProcessPoolExecutor
import shutil

from audio_preprocessing import AudioProcessor
from visual_feature_extraction import VisualFeatureExtractor
from meg_processing import MEGProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_sample(
    sample: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    processors: Dict[str, Any]
) -> Dict[str, str]:
    """Process a single sample and save features"""
    try:
        # Create sample directory
        sample_dir = output_dir / sample['id']
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Process audio
        audio_path = data_dir / sample['audio_path']
        audio_features, enhanced_audio = processors['audio'].process_chunk(
            torch.load(audio_path)
        )
        torch.save(audio_features, sample_dir / "audio_features.pt")
        torch.save(enhanced_audio, sample_dir / "enhanced_audio.pt")
        
        # Process video
        video_path = data_dir / sample['video_path']
        video_features = processors['visual'].process_frame(
            np.load(video_path)
        )
        torch.save(video_features, sample_dir / "video_features.pt")
        
        # Process MEG
        meg_path = data_dir / sample['meg_path']
        meg_features, attention = processors['meg'].process_chunk(
            np.load(meg_path)
        )
        torch.save(meg_features, sample_dir / "meg_features.pt")
        torch.save(torch.tensor(attention), sample_dir / "attention.pt")
        
        # Save target (clean audio features)
        target_path = data_dir / sample['target_path']
        target = torch.load(target_path)
        torch.save(target, sample_dir / "target.pt")
        
        return {
            'id': sample['id'],
            'audio_path': str(sample_dir / "audio_features.pt"),
            'video_path': str(sample_dir / "video_features.pt"),
            'meg_path': str(sample_dir / "meg_features.pt"),
            'target_path': str(sample_dir / "target.pt"),
            'attention': float(attention)
        }
        
    except Exception as e:
        logger.error(f"Error processing sample {sample['id']}: {e}")
        return None

def prepare_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4
):
    """Prepare dataset from raw recordings"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # Initialize processors
    processors = {
        'audio': AudioProcessor(),
        'visual': VisualFeatureExtractor(),
        'meg': MEGProcessor()
    }
    
    # Shuffle and split data
    random.shuffle(metadata)
    n_samples = len(metadata)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    splits = {
        'train': metadata[:n_train],
        'val': metadata[n_train:n_train + n_val],
        'test': metadata[n_train + n_val:]
    }
    
    # Process samples in parallel
    for split_name, split_data in splits.items():
        logger.info(f"Processing {split_name} split...")
        
        # Process samples
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_sample,
                    sample,
                    data_dir,
                    output_dir / split_name,
                    processors
                )
                for sample in split_data
            ]
            
            # Collect results
            results = []
            for future in tqdm(futures, desc=f"Processing {split_name}"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Save metadata
        with open(output_dir / f"{split_name}_metadata.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved {len(results)} samples for {split_name} split")
    
    # Save dataset info
    dataset_info = {
        'num_samples': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'feature_dims': {
            'audio': processors['audio'].config.n_mels,
            'visual': 128,  # From VisualFeatureExtractor
            'meg': 64  # From MEGProcessor
        },
        'sample_rate': {
            'audio': processors['audio'].config.sample_rate,
            'video': 30,  # Assuming 30 FPS
            'meg': processors['meg'].config.sample_rate
        }
    }
    
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info("Dataset preparation completed")

def main():
    parser = argparse.ArgumentParser(description="Prepare AVSE training dataset")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing raw recordings")
    parser.add_argument("--output-dir", type=str, default="dataset",
                      help="Directory to save processed dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                      help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                      help="Ratio of test data")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of worker processes")
    parser.add_argument("--clean", action="store_true",
                      help="Clean output directory before processing")
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean and Path(args.output_dir).exists():
        shutil.rmtree(args.output_dir)
    
    # Prepare dataset
    prepare_dataset(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.num_workers
    )

if __name__ == "__main__":
    main() 