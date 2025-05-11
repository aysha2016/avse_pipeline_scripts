#!/usr/bin/env python3

import asyncio
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

from real_time_pipeline import RealTimePipeline, StreamConfig
from audio_preprocessing import AudioProcessor, MicrophoneStream, FileStream as AudioFileStream
from visual_feature_extraction import VisualFeatureExtractor, WebcamStream, FileVideoStream
from meg_processing import MEGProcessor, FileMEGStream, RealTimeMEGStream
from fusion_model import AVSEFusionModel, FusionConfig
from language_model import AsyncLLMProcessor, LLMConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AVSEDisplay:
    """Display class for real-time visualization"""
    def __init__(self, window_name: str = "AVSE-Cognition-AI"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Create figure for audio visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plt.ion()  # Enable interactive mode
        
    def update(self, result: Dict[str, Any], frame: Optional[np.ndarray] = None):
        """Update display with new results"""
        # Create display image
        display = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add video frame if available
        if frame is not None:
            # Resize frame to fit in display
            h, w = frame.shape[:2]
            scale = min(480/h, 640/w)
            new_h, new_w = int(h*scale), int(w*scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Place frame in top-left corner
            display[10:10+new_h, 10:10+new_w] = frame
            
        # Add transcription and context
        y_offset = 500
        cv2.putText(display, "Transcription:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, result.get('transcription', ''), (10, y_offset+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add summary
        cv2.putText(display, "Summary:", (10, y_offset+80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, result.get('context', {}).get('summary', ''), (10, y_offset+110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add clarifications
        cv2.putText(display, "Clarifications:", (10, y_offset+160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, result.get('context', {}).get('clarifications', ''), (10, y_offset+190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add response
        cv2.putText(display, "Response:", (10, y_offset+240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, result.get('context', {}).get('response', ''), (10, y_offset+270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Update audio visualization
        if 'enhanced_audio' in result:
            enhanced_audio = result['enhanced_audio'].cpu().numpy()
            
            # Plot waveform
            self.ax1.clear()
            self.ax1.plot(enhanced_audio)
            self.ax1.set_title('Enhanced Audio Waveform')
            self.ax1.set_xlabel('Sample')
            self.ax1.set_ylabel('Amplitude')
            
            # Plot spectrogram
            self.ax2.clear()
            self.ax2.specgram(enhanced_audio, Fs=16000)
            self.ax2.set_title('Spectrogram')
            self.ax2.set_xlabel('Time (s)')
            self.ax2.set_ylabel('Frequency (Hz)')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        
        # Show display
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)
        
    def close(self):
        """Close all display windows"""
        cv2.destroyAllWindows()
        plt.close('all')

class AVSERecorder:
    """Record system outputs to files"""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize output files
        self.audio_file = self.output_dir / f"enhanced_audio_{self.timestamp}.wav"
        self.video_file = self.output_dir / f"video_{self.timestamp}.mp4"
        self.transcript_file = self.output_dir / f"transcript_{self.timestamp}.json"
        
        # Initialize video writer
        self.video_writer = None
        
    def start_recording(self, frame_size: tuple):
        """Start recording video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_file),
            fourcc,
            30.0,
            frame_size
        )
        
    def record_frame(self, frame: np.ndarray):
        """Record a video frame"""
        if self.video_writer is not None:
            self.video_writer.write(frame)
            
    def record_audio(self, audio: torch.Tensor, sample_rate: int):
        """Record enhanced audio"""
        import soundfile as sf
        sf.write(
            str(self.audio_file),
            audio.cpu().numpy(),
            sample_rate
        )
        
    def record_transcript(self, result: Dict[str, Any]):
        """Record transcription and context"""
        with open(self.transcript_file, 'a') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'transcription': result.get('transcription', ''),
                'context': result.get('context', {})
            }, f)
            f.write('\n')
            
    def stop_recording(self):
        """Stop recording and close files"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

async def run_realtime(config: StreamConfig, display: Optional[AVSEDisplay] = None,
                      recorder: Optional[AVSERecorder] = None):
    """Run the system with real-time inputs"""
    try:
        # Initialize pipeline
        pipeline = RealTimePipeline(config)
        
        # Initialize input sources
        audio_source = MicrophoneStream()
        video_source = WebcamStream()
        meg_source = RealTimeMEGStream(device_id="simulated")  # Replace with actual device ID
        
        # Start recording if enabled
        if recorder is not None:
            recorder.start_recording(config.frame_size)
        
        # Start pipeline
        await pipeline.start(audio_source, video_source, meg_source)
        
        # Main loop
        while True:
            try:
                # Get latest result
                result = pipeline.output_queue.get(timeout=0.1)
                
                # Get latest video frame
                ret, frame = video_source.read()
                if ret and frame is not None:
                    # Update display
                    if display is not None:
                        display.update(result, frame)
                    
                    # Record frame
                    if recorder is not None:
                        recorder.record_frame(frame)
                
                # Record audio and transcript
                if recorder is not None:
                    recorder.record_audio(result['enhanced_audio'], config.audio_sample_rate)
                    recorder.record_transcript(result)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue
                
    finally:
        # Cleanup
        pipeline.stop()
        if recorder is not None:
            recorder.stop_recording()
        if display is not None:
            display.close()

async def run_from_files(config: StreamConfig, audio_path: str, video_path: str,
                        meg_path: str, display: Optional[AVSEDisplay] = None,
                        recorder: Optional[AVSERecorder] = None):
    """Run the system with file inputs"""
    try:
        # Initialize pipeline
        pipeline = RealTimePipeline(config)
        
        # Initialize input sources
        audio_source = AudioFileStream(audio_path)
        video_source = FileVideoStream(video_path)
        meg_source = FileMEGStream(meg_path)
        
        # Start recording if enabled
        if recorder is not None:
            recorder.start_recording(config.frame_size)
        
        # Start pipeline
        await pipeline.start(audio_source, video_source, meg_source)
        
        # Main loop
        while True:
            try:
                # Get latest result
                result = pipeline.output_queue.get(timeout=0.1)
                
                # Get latest video frame
                ret, frame = video_source.read()
                if not ret:
                    break
                    
                if frame is not None:
                    # Update display
                    if display is not None:
                        display.update(result, frame)
                    
                    # Record frame
                    if recorder is not None:
                        recorder.record_frame(frame)
                
                # Record audio and transcript
                if recorder is not None:
                    recorder.record_audio(result['enhanced_audio'], config.audio_sample_rate)
                    recorder.record_transcript(result)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue
                
    finally:
        # Cleanup
        pipeline.stop()
        if recorder is not None:
            recorder.stop_recording()
        if display is not None:
            display.close()

def main():
    parser = argparse.ArgumentParser(description="Run AVSE-Cognition-AI system")
    parser.add_argument("--mode", choices=["realtime", "file"], default="realtime",
                      help="Run mode: realtime or file")
    parser.add_argument("--audio", type=str, help="Path to audio file (for file mode)")
    parser.add_argument("--video", type=str, help="Path to video file (for file mode)")
    parser.add_argument("--meg", type=str, help="Path to MEG data file (for file mode)")
    parser.add_argument("--output", type=str, default="outputs",
                      help="Output directory for recordings")
    parser.add_argument("--no-display", action="store_true",
                      help="Disable real-time display")
    parser.add_argument("--no-record", action="store_true",
                      help="Disable recording")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize display and recorder
    display = None if args.no_display else AVSEDisplay()
    recorder = None if args.no_record else AVSERecorder(str(output_dir))
    
    # Configure system
    config = StreamConfig()
    
    try:
        if args.mode == "realtime":
            asyncio.run(run_realtime(config, display, recorder))
        else:
            if not all([args.audio, args.video, args.meg]):
                parser.error("File mode requires --audio, --video, and --meg arguments")
            asyncio.run(run_from_files(
                config, args.audio, args.video, args.meg,
                display, recorder
            ))
    except KeyboardInterrupt:
        logger.info("Stopping system...")
    except Exception as e:
        logger.error(f"Error running system: {e}")
    finally:
        if display is not None:
            display.close()
        if recorder is not None:
            recorder.stop_recording()

if __name__ == "__main__":
    main() 