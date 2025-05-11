import torch
import torchaudio
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

@dataclass
class LLMConfig:
    asr_model: str = "openai/whisper-medium"
    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"  # or your preferred model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class LLMProcessor:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize ASR model
        self.asr_processor = WhisperProcessor.from_pretrained(self.config.asr_model)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            self.config.asr_model
        ).to(self.device)
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        
        # Buffer for context
        self.context_buffer = []
        self.max_context_length = 10  # Number of previous utterances to keep
        
    async def transcribe(self, audio_features: torch.Tensor) -> str:
        """
        Transcribe audio features to text using Whisper
        
        Args:
            audio_features: Audio features of shape [batch_size, feature_dim]
            
        Returns:
            Transcribed text
        """
        try:
            # Convert features to mel spectrogram if needed
            if len(audio_features.shape) == 2:
                # Assuming input is already mel spectrogram
                mel_features = audio_features
            else:
                # Convert to mel spectrogram
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=80
                ).to(self.device)
                mel_features = mel_transform(audio_features)
            
            # Prepare input for Whisper
            input_features = self.asr_processor(
                mel_features.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.asr_model.generate(input_features)
                transcription = self.asr_processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
            
            # Update context buffer
            self.context_buffer.append(transcription)
            if len(self.context_buffer) > self.max_context_length:
                self.context_buffer.pop(0)
                
            return transcription
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
            
    async def get_context(self, transcription: str) -> Dict[str, Any]:
        """
        Get contextual information and generate response using LLM
        
        Args:
            transcription: Current transcription
            
        Returns:
            Dictionary containing context and response
        """
        try:
            # Prepare prompt with context
            context = " ".join(self.context_buffer)
            prompt = f"""Context: {context}
Current utterance: {transcription}
Please provide:
1. A summary of the key points
2. Any relevant clarifications or corrections
3. A natural response if appropriate

Response:"""
            
            # Generate response
            response = self.generator(
                prompt,
                max_length=self.config.max_length,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Parse response
            try:
                # Extract summary, clarifications, and response
                parts = response.split("\n")
                summary = next((p for p in parts if p.startswith("1.")), "")
                clarifications = next((p for p in parts if p.startswith("2.")), "")
                natural_response = next((p for p in parts if p.startswith("3.")), "")
                
                return {
                    "summary": summary.strip(),
                    "clarifications": clarifications.strip(),
                    "response": natural_response.strip(),
                    "raw_response": response
                }
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                return {
                    "summary": "",
                    "clarifications": "",
                    "response": "",
                    "raw_response": response
                }
                
        except Exception as e:
            print(f"Error in context generation: {e}")
            return {
                "summary": "",
                "clarifications": "",
                "response": "",
                "raw_response": ""
            }
            
    def reset(self):
        """Reset the context buffer"""
        self.context_buffer = []

class AsyncLLMProcessor(LLMProcessor):
    """Asynchronous version of LLMProcessor with batching"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transcription_queue = asyncio.Queue()
        self.context_queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """Start the async processor"""
        self.is_running = True
        asyncio.create_task(self._process_transcriptions())
        asyncio.create_task(self._process_contexts())
        
    async def stop(self):
        """Stop the async processor"""
        self.is_running = False
        
    async def _process_transcriptions(self):
        """Process transcription queue in batches"""
        while self.is_running:
            try:
                # Collect batch of audio features
                batch = []
                while len(batch) < self.config.batch_size:
                    try:
                        audio_features = await asyncio.wait_for(
                            self.transcription_queue.get(),
                            timeout=0.1
                        )
                        batch.append(audio_features)
                    except asyncio.TimeoutError:
                        break
                        
                if not batch:
                    continue
                    
                # Process batch
                transcriptions = await asyncio.gather(*[
                    self.transcribe(features) for features in batch
                ])
                
                # Put results in context queue
                for trans in transcriptions:
                    await self.context_queue.put(trans)
                    
            except Exception as e:
                print(f"Error in transcription processing: {e}")
                continue
                
    async def _process_contexts(self):
        """Process context queue"""
        while self.is_running:
            try:
                transcription = await self.context_queue.get()
                context = await self.get_context(transcription)
                # Handle context (e.g., display or store)
                self.context_queue.task_done()
            except Exception as e:
                print(f"Error in context processing: {e}")
                continue

def contextual_correction(transcription):
    # Placeholder: Simulate LLM correction by rule
    words = transcription.split()
    corrected = []
    for w in words:
        if w.lower() == "noize":
            corrected.append("noise")
        else:
            corrected.append(w)
    return " ".join(corrected)