from audio_preprocessing import denoise_audio
from visual_feature_extraction import extract_lip_features
from meg_processing import process_meg_signal
from fusion_model import fuse_audio_visual_meg
from language_model import contextual_correction

def run_pipeline(audio_path, video_path, meg_data, transcription):
    audio_features, sr = denoise_audio(audio_path)
    visual_features = extract_lip_features(video_path)
    attention_flag = process_meg_signal(meg_data)
    fused_output = fuse_audio_visual_meg(audio_features, visual_features, attention_flag)
    final_transcription = contextual_correction(transcription)
    return fused_output, final_transcription