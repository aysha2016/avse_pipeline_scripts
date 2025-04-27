import numpy as np

def fuse_audio_visual_meg(audio_features, visual_features, attention_flag):
    if attention_flag:
        fusion = (audio_features + np.mean(visual_features)) * 0.5
    else:
        fusion = audio_features * 0.3
    return fusion