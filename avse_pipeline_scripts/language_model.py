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