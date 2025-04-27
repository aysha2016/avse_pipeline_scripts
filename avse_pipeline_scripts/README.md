
# AVSE-Cognition-AI System

This project integrates Audio-Visual Speech Enhancement (AVSE), Magnetoencephalography (MEG) classifiers, and Large Language Models (LLMs) to build a real-time speech enhancement and understanding system.

## Project Structure

1. `data_preprocessing/` – Scripts for audio, video, and MEG preprocessing.
2. `avse_model/` – Audio-Visual Speech Enhancement model using U-Net and Transformer.
3. `meg_classifier/` – Classifier to detect attention/cognition states using MEG.
4. `llm_integration/` – Scripts to use LLMs for ASR correction and semantic enhancement.
5. `realtime_pipeline/` – Real-time integration and streaming logic.

## Instructions

Run the models in the order of the folders or integrate into a streaming application.

## Evaluation

Run evaluation using STOI, PESQ, and WER metrics provided in the `evaluation/` folder.
