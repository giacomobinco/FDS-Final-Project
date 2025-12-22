import torch
import numpy as np
import torchaudio
import gradio as gr
from transformers import AutoFeatureExtractor, ASTForAudioClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# label mapping 
id2label = {0: 'Electronic', 
            1: 'Experimental', 
            2: 'Folk', 
            3: 'Hip-Hop', 
            4: 'Instrumental', 
            5: 'International', 
            6: 'Pop', 
            7: 'Rock'}

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

# Load base AST model
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True
)

# Load fine-tuned weights
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

# Fix label maps
model.config.id2label = id2label
model.config.label2id = {v: k for k, v in id2label.items()}

model.to(DEVICE)
model.eval()


# Prediction Function 
MAX_SECONDS = 30
TARGET_SR = 16000

def predict_genre(audio):
    sr, waveform = audio

    # Convert to torch tensor
    waveform = torch.tensor(waveform).float()

    # Convert stereo → mono safely
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    # Resample if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    samples_per_chunk = MAX_SECONDS * TARGET_SR
    total_samples = waveform.shape[0]

    # Split into chunks
    chunks = []
    for start in range(0, total_samples, samples_per_chunk):
        chunk = waveform[start:start + samples_per_chunk]

        # Skip very short tail
        if chunk.shape[0] < TARGET_SR * 5:
            continue

        # Pad last chunk if needed
        if chunk.shape[0] < samples_per_chunk:
            pad = samples_per_chunk - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        chunks.append(chunk)

    if len(chunks) == 0:
        raise gr.Error("Audio too short or invalid")

    all_probs = []

    for chunk in chunks:
        inputs = feature_extractor(
            chunk,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).input_values.to(DEVICE)

        with torch.no_grad():
            logits = model(inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        all_probs.append(probs.cpu().numpy())

    # Average probabilities across chunks
    avg_probs = np.mean(all_probs, axis=0)

    return {id2label[i]: float(avg_probs[i]) for i in range(len(id2label))}

# Gradio Interface 
interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="numpy", label="Upload an audio file"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Genre"),
    title="Music Genre Classification (AST Model)",
    description="Upload an audio clip to classify it into a music genre using a fine-tuned AST Transformer model.",
)

interface.launch(share=True)
