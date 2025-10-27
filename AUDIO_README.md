# Audio RAE (Representation Autoencoder for Audio)

This guide explains how to use the Audio RAE extension for audio/speech processing.

## Overview

Audio RAE extends the original image-based RAE framework to support audio inputs. The same two-stage architecture applies:

1. **Stage 1**: Audio Representation Autoencoder (encoder frozen, decoder trained)
2. **Stage 2**: Latent diffusion model for audio generation

## Key Features

- **Multiple Encoder Types**:
  - `Wav2Vec2Encoder`: Uses Facebook's Wav2Vec2 for raw waveform encoding
  - `HuBERTEncoder`: Uses Facebook's HuBERT for robust audio representations
  - `SpectrogramEncoder`: Treats mel-spectrograms as images, uses vision models (DINOv2)

- **Flexible Input Formats**:
  - Raw waveforms (for Wav2Vec2/HuBERT)
  - Mel-spectrograms (for vision-based encoders)

- **Automatic Preprocessing**:
  - Waveform normalization and padding/trimming
  - Mel-spectrogram conversion
  - Sample rate conversion

## Installation

### Additional Dependencies for Audio

```bash
# Install audio-specific dependencies
pip install torchaudio librosa soundfile
```

All other dependencies are the same as the base RAE project.

## Quick Start

### 1. Basic Audio Reconstruction

Reconstruct an audio file using a pretrained AudioRAE model:

```bash
python src/audio_sample.py \
  --config configs/stage1/audio/wav2vec2_base.yaml \
  --audio path/to/your/audio.wav \
  --output reconstructed.wav
```

### 2. Training Stage 1 AudioRAE

Train the audio decoder (encoder is frozen):

```bash
torchrun --standalone --nproc_per_node=N \
  src/train_stage1.py \
  --config configs/stage1/audio/wav2vec2_base.yaml \
  --data-path path/to/audio/dataset \
  --results-dir results/audio_stage1 \
  --precision bf16
```

**Dataset Format**: The audio dataset should be organized similarly to ImageNet:
```
audio_dataset/
├── train/
│   ├── class1/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   ├── class2/
│   │   └── ...
└── val/
    └── ...
```

For non-classification tasks, put all audio files in a single directory.

## Configuration Files

Three example configurations are provided:

### 1. Wav2Vec2-based (Recommended for Speech)

```yaml
# configs/stage1/audio/wav2vec2_base.yaml
stage_1:
  target: stage1.audio_rae.AudioRAE
  params:
    audio_type: 'waveform'
    sample_rate: 16000
    max_duration: 10.0
    encoder_cls: 'Wav2Vec2Encoder'
    encoder_config_path: 'facebook/wav2vec2-base'
    # ... other params
```

Best for: Speech recognition, voice-related tasks

### 2. HuBERT-based (Recommended for General Audio)

```yaml
# configs/stage1/audio/hubert_base.yaml
stage_1:
  target: stage1.audio_rae.AudioRAE
  params:
    audio_type: 'waveform'
    encoder_cls: 'HuBERTEncoder'
    # ... other params
```

Best for: General audio, music, environmental sounds

### 3. Spectrogram-based (Vision Model)

```yaml
# configs/stage1/audio/spectrogram_dinov2.yaml
stage_1:
  target: stage1.audio_rae.AudioRAE
  params:
    audio_type: 'spectrogram'
    n_mels: 128
    encoder_cls: 'SpectrogramEncoder'
    # ... other params
```

Best for: When you want to leverage pretrained vision models, music analysis

## Architecture Details

### AudioRAE Class

The `AudioRAE` class (in `src/stage1/audio_rae.py`) extends the base `RAE` class with:

- **Audio preprocessing**: Converts raw audio to appropriate format
- **Flexible encoding**: Supports both waveform and spectrogram inputs
- **Temporal handling**: Manages variable-length audio sequences

### Audio Encoders

Located in `src/stage1/encoders/audio_encoders.py`:

1. **Wav2Vec2Encoder**:
   - Input: Raw waveform (16kHz)
   - Output: Contextualized frame representations
   - Patch size: 320 samples (20ms)
   - Hidden size: 768 (base model)

2. **HuBERTEncoder**:
   - Similar to Wav2Vec2
   - More robust to noise and variations
   - Better for diverse audio types

3. **SpectrogramEncoder**:
   - Input: Mel-spectrogram treated as image
   - Uses vision transformers (e.g., DINOv2)
   - Output: Patch-based representations

## Usage Examples

### Example 1: Process a Single Audio File

```python
import torch
from omegaconf import OmegaConf
from stage1.audio_rae import AudioRAE
from utils.model_utils import instantiate_from_config

# Load config
config = OmegaConf.load('configs/stage1/audio/wav2vec2_base.yaml')

# Create model
model = instantiate_from_config(config.stage_1)
model.eval()

# Load and process audio
import torchaudio
waveform, sr = torchaudio.load('audio.wav')

# Resample if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)

# Encode
with torch.no_grad():
    latent = model.encode(waveform.unsqueeze(0))
    print(f"Latent shape: {latent.shape}")
    
    # Decode
    reconstructed = model.decode(latent)
    print(f"Reconstructed shape: {reconstructed.shape}")
```

### Example 2: Custom Audio Encoder

Create your own audio encoder by following this template:

```python
from torch import nn
import torch
from stage1.encoders import register_encoder

@register_encoder()
class CustomAudioEncoder(nn.Module):
    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        # Initialize your encoder
        self.hidden_size = 768  # Required attribute
        self.patch_size = 320   # Required attribute
        # ... your initialization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length) for waveform
        # Returns: (batch, num_frames, hidden_size)
        # ... your encoding logic
        pass
```

Then register it in `src/stage1/encoders/__init__.py` and use it in your config.

## Important Notes

### Decoder Output Format

The current ViT-based decoder outputs patches that need to be converted back to audio:

- For **spectrogram mode**: Decoder outputs spectrogram patches that can be visualized
- For **waveform mode**: Additional post-processing may be needed (vocoder, etc.)

### Vocoder Integration

For spectrogram-based approaches, you'll need a vocoder to convert spectrograms back to waveforms:

- **Griffin-Lim**: Simple, no training required
- **Neural vocoders**: WaveGlow, HiFi-GAN, etc. (better quality)

Example using Griffin-Lim:

```python
import torchaudio.transforms as T

# Convert spectrogram to waveform
vocoder = T.GriffinLim(n_fft=400, hop_length=160)
waveform = vocoder(spectrogram)
```

## Computing Normalization Statistics

For best results, compute normalization statistics on your audio dataset:

```bash
# This script needs to be created based on your dataset
python scripts/compute_audio_stats.py \
  --config configs/stage1/audio/wav2vec2_base.yaml \
  --data-path path/to/audio/dataset \
  --output models/stats/audio/wav2vec2_base_stats.pt
```

Then update your config:
```yaml
normalization_stat_path: 'models/stats/audio/wav2vec2_base_stats.pt'
```

## Tips and Best Practices

1. **Audio Duration**: Start with 5-10 second clips for training
2. **Sample Rate**: Use 16kHz for speech, consider 22.05kHz or 44.1kHz for music
3. **Batch Size**: Audio models are memory-intensive, reduce batch size if needed
4. **Encoder Choice**:
   - Speech → Wav2Vec2
   - Music/General → HuBERT or Spectrogram
   - Maximum quality → Spectrogram with DINOv2

5. **Preprocessing**: Ensure consistent audio quality (remove silence, normalize volume)

## Troubleshooting

### Out of Memory

- Reduce `max_duration` in config
- Reduce batch size
- Use gradient checkpointing
- Use mixed precision (bf16)

### Poor Reconstruction Quality

- Check audio normalization
- Compute and use normalization statistics
- Try different encoder types
- Increase decoder capacity

### Mismatched Dimensions

- Ensure `target_length` is divisible by encoder's patch size
- Check that decoder configuration matches encoder output dimensions

## Future Improvements

Potential enhancements for audio RAE:

1. **Audio-specific decoder**: Custom decoder designed for audio (not repurposed from vision)
2. **Neural vocoder integration**: Direct waveform synthesis
3. **Longer context**: Support for longer audio sequences (30s+)
4. **Multi-scale processing**: Process audio at multiple resolutions
5. **Conditional generation**: Class-conditional or text-conditional audio generation

## References

- Original RAE paper: [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)
- Wav2Vec2: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- HuBERT: [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction](https://arxiv.org/abs/2106.07447)

## Citation

If you use Audio RAE in your research, please cite both the original RAE paper and the audio models you use.
