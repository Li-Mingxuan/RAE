#!/usr/bin/env python
"""
Audio sampling/reconstruction script for Stage 1 AudioRAE.

This script demonstrates how to:
1. Load a pretrained AudioRAE model
2. Load an audio file
3. Encode it to latent representation
4. Decode it back to reconstruct the audio
5. Save the reconstructed audio

Usage:
    python src/audio_sample.py \\
        --config configs/stage1/audio/wav2vec2_base.yaml \\
        --audio path/to/audio.wav \\
        --output reconstructed.wav
"""

import argparse
import torch
import torchaudio
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1.audio_rae import AudioRAE
from utils.model_utils import instantiate_from_config


def load_audio(audio_path: str, sample_rate: int = 16000):
    """Load audio file and resample if necessary.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tensor of shape (1, samples)
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform


def save_audio(waveform: torch.Tensor, output_path: str, sample_rate: int = 16000):
    """Save audio tensor to file.
    
    Args:
        waveform: Audio tensor of shape (1, samples) or (samples,)
        output_path: Output file path
        sample_rate: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Ensure waveform is on CPU
    waveform = waveform.cpu()
    
    # Normalize to [-1, 1] if needed
    max_val = waveform.abs().max()
    if max_val > 1.0:
        waveform = waveform / max_val
    
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"Saved audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audio reconstruction with AudioRAE")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to input audio file")
    parser.add_argument("--output", type=str, default="reconstructed.wav",
                        help="Path to save reconstructed audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint file (optional)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    
    # Create model
    print("Creating AudioRAE model...")
    model: AudioRAE = instantiate_from_config(config.stage_1)
    
    # Load checkpoint if provided
    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
    
    model = model.to(args.device)
    model.eval()
    
    # Load audio
    print(f"Loading audio from {args.audio}")
    sample_rate = config.stage_1.params.get('sample_rate', 16000)
    waveform = load_audio(args.audio, sample_rate=sample_rate)
    print(f"Audio shape: {waveform.shape}, duration: {waveform.shape[1] / sample_rate:.2f}s")
    
    # Add batch dimension and move to device
    waveform = waveform.unsqueeze(0).to(args.device)  # (1, 1, samples)
    
    # Encode and decode
    print("Encoding audio to latent representation...")
    with torch.no_grad():
        # Encode
        latent = model.encode(waveform)
        print(f"Latent shape: {latent.shape}")
        
        # Decode
        print("Decoding latent back to audio...")
        reconstructed = model.decode(latent)
        print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Handle different output formats
    if model.audio_type == 'spectrogram':
        # If output is spectrogram, we need to convert back to waveform
        # This would require a vocoder (e.g., Griffin-Lim or neural vocoder)
        print("Warning: Spectrogram output requires a vocoder to convert to waveform.")
        print("Saving spectrogram as image instead...")
        
        # Save spectrogram as image
        import matplotlib.pyplot as plt
        spec = reconstructed[0].cpu().numpy()
        if spec.shape[0] == 3:
            spec = spec.mean(axis=0)  # Average RGB channels
        elif spec.shape[0] == 1:
            spec = spec[0]
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Reconstructed Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        output_img = args.output.replace('.wav', '.png')
        plt.savefig(output_img)
        print(f"Saved spectrogram to {output_img}")
        
    else:  # waveform
        # For waveform models, the decoder output needs proper handling
        # The decoder outputs patches that need to be converted back to waveform
        print("Note: Direct waveform reconstruction from ViT decoder requires custom unpatchify logic.")
        print("Saving latent representation instead...")
        
        # Save latent
        latent_path = args.output.replace('.wav', '_latent.pt')
        torch.save(latent.cpu(), latent_path)
        print(f"Saved latent representation to {latent_path}")
        
        # Also save a simple reconstruction attempt
        # This is a placeholder - in practice, you'd need a proper audio decoder
        try:
            # Attempt to reshape and save
            if reconstructed.dim() == 4:  # (B, C, H, W)
                # Flatten spatial dimensions
                B, C, H, W = reconstructed.shape
                audio_out = reconstructed.view(B, -1)  # (B, C*H*W)
                
                # Trim or pad to match input length
                target_len = waveform.shape[-1]
                if audio_out.shape[1] > target_len:
                    audio_out = audio_out[:, :target_len]
                else:
                    pad_len = target_len - audio_out.shape[1]
                    audio_out = torch.nn.functional.pad(audio_out, (0, pad_len))
                
                save_audio(audio_out[0], args.output, sample_rate)
        except Exception as e:
            print(f"Could not save audio reconstruction: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
