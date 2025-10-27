#!/usr/bin/env python
"""
Simple example demonstrating Audio RAE usage.

This script shows a complete workflow:
1. Generate a test audio file
2. Create an AudioRAE model
3. Encode the audio to latent representation
4. Decode back to reconstruct the audio

This is a minimal example for documentation purposes.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def generate_simple_audio(duration=2.0, sample_rate=16000):
    """Generate a simple test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave
    signal = np.sin(2 * np.pi * 440 * t)  # A4 note
    return torch.from_numpy(signal).float().unsqueeze(0)


def example_audio_rae():
    """Example of using AudioRAE with minimal configuration."""
    
    print("=" * 60)
    print("Audio RAE Example")
    print("=" * 60 + "\n")
    
    # Note: This example demonstrates the API structure
    # In practice, you need to install dependencies first:
    # pip install torch torchaudio transformers
    
    try:
        from stage1.audio_rae import AudioRAE
        from stage1.encoders.audio_encoders import Wav2Vec2Encoder
        
        print("1. Creating Audio RAE model...")
        model = AudioRAE(
            # Audio settings
            audio_type='waveform',
            sample_rate=16000,
            max_duration=5.0,
            target_length=80000,  # 5 seconds at 16kHz
            
            # Encoder settings
            encoder_cls='Wav2Vec2Encoder',
            encoder_config_path='facebook/wav2vec2-base',
            encoder_input_size=80000,
            encoder_params={
                'model_path': 'facebook/wav2vec2-base',
                'normalize': True
            },
            
            # Decoder settings
            decoder_config_path='facebook/vit-mae-base',
            decoder_patch_size=16,
            
            # Training params
            noise_tau=0.0,  # No noise for testing
            reshape_to_2d=True,
        )
        
        model.eval()
        print("   ✓ Model created successfully\n")
        
        print("2. Generating test audio...")
        audio = generate_simple_audio(duration=2.0, sample_rate=16000)
        print(f"   Audio shape: {audio.shape}")
        print(f"   Duration: 2.0 seconds\n")
        
        print("3. Encoding audio to latent representation...")
        with torch.no_grad():
            latent = model.encode(audio.unsqueeze(0))
        print(f"   Latent shape: {latent.shape}")
        print(f"   Compression ratio: {audio.numel() / latent.numel():.2f}x\n")
        
        print("4. Decoding latent back to audio...")
        with torch.no_grad():
            reconstructed = model.decode(latent)
        print(f"   Reconstructed shape: {reconstructed.shape}\n")
        
        print("=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Train the decoder on your audio dataset")
        print("  - Compute normalization statistics")
        print("  - Use trained model for audio generation")
        print("  - See AUDIO_README.md for full documentation")
        
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("\nPlease install required packages:")
        print("  pip install torch torchaudio transformers")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(example_audio_rae())
