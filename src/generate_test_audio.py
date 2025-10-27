#!/usr/bin/env python
"""
Generate a simple test audio file for demonstrating Audio RAE.

This creates a synthetic audio file with a few tones that can be used
for testing the audio reconstruction pipeline.
"""

import numpy as np
import torch
import torchaudio
import argparse


def generate_test_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
    output_path: str = "test_audio.wav"
):
    """Generate a simple test audio file with multiple tones.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        output_path: Output file path
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a few tones (C, E, G chord)
    freq1 = 261.63  # C4
    freq2 = 329.63  # E4
    freq3 = 392.00  # G4
    
    # Create waveform with envelope
    signal = (
        0.3 * np.sin(2 * np.pi * freq1 * t) +
        0.2 * np.sin(2 * np.pi * freq2 * t) +
        0.2 * np.sin(2 * np.pi * freq3 * t)
    )
    
    # Apply envelope (fade in and out)
    envelope = np.ones_like(signal)
    fade_samples = int(0.1 * sample_rate)  # 100ms fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    signal = signal * envelope
    
    # Normalize
    signal = signal / np.abs(signal).max()
    
    # Convert to tensor
    waveform = torch.from_numpy(signal).float().unsqueeze(0)
    
    # Save
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"Generated test audio: {output_path}")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate}Hz")
    print(f"  Shape: {waveform.shape}")
    
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Generate test audio file")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz")
    parser.add_argument("--output", type=str, default="test_audio.wav",
                        help="Output file path")
    
    args = parser.parse_args()
    
    generate_test_audio(
        duration=args.duration,
        sample_rate=args.sample_rate,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
