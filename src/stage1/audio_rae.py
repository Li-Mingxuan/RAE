"""Audio Representation Autoencoder (AudioRAE).

This module extends the RAE framework to support audio inputs, including:
- Raw waveform processing
- Mel-spectrogram conversion
- Audio-specific normalization
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from .rae import RAE
from typing import Optional
from math import sqrt


class AudioRAE(RAE):
    """Audio Representation Autoencoder.
    
    Extends RAE to handle audio inputs. Supports both:
    1. Raw waveform processing (for Wav2Vec2/HuBERT encoders)
    2. Spectrogram processing (for vision-based encoders)
    
    Args:
        audio_type: Type of audio input - 'waveform' or 'spectrogram'
        sample_rate: Audio sample rate in Hz (default: 16000)
        n_fft: FFT size for spectrogram (default: 400)
        hop_length: Hop length for spectrogram (default: 160)
        n_mels: Number of mel bins (default: 128)
        max_duration: Maximum audio duration in seconds (default: 10.0)
        target_length: Target sequence length for waveform (optional)
        **kwargs: Arguments passed to parent RAE class
    """
    
    def __init__(
        self,
        # Audio-specific configs
        audio_type: str = 'waveform',  # 'waveform' or 'spectrogram'
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160, 
        n_mels: int = 128,
        max_duration: float = 10.0,
        target_length: Optional[int] = None,
        # RAE configs (passed to parent)
        **kwargs
    ):
        # Initialize parent RAE
        super().__init__(**kwargs)
        
        self.audio_type = audio_type
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.target_length = target_length or int(sample_rate * max_duration)
        
        # Setup audio preprocessing transforms
        if audio_type == 'spectrogram':
            # Mel-spectrogram transform
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                normalized=True
            )
            # Convert to dB scale
            self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
            
            # Override encoder input size for spectrograms
            # Spectrograms will be resized to match encoder expected input
            self.n_mels = n_mels
            
        elif audio_type == 'waveform':
            # For waveform-based encoders (Wav2Vec2, HuBERT)
            # No preprocessing needed, just normalization
            pass
        else:
            raise ValueError(f"Unknown audio_type: {audio_type}. Choose 'waveform' or 'spectrogram'")
    
    def preprocess_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Preprocess raw waveform.
        
        Args:
            waveform: Input waveform of shape (batch, channels, samples) or (batch, samples)
            
        Returns:
            Preprocessed waveform of shape (batch, target_length)
        """
        # Handle different input shapes
        if waveform.dim() == 3:
            # (batch, channels, samples) -> take first channel or average
            if waveform.shape[1] > 1:
                waveform = waveform.mean(dim=1)  # Average channels
            else:
                waveform = waveform.squeeze(1)
        
        batch_size, seq_len = waveform.shape
        
        # Pad or trim to target length
        if seq_len < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - seq_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif seq_len > self.target_length:
            # Trim from center
            start = (seq_len - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
        
        # Normalize to [-1, 1]
        waveform = waveform / (waveform.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        
        return waveform
    
    def preprocess_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram.
        
        Args:
            waveform: Input waveform of shape (batch, channels, samples) or (batch, samples)
            
        Returns:
            Mel-spectrogram of shape (batch, channels, mel_bins, time_frames)
        """
        # Handle different input shapes
        if waveform.dim() == 2:
            # (batch, samples) -> (batch, 1, samples)
            waveform = waveform.unsqueeze(1)
        
        # Average channels if stereo
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        
        batch_size = waveform.shape[0]
        spectrograms = []
        
        for i in range(batch_size):
            # Compute mel-spectrogram
            mel_spec = self.mel_transform(waveform[i])  # (1, n_mels, time)
            
            # Convert to dB
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            # Normalize to [0, 1]
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            spectrograms.append(mel_spec_db)
        
        spectrograms = torch.stack(spectrograms, dim=0)  # (batch, 1, n_mels, time)
        
        # Convert to 3-channel format for vision encoders
        spectrograms = spectrograms.repeat(1, 3, 1, 1)
        
        return spectrograms
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent representation.
        
        Args:
            x: Input audio - either waveform or already preprocessed
               For waveform type: (batch, samples) or (batch, channels, samples)
               For spectrogram type: (batch, channels, height, width) if preprocessed,
                                     or (batch, samples) for raw waveform
        
        Returns:
            Latent representation
        """
        # Preprocess based on audio type
        if self.audio_type == 'waveform':
            # For waveform-based encoders (Wav2Vec2, HuBERT)
            if x.dim() <= 2 or x.shape[1] != self.encoder.hidden_size:
                # Input is raw waveform, preprocess it
                x = self.preprocess_waveform(x)
            
            # Encode with audio encoder (no image normalization needed)
            z = self.encoder(x)
            
            # Apply noising if training
            if self.training and self.noise_tau > 0:
                z = self.noising(z)
            
            # Reshape to 2D if needed
            if self.reshape_to_2d:
                b, n, c = z.shape
                h = w = int(sqrt(n))
                # If not perfect square, pad or trim
                if h * w != n:
                    # Find closest square
                    h = w = int(sqrt(n))
                    target_n = h * w
                    if target_n < n:
                        z = z[:, :target_n, :]
                    else:
                        # Pad with zeros
                        padding = target_n - n
                        z = torch.nn.functional.pad(z, (0, 0, 0, padding))
                z = z.transpose(1, 2).view(b, c, h, w)
            
        else:  # spectrogram
            # For spectrogram-based encoders (vision models)
            if x.dim() == 2 or (x.dim() == 3 and x.shape[1] == 1 and x.shape[2] < 1000):
                # Input is raw waveform, convert to spectrogram
                x = self.preprocess_spectrogram(x)
            
            # Use parent's encode method (handles vision model normalization)
            z = super().encode(x)
        
        # Apply normalization if configured
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to audio.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed audio (spectrogram format)
        """
        # Use parent's decode method
        # The output will be in the same format as encoder input
        return super().decode(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode.
        
        Args:
            x: Input audio
            
        Returns:
            Reconstructed audio
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
