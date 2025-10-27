"""Audio encoders for Stage 1 RAE models.

This module provides audio encoder wrappers similar to image encoders (DINOv2, SigLIP)
but designed to work with audio inputs. Supports pretrained models like Wav2Vec2 and HuBERT.
"""

from transformers import Wav2Vec2Model, HubertModel
from torch import nn
import torch
from . import register_encoder


@register_encoder()
class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2 encoder for audio representation learning.
    
    This encoder processes raw audio waveforms and outputs contextualized representations.
    The encoder is frozen during RAE training, similar to DINOv2 for images.
    
    Args:
        model_path: Path or HuggingFace model ID for Wav2Vec2
        normalize: Whether to apply normalization to output features
    """
    
    def __init__(
        self,
        model_path: str = "facebook/wav2vec2-base",
        normalize: bool = True,
    ):
        super().__init__()
        # Load pretrained Wav2Vec2 model
        try:
            self.encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=False)
        
        # Freeze encoder parameters
        self.encoder.requires_grad_(False)
        
        # Get model dimensions
        self.hidden_size = self.encoder.config.hidden_size
        # Wav2Vec2 doesn't have patch_size like vision transformers, 
        # but we define it based on the convolution stride for compatibility
        # The model downsamples by a factor based on conv layers
        self.patch_size = 320  # Default stride for wav2vec2-base (320 samples = 20ms at 16kHz)
        
        self.normalize = normalize
        if normalize:
            self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Wav2Vec2 encoder.
        
        Args:
            x: Input audio tensor of shape (batch_size, sequence_length)
               Expected to be raw waveform at 16kHz
               
        Returns:
            Encoded features of shape (batch_size, num_frames, hidden_size)
        """
        # Wav2Vec2 expects input_values of shape (batch, sequence_length)
        outputs = self.encoder(x, output_hidden_states=True)
        
        # Get last hidden state
        # Shape: (batch_size, sequence_length, hidden_size)
        features = outputs.last_hidden_state
        
        if self.normalize:
            features = self.layer_norm(features)
            
        return features


@register_encoder()
class HuBERTEncoder(nn.Module):
    """HuBERT encoder for audio representation learning.
    
    Similar to Wav2Vec2 but uses HuBERT architecture which is trained with 
    masked prediction on cluster assignments.
    
    Args:
        model_path: Path or HuggingFace model ID for HuBERT
        normalize: Whether to apply normalization to output features
    """
    
    def __init__(
        self,
        model_path: str = "facebook/hubert-base-ls960",
        normalize: bool = True,
    ):
        super().__init__()
        # Load pretrained HuBERT model
        try:
            self.encoder = HubertModel.from_pretrained(model_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = HubertModel.from_pretrained(model_path, local_files_only=False)
        
        # Freeze encoder parameters
        self.encoder.requires_grad_(False)
        
        # Get model dimensions
        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = 320  # Similar to Wav2Vec2
        
        self.normalize = normalize
        if normalize:
            self.layer_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HuBERT encoder.
        
        Args:
            x: Input audio tensor of shape (batch_size, sequence_length)
               Expected to be raw waveform at 16kHz
               
        Returns:
            Encoded features of shape (batch_size, num_frames, hidden_size)
        """
        outputs = self.encoder(x, output_hidden_states=True)
        
        # Get last hidden state
        features = outputs.last_hidden_state
        
        if self.normalize:
            features = self.layer_norm(features)
            
        return features


@register_encoder()
class SpectrogramEncoder(nn.Module):
    """Spectrogram-based encoder using Vision Transformer on mel-spectrograms.
    
    This encoder treats spectrograms as images and uses a ViT-like architecture.
    Useful when you want to leverage pretrained vision models for audio.
    
    Args:
        model_path: Path or model ID for the backbone (e.g., DINOv2)
        mel_bins: Number of mel frequency bins
        normalize: Whether to apply normalization
    """
    
    def __init__(
        self,
        model_path: str = "facebook/dinov2-base",
        mel_bins: int = 128,
        normalize: bool = True,
    ):
        super().__init__()
        # This is a placeholder for spectrogram-based encoding
        # In practice, you might use a vision transformer adapted for spectrograms
        from transformers import AutoModel
        
        try:
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=False)
        
        self.encoder.requires_grad_(False)
        
        # For spectrograms treated as images
        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = 16  # Typical patch size for ViT
        self.mel_bins = mel_bins
        
        self.normalize = normalize
        if normalize and hasattr(self.encoder, 'layernorm'):
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for spectrogram input.
        
        Args:
            x: Input spectrogram tensor of shape (batch_size, channels, height, width)
               Typically (batch, 1, mel_bins, time_frames) or (batch, 3, mel_bins, time_frames)
               
        Returns:
            Encoded features of shape (batch_size, num_patches, hidden_size)
        """
        # If input has 1 channel, replicate to 3 channels for vision models
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        outputs = self.encoder(x, output_hidden_states=True)
        
        # Extract features (depends on model type)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
            # Remove CLS token if present
            if features.shape[1] > 1 and hasattr(self.encoder.config, 'num_register_tokens'):
                unused_tokens = 1 + getattr(self.encoder.config, 'num_register_tokens', 0)
                features = features[:, unused_tokens:]
        else:
            # Fallback for different model types
            features = outputs[0]
            
        return features
