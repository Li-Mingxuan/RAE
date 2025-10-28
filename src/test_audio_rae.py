#!/usr/bin/env python
"""
Test script for Audio RAE functionality.

This script tests:
1. Loading audio encoders
2. Creating AudioRAE models
3. Encoding and decoding audio
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from stage1.audio_rae import AudioRAE
from stage1.encoders import ARCHS


def test_encoder_registration():
    """Test that audio encoders are registered."""
    print("Testing encoder registration...")
    
    expected_encoders = ['Wav2Vec2Encoder', 'HuBERTEncoder', 'SpectrogramEncoder']
    for encoder_name in expected_encoders:
        assert encoder_name in ARCHS, f"{encoder_name} not registered"
        print(f"  ✓ {encoder_name} registered")
    
    print("All encoders registered successfully!\n")


def test_wav2vec2_encoder():
    """Test Wav2Vec2 encoder."""
    print("Testing Wav2Vec2Encoder...")
    
    from stage1.encoders.audio_encoders import Wav2Vec2Encoder
    
    # Create encoder
    encoder = Wav2Vec2Encoder(model_path='facebook/wav2vec2-base')
    
    # Create dummy waveform (1 second at 16kHz)
    batch_size = 2
    seq_len = 16000
    waveform = torch.randn(batch_size, seq_len)
    
    # Encode
    with torch.no_grad():
        features = encoder(waveform)
    
    print(f"  Input shape: {waveform.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Hidden size: {encoder.hidden_size}")
    print(f"  Patch size: {encoder.patch_size}")
    
    assert features.shape[0] == batch_size, "Batch size mismatch"
    assert features.shape[2] == encoder.hidden_size, "Hidden size mismatch"
    
    print("  ✓ Wav2Vec2Encoder test passed!\n")
    return True


def test_hubert_encoder():
    """Test HuBERT encoder."""
    print("Testing HuBERTEncoder...")
    
    from stage1.encoders.audio_encoders import HuBERTEncoder
    
    # Create encoder
    encoder = HuBERTEncoder(model_path='facebook/hubert-base-ls960')
    
    # Create dummy waveform
    batch_size = 2
    seq_len = 16000
    waveform = torch.randn(batch_size, seq_len)
    
    # Encode
    with torch.no_grad():
        features = encoder(waveform)
    
    print(f"  Input shape: {waveform.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Hidden size: {encoder.hidden_size}")
    
    assert features.shape[0] == batch_size, "Batch size mismatch"
    assert features.shape[2] == encoder.hidden_size, "Hidden size mismatch"
    
    print("  ✓ HuBERTEncoder test passed!\n")
    return True


def test_audio_rae_waveform():
    """Test AudioRAE with waveform input."""
    print("Testing AudioRAE with waveform input...")
    
    try:
        # Create AudioRAE model (minimal config)
        model = AudioRAE(
            audio_type='waveform',
            sample_rate=16000,
            max_duration=5.0,
            encoder_cls='Wav2Vec2Encoder',
            encoder_config_path='facebook/wav2vec2-base',
            encoder_input_size=80000,  # 5 seconds
            encoder_params={'model_path': 'facebook/wav2vec2-base', 'normalize': True},
            decoder_config_path='facebook/vit-mae-base',
            decoder_patch_size=16,
            noise_tau=0.0,  # Disable noise for testing
            reshape_to_2d=True,
        )
        
        model.eval()
        
        # Create dummy audio (3 seconds)
        batch_size = 1
        audio_duration = 3.0
        waveform = torch.randn(batch_size, int(16000 * audio_duration))
        
        # Test encoding
        with torch.no_grad():
            latent = model.encode(waveform)
        
        print(f"  Input shape: {waveform.shape}")
        print(f"  Latent shape: {latent.shape}")
        
        # Test decoding
        with torch.no_grad():
            reconstructed = model.decode(latent)
        
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        print("  ✓ AudioRAE waveform test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ AudioRAE waveform test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_audio_preprocessing():
    """Test audio preprocessing functions."""
    print("Testing audio preprocessing...")
    
    model = AudioRAE(
        audio_type='waveform',
        sample_rate=16000,
        max_duration=5.0,
        target_length=80000,
        encoder_cls='Wav2Vec2Encoder',
        encoder_config_path='facebook/wav2vec2-base',
        encoder_input_size=80000,
        encoder_params={'model_path': 'facebook/wav2vec2-base'},
        decoder_config_path='facebook/vit-mae-base',
        decoder_patch_size=16,
    )
    
    # Test padding
    short_audio = torch.randn(1, 40000)  # 2.5 seconds
    processed = model.preprocess_waveform(short_audio)
    assert processed.shape[1] == 80000, f"Padding failed: {processed.shape}"
    print(f"  ✓ Padding test passed (40000 → 80000)")
    
    # Test trimming
    long_audio = torch.randn(1, 160000)  # 10 seconds
    processed = model.preprocess_waveform(long_audio)
    assert processed.shape[1] == 80000, f"Trimming failed: {processed.shape}"
    print(f"  ✓ Trimming test passed (160000 → 80000)")
    
    # Test normalization
    assert processed.abs().max() <= 1.0, "Normalization failed"
    print(f"  ✓ Normalization test passed")
    
    print("  ✓ Audio preprocessing tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Audio RAE Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Encoder registration
    try:
        test_encoder_registration()
        results.append(("Encoder Registration", True))
    except Exception as e:
        print(f"✗ Encoder registration failed: {e}\n")
        results.append(("Encoder Registration", False))
    
    # Test 2: Wav2Vec2 encoder
    try:
        success = test_wav2vec2_encoder()
        results.append(("Wav2Vec2 Encoder", success))
    except Exception as e:
        print(f"✗ Wav2Vec2 encoder test failed: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Wav2Vec2 Encoder", False))
    
    # Test 3: HuBERT encoder
    try:
        success = test_hubert_encoder()
        results.append(("HuBERT Encoder", success))
    except Exception as e:
        print(f"✗ HuBERT encoder test failed: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("HuBERT Encoder", False))
    
    # Test 4: Audio preprocessing
    try:
        success = test_audio_preprocessing()
        results.append(("Audio Preprocessing", success))
    except Exception as e:
        print(f"✗ Audio preprocessing test failed: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("Audio Preprocessing", False))
    
    # Test 5: AudioRAE with waveform
    try:
        success = test_audio_rae_waveform()
        results.append(("AudioRAE Waveform", success))
    except Exception as e:
        print(f"✗ AudioRAE waveform test failed: {e}\n")
        import traceback
        traceback.print_exc()
        results.append(("AudioRAE Waveform", False))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
