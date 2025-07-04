"""
O2TAB-GAN: Orthopaedic Oncology Tabular GAN
Author: Dr. Ehsan Pendar
Date: July 4, 2025
Description: Test script for O2TAB-GAN generator components
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
from model.synthesizer.o2_generator import (
    FourierFeatures, 
    FourierFeatureMLP, 
    FTTransformerEncoder,
    O2Generator,
    O2Discriminator
)

def test_fourier_features():
    """Test Fourier Features component."""
    print("🧪 Testing Fourier Features...")
    
    # Test parameters
    batch_size = 32
    in_features = 5
    out_features = 64
    scale = 10.0
    
    # Create model and input
    fourier_features = FourierFeatures(in_features, out_features, scale)
    x = torch.randn(batch_size, in_features)
    
    # Forward pass
    output = fourier_features(x)
    
    # Assertions
    assert output.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print(f"   ✅ Input shape: {x.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    

def test_fourier_feature_mlp():
    """Test Fourier Feature MLP component."""
    print("\n🧪 Testing Fourier Feature MLP...")
    
    # Test parameters
    batch_size = 32
    num_features = 3
    fourier_dim = 128
    
    # Test with numerical features
    mlp = FourierFeatureMLP(num_features, fourier_dim=fourier_dim)
    x = torch.randn(batch_size, num_features)
    
    output = mlp(x)
    
    # Assertions
    assert output.shape == (batch_size, num_features), f"Expected shape {(batch_size, num_features)}, got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print(f"   ✅ Input shape: {x.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    
    # Test with zero features (edge case)
    mlp_empty = FourierFeatureMLP(0)
    x_empty = torch.randn(batch_size, 0)
    output_empty = mlp_empty(x_empty)
    assert output_empty.shape == (batch_size, 0), "Zero features test failed"
    print(f"   ✅ Zero features test passed")


def test_ft_transformer_encoder():
    """Test FT-Transformer Encoder component."""
    print("\n🧪 Testing FT-Transformer Encoder...")
    
    # Test parameters
    batch_size = 32
    num_numerical = 2
    cat_cardinalities = [10, 5, 20]  # Three categorical features
    d_token = 64
    
    # Create encoder
    encoder = FTTransformerEncoder(
        num_numerical_features=num_numerical,
        cat_cardinalities=cat_cardinalities,
        d_token=d_token
    )
    
    # Create dummy inputs
    x_num = torch.randn(batch_size, num_numerical) if num_numerical > 0 else None
    x_cat = torch.randint(0, 5, (batch_size, len(cat_cardinalities))) if cat_cardinalities else None
    
    # Forward pass
    output = encoder(x_num, x_cat)
    
    # Assertions
    assert output.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {output.shape[0]}"
    assert len(output.shape) == 2, f"Expected 2D output, got shape {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print(f"   ✅ Numerical input shape: {x_num.shape if x_num is not None else 'None'}")
    print(f"   ✅ Categorical input shape: {x_cat.shape if x_cat is not None else 'None'}")
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Output dim: {encoder.output_dim}")


def test_o2_generator():
    """Test the complete O2Generator."""
    print("\n🧪 Testing O2Generator...")
    
    # Test parameters
    batch_size = 32
    noise_dim = 100
    cond_dim = 19  # Based on our age groups from data analysis
    input_dim = noise_dim + cond_dim
    
    # Mock output_info (simplified version based on SEER data)
    output_info = [
        (1, 'tanh'),    # Year of diagnosis (numerical)
        (1, 'tanh'),    # Survival months (numerical)
        (19, 'softmax'), # Age recode (categorical)
        (4, 'softmax'),  # Race recode (categorical)
        (2, 'softmax'),  # Sex (categorical)
        (2, 'softmax'),  # Chemotherapy recode (categorical)
    ]
    
    num_numerical_features = 2
    cat_cardinalities = [19, 4, 2, 2]
    
    # Create generator
    generator = O2Generator(
        input_dim=input_dim,
        output_info=output_info,
        num_numerical_features=num_numerical_features,
        cat_cardinalities=cat_cardinalities,
        d_token=64,
        fourier_dim=64
    )
    
    # Create inputs
    noise = torch.randn(batch_size, noise_dim)
    cond = torch.randn(batch_size, cond_dim)
    
    # Forward pass
    output = generator(noise, cond)
    
    # Calculate expected output dimension
    expected_output_dim = sum(item[0] for item in output_info)
    
    # Assertions
    assert output.shape == (batch_size, expected_output_dim), \
        f"Expected shape {(batch_size, expected_output_dim)}, got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print(f"   ✅ Noise shape: {noise.shape}")
    print(f"   ✅ Conditional shape: {cond.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Expected output dim: {expected_output_dim}")
    print(f"   ✅ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")


def test_o2_discriminator():
    """Test the O2Discriminator."""
    print("\n🧪 Testing O2Discriminator...")
    
    # Test parameters
    batch_size = 32
    input_dim = 29  # Sum of output_info dimensions from generator test
    
    # Create discriminator
    discriminator = O2Discriminator(input_dim=input_dim)
    
    # Create input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = discriminator(x)
    
    # Assertions
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    print(f"   ✅ Input shape: {x.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")


def test_gpu_compatibility():
    """Test GPU compatibility if available."""
    print("\n🧪 Testing GPU Compatibility...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   🔧 Using device: {device}")
        
        # Create simple generator and move to GPU
        generator = O2Generator(
            input_dim=120,
            output_info=[(1, 'tanh'), (2, 'softmax')],
            num_numerical_features=1,
            cat_cardinalities=[2]
        ).to(device)
        
        # Test forward pass on GPU
        noise = torch.randn(16, 100, device=device)
        cond = torch.randn(16, 20, device=device)
        
        output = generator(noise, cond)
        
        assert output.device.type == device.type, f"Output not on correct device: {output.device} vs {device}"
        print(f"   ✅ GPU forward pass successful")
        print(f"   ✅ Output device: {output.device}")
        
    else:
        print(f"   ⚠️ CUDA not available, skipping GPU test")


def main():
    """Run all tests."""
    print("="*60)
    print("O2TAB-GAN GENERATOR COMPONENT TESTS")
    print("="*60)
    
    try:
        test_fourier_features()
        test_fourier_feature_mlp()
        test_ft_transformer_encoder()
        test_o2_generator()
        test_o2_discriminator()
        test_gpu_compatibility()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! O2TAB-GAN components are working correctly.")
        print("="*60)
        
        print("\n📊 Summary:")
        print("   ✅ Fourier Features: Working")
        print("   ✅ Fourier Feature MLP: Working")
        print("   ✅ FT-Transformer Encoder: Working")
        print("   ✅ O2Generator: Working")
        print("   ✅ O2Discriminator: Working")
        print(f"   ✅ GPU Support: {'Available' if torch.cuda.is_available() else 'Not Available'}")
        
        print("\n🚀 Ready for Phase 3: Hyperparameter Optimization!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 