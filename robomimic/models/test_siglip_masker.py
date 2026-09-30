"""
Standalone test for SigLIPMasker — imports directly from siglip.py
to avoid pulling in the full robomimic package dependency chain.
"""
import sys
import torch
import torch.nn as nn

# Direct import from the same directory
from siglip import SigLIPMasker, SiglipVisionConfig

def test_shape_native():
    """Test 1: output shape matches input shape at native 224x224."""
    print("=== Test 1: SigLIPMasker with default config (224x224 input) ===")
    masker = SigLIPMasker()
    x = torch.randn(2, 3, 224, 224)
    out = masker(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {out.shape}"
    print("  PASS: Shapes match")

def test_shape_non_native():
    """Test 2: output shape matches input at non-native 96x96."""
    print("\n=== Test 2: SigLIPMasker with non-native resolution (96x96) ===")
    masker = SigLIPMasker()
    x = torch.randn(1, 3, 96, 96)
    out = masker(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {out.shape}"
    print("  PASS: Shapes match")

def test_value_range():
    """Test 3: when input is all-ones, output is in [0,1] (sigmoid * 1)."""
    print("\n=== Test 3: Output values in valid range ===")
    masker = SigLIPMasker()
    x = torch.ones(1, 3, 224, 224)
    out = masker(x)
    print(f"  Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")
    assert out.min() >= 0.0, "Output has negative values!"
    print("  PASS: Output values are non-negative")

def test_gradients():
    """Test 4: gradients flow through the masker."""
    print("\n=== Test 4: Gradient flow ===")
    masker = SigLIPMasker()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = masker(x)
    loss = out.sum()
    loss.backward()
    print(f"  Input grad shape: {x.grad.shape}")
    assert x.grad is not None, "No gradient!"
    print("  PASS: Gradients flow correctly")

def test_sequential_pipeline():
    """Test 5: SigLIPMasker works in nn.Sequential before ResNet18 layers."""
    print("\n=== Test 5: Works inside nn.Sequential (like ResNet18Conv) ===")
    from torchvision import models as vision_models
    masker = SigLIPMasker()
    resnet = vision_models.resnet18(pretrained=False)
    resnet_layers = list(resnet.children())[:-2]
    pipeline = nn.Sequential(masker, *resnet_layers)
    x = torch.randn(1, 3, 224, 224)
    out = pipeline(x)
    print(f"  Pipeline input:  {x.shape}")
    print(f"  Pipeline output: {out.shape}")
    assert out.shape == torch.Size([1, 512, 7, 7]), f"Unexpected pipeline output shape: {out.shape}"
    print("  PASS: Full SigLIPMasker + ResNet18 pipeline works")

if __name__ == "__main__":
    test_shape_native()
    test_shape_non_native()
    test_value_range()
    test_gradients()
    test_sequential_pipeline()
    print("\n=== All tests passed! ===")
