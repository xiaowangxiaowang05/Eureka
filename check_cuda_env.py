#!/usr/bin/env python3
"""
CUDA Environment Checker for Isaac Gym
This script helps diagnose CUDA/GPU setup issues for Isaac Gym training.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_cuda_availability():
    """Check if CUDA is available via PyTorch."""
    print("="*80)
    print("CUDA Environment Check for Isaac Gym")
    print("="*80)
    print()
    
    # Check PyTorch CUDA
    print("1. PyTorch CUDA Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ PyTorch can see CUDA")
            print(f"   ✓ CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ CUDA Version: {torch.version.cuda}")
            print(f"   ✓ Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("   ✗ PyTorch cannot see CUDA")
            return False
    except ImportError:
        print("   ✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"   ✗ Error checking PyTorch CUDA: {e}")
        return False
    
    print()
    
    # Check nvidia-smi
    print("2. NVIDIA Driver Check:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✓ nvidia-smi works")
            # Extract driver version
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("   ✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("   ✗ nvidia-smi not found (NVIDIA drivers may not be installed)")
        return False
    except Exception as e:
        print(f"   ✗ Error running nvidia-smi: {e}")
        return False
    
    print()
    
    # Check LD_LIBRARY_PATH
    print("3. Library Path Check:")
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        print(f"   LD_LIBRARY_PATH: {ld_path}")
        paths = ld_path.split(':')
        cuda_found = False
        for path in paths:
            if path and os.path.exists(path):
                try:
                    files = os.listdir(path)
                    if any('cuda' in f.lower() for f in files):
                        print(f"   ✓ Found CUDA libraries in: {path}")
                        cuda_found = True
                except:
                    pass
        if not cuda_found:
            print("   ⚠ No CUDA libraries found in LD_LIBRARY_PATH")
    else:
        print("   ⚠ LD_LIBRARY_PATH is not set")
    
    # Check common CUDA paths
    print()
    print("4. Common CUDA Library Locations:")
    common_paths = [
        '/usr/lib/wsl/lib',  # WSL2: libcuda.so location
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib64',
    ]
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        common_paths.insert(1, os.path.join(conda_prefix, 'lib'))  # Insert after WSL path
    
    cuda_lib_found = False
    for path in common_paths:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                cuda_libs = [f for f in files if 'cuda' in f.lower() or 'cudart' in f.lower()]
                if cuda_libs:
                    print(f"   ✓ Found CUDA libraries in: {path}")
                    print(f"     Sample files: {', '.join(cuda_libs[:3])}")
                    cuda_lib_found = True
            except:
                pass
    
    if not cuda_lib_found:
        print("   ✗ Could not find CUDA libraries in common locations")
        print()
        print("   Suggested fix:")
        print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print("   (or wherever your CUDA libraries are installed)")
        return False
    
    print()
    
    # Check for libcuda.so specifically
    print("5. libcuda.so Check (required for PhysX GPU):")
    libcuda_found = False
    search_paths = ld_path.split(':') if ld_path else []
    search_paths.extend(common_paths)
    
    for path in search_paths:
        if path and os.path.exists(path):
            try:
                files = os.listdir(path)
                if any('libcuda.so' in f for f in files):
                    print(f"   ✓ Found libcuda.so in: {path}")
                    libcuda_files = [f for f in files if 'libcuda.so' in f]
                    print(f"     Files: {', '.join(libcuda_files)}")
                    libcuda_found = True
                    break
            except:
                pass
    
    if not libcuda_found:
        print("   ✗ libcuda.so not found in LD_LIBRARY_PATH or common locations")
        print()
        print("   This is critical for PhysX GPU acceleration!")
        print("   For WSL2 users:")
        print("   - libcuda.so is usually in /usr/lib/wsl/lib/")
        print("   - Add it to LD_LIBRARY_PATH:")
        print("     export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH")
        print("   - Or ensure NVIDIA drivers are installed in Windows")
        return False
    
    print()
    print("="*80)
    print("✓ All CUDA checks passed! GPU acceleration should work.")
    print("="*80)
    return True

if __name__ == "__main__":
    success = check_cuda_availability()
    sys.exit(0 if success else 1)

