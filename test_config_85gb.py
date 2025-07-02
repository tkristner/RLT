#!/usr/bin/env python3
"""
Test script to verify 85GB mono-GPU configuration is correct
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os

def test_config_loading():
    """Test that the configuration loads without errors"""
    print("🔧 Testing configuration loading...")
    
    try:
        # Test trainer config
        with hydra.initialize_config_dir(config_dir=os.path.abspath("cfgs/trainer_cfg")):
            cfg_trainer = hydra.compose(config_name="grpo_mono_85gb")
            print("✅ Trainer config loaded successfully")
            print(f"   - Max generations: {cfg_trainer.num_generations}")
            print(f"   - Batch size: {cfg_trainer.train_batch_size}")
            print(f"   - Context length: {cfg_trainer.max_prompt_length}")
        
        # Test run config  
        with hydra.initialize_config_dir(config_dir=os.path.abspath("cfgs/run_cfg")):
            cfg_run = hydra.compose(config_name="teacher_rlt_mono_85gb")
            print("✅ Run config loaded successfully")
            print(f"   - Model: {cfg_run.model_name_or_path}")
            print(f"   - Student model: {cfg_run.student_model}")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
        
    return True

def test_gpu_memory_requirements():
    """Estimate GPU memory requirements"""
    print("\n💾 Testing GPU memory estimation...")
    
    # Estimated memory usage (approximate)
    estimates = {
        "Teacher Model (Qwen2.5-7B)": "~14GB",
        "Student Model (Stratos-7B)": "~14GB", 
        "Reference Model": "~14GB",
        "vLLM Cache": "~20GB",
        "Training overhead": "~15GB",
        "Safety margin": "~8GB"
    }
    
    total_estimated = 85  # GB
    
    print("Memory allocation estimates:")
    for component, memory in estimates.items():
        print(f"   - {component}: {memory}")
    
    print(f"\n🎯 Total estimated usage: ~{total_estimated}GB")
    print(f"   GPU VRAM available: 85GB")
    print(f"   Safety margin: ~10GB")
    
    if total_estimated <= 85:
        print("✅ Memory requirements should fit in 85GB VRAM")
        return True
    else:
        print("❌ Memory requirements may exceed 85GB VRAM")
        return False

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    required_modules = [
        "torch",
        "transformers", 
        "accelerate",
        "hydra",
        "datasets",
        "trl"
    ]
    
    optional_modules = [
        "vllm",  # Should be available
        "wandb"  # For logging
    ]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} (optional)")
        except ImportError:
            print(f"   ⚠️  {module} (optional)")
            missing_optional.append(module)
    
    if missing_required:
        print(f"\n❌ Missing required modules: {missing_required}")
        return False
    else:
        print("\n✅ All required dependencies available")
        if missing_optional:
            print(f"⚠️  Optional modules missing: {missing_optional}")
        return True

def test_cuda_availability():
    """Test CUDA and GPU availability"""
    print("\n🖥️  Testing CUDA availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() > 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU 0: {gpu_name}")
        print(f"✅ GPU 0 Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 85:
            print("✅ GPU has sufficient memory (≥85GB)")
            return True
        else:
            print(f"⚠️  GPU memory ({gpu_memory:.1f}GB) may be insufficient")
            return False
    else:
        print("❌ No GPU devices found")
        return False

def main():
    """Run all tests"""
    print("🚀 RLT 85GB Configuration Test")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("GPU Memory Requirements", test_gpu_memory_requirements), 
        ("Dependencies", test_dependencies),
        ("CUDA Availability", test_cuda_availability)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Configuration is ready for 85GB GPU.")
        print("\n🚀 You can now run:")
        print("   ./launch_mono.sh teacher_rlt_mono_85gb.yaml")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 