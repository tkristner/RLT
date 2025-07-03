#!/usr/bin/env python3
"""
Quick test to verify configuration loading is fixed
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os

def test_85gb_config():
    """Test 85GB configuration loading"""
    print("🔧 Testing 85GB configuration...")
    
    try:
        with hydra.initialize_config_dir(
            config_dir=os.path.abspath("cfgs/run_cfg"), 
            version_base="1.1"
        ):
            cfg = hydra.compose(config_name="teacher_rlt_mono_85gb")
            print("✅ 85GB config loaded successfully!")
            print(f"   - use_vllm: {cfg.use_vllm}")
            print(f"   - use_vllm_server: {cfg.use_vllm_server}")
            print(f"   - vllm_gpu_memory_utilization: {cfg.vllm_gpu_memory_utilization}")
            print(f"   - train_batch_size: {cfg.train_batch_size}")
            print(f"   - num_generations: {cfg.num_generations}")
        return True
    except Exception as e:
        print(f"❌ 85GB config failed: {e}")
        return False

def test_96gb_config():
    """Test 96GB configuration loading"""
    print("\n🔧 Testing 96GB configuration...")
    
    try:
        with hydra.initialize_config_dir(
            config_dir=os.path.abspath("cfgs/run_cfg"), 
            version_base="1.1"
        ):
            cfg = hydra.compose(config_name="teacher_rlt_mono_96gb")
            print("✅ 96GB config loaded successfully!")
            print(f"   - use_vllm: {cfg.use_vllm}")
            print(f"   - use_vllm_server: {cfg.use_vllm_server}")
            print(f"   - train_batch_size: {cfg.train_batch_size}")
            print(f"   - num_generations: {cfg.num_generations}")
        return True
    except Exception as e:
        print(f"❌ 96GB config failed: {e}")
        return False

def main():
    print("🚀 Quick Configuration Test")
    print("=" * 40)
    
    results = []
    results.append(("85GB Config", test_85gb_config()))
    results.append(("96GB Config", test_96gb_config()))
    
    print("\n" + "=" * 40)
    print("📋 Results:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All configurations load correctly!")
        print("Ready to run: ./launch_mono.sh teacher_rlt_mono_85gb.yaml")
    else:
        print("\n⚠️  Some configurations have issues.")

if __name__ == "__main__":
    main() 