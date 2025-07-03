#!/usr/bin/env python3
"""
Quick check of final 85GB configuration parameters
"""

import yaml

def main():
    print("🔍 Final 85GB Configuration Check")
    print("=" * 45)
    
    config_path = "cfgs/run_cfg/teacher_rlt_mono_85gb.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("📋 Key Parameters:")
        print(f"   🔢 train_batch_size: {config.get('train_batch_size')}")
        print(f"   🔢 per_device_train_batch_size: {config.get('per_device_train_batch_size')}")
        print(f"   🔢 num_generations: {config.get('num_generations')}")
        print(f"   📏 max_prompt_length: {config.get('max_prompt_length')}")
        print(f"   📏 max_completion_length: {config.get('max_completion_length')}")
        print(f"   🚀 vllm_gpu_memory_utilization: {config.get('vllm_gpu_memory_utilization')}")
        print(f"   ⚡ use_vllm: {config.get('use_vllm')}")
        print(f"   🖥️  use_vllm_server: {config.get('use_vllm_server')}")
        
        print("\n💾 Memory Estimation (Realistic):")
        
        # More realistic calculation
        batch_size = config.get('train_batch_size', 64)
        context_length = config.get('max_prompt_length', 6144)
        generations = config.get('num_generations', 16)
        
        # Model memory
        model_memory = 14 * 3  # 3 models of 7B each
        
        # Context memory (much more conservative)
        tokens_per_batch = context_length * batch_size
        context_memory_gb = (tokens_per_batch * 2) / (1024**3)  # bfloat16
        activation_memory = context_memory_gb * 4  # rough multiplier for activations
        
        # Other components
        vllm_cache = 12  # Conservative
        training_overhead = 8  # Conservative
        safety_margin = 5   # Additional safety
        
        total = model_memory + activation_memory + vllm_cache + training_overhead + safety_margin
        
        print(f"   🤖 Models (3x7B): {model_memory}GB")
        print(f"   🧠 Activations/Context: {activation_memory:.1f}GB")
        print(f"   ⚡ vLLM Cache: {vllm_cache}GB")
        print(f"   🔧 Training Overhead: {training_overhead}GB")
        print(f"   🛡️ Safety Margin: {safety_margin}GB")
        print(f"   ═══════════════════════════")
        print(f"   🎯 Total Estimated: {total:.1f}GB")
        
        if total <= 85:
            print(f"   ✅ Should fit in 85GB GPU! ({85-total:.1f}GB free)")
            return True
        else:
            print(f"   ❌ May exceed 85GB by {total-85:.1f}GB")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 Configuration is ready!")
        print(f"🚀 Run: python train.py --config-path=cfgs/run_cfg --config-name=teacher_rlt_mono_85gb")
    else:
        print(f"\n⚠️  May need further adjustments") 