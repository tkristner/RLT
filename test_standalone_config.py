#!/usr/bin/env python3
"""
Test standalone configurations for RLT 85GB
"""

import yaml
import os

def test_yaml_loading():
    """Test that YAML files load without syntax errors"""
    print("🔧 Testing YAML syntax...")
    
    configs_to_test = [
        "cfgs/run_cfg/teacher_rlt_mono_85gb.yaml"
    ]
    
    results = []
    
    for config_path in configs_to_test:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config_name = os.path.basename(config_path)
            print(f"✅ {config_name} - YAML syntax OK")
            
            # Check key parameters
            if 'train_batch_size' in config:
                print(f"   - Batch size: {config['train_batch_size']}")
            if 'num_generations' in config:
                print(f"   - Generations: {config['num_generations']}")
            if 'max_prompt_length' in config:
                print(f"   - Context length: {config['max_prompt_length']}")
            if 'use_vllm' in config:
                print(f"   - Use vLLM: {config['use_vllm']}")
            if 'use_vllm_server' in config:
                print(f"   - vLLM server: {config['use_vllm_server']}")
                
            results.append((config_name, True))
            
        except yaml.YAMLError as e:
            print(f"❌ {config_path} - YAML error: {e}")
            results.append((config_path, False))
        except FileNotFoundError:
            print(f"❌ {config_path} - File not found")
            results.append((config_path, False))
        except Exception as e:
            print(f"❌ {config_path} - Error: {e}")
            results.append((config_path, False))
    
    return results

def test_required_parameters():
    """Test that required parameters are present"""
    print("\n🔍 Testing required parameters...")
    
    config_path = "cfgs/run_cfg/teacher_rlt_mono_85gb.yaml"
    
    required_params = [
        'model_name_or_path',
        'train_batch_size', 
        'num_generations',
        'max_prompt_length',
        'max_completion_length',
        'student_model',
        'use_vllm',
        'use_vllm_server',
        'learning_rate',
        'beta',
        'wandb_project',
        'output_dir'
    ]
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        missing_params = []
        for param in required_params:
            if param not in config:
                missing_params.append(param)
            else:
                print(f"   ✅ {param}: {config[param]}")
        
        if missing_params:
            print(f"   ❌ Missing parameters: {missing_params}")
            return False
        else:
            print("   ✅ All required parameters present")
            return True
            
    except Exception as e:
        print(f"   ❌ Error checking parameters: {e}")
        return False

def test_memory_estimation():
    """Quick memory estimation"""
    print("\n💾 Testing memory estimation...")
    
    # Load config to get actual values
    try:
        with open("cfgs/run_cfg/teacher_rlt_mono_85gb.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        batch_size = config.get('train_batch_size', 96)
        context_length = config.get('max_prompt_length', 6144)
        generations = config.get('num_generations', 24)
        
        # Realistic estimates based on configuration
        model_memory = 14 * 3  # Teacher + Student + Reference (7B each in bfloat16)
        
        # More accurate context memory calculation
        # context_length * batch_size * bfloat16_size (2 bytes) / 1024^3 for GB
        context_memory_per_batch = (context_length * batch_size * 2) / (1024**3)
        # Add some multiplier for activations and gradients
        context_memory = context_memory_per_batch * 8  # rough multiplier for activations
        
        vllm_cache = 15  # More conservative estimate
        overhead = 10   # More conservative training overhead
        
        total_estimated = model_memory + context_memory + vllm_cache + overhead
        
        print(f"   - Models (3x7B): ~{model_memory}GB")
        print(f"   - Context/activations: ~{context_memory:.1f}GB") 
        print(f"   - vLLM cache: ~{vllm_cache}GB")
        print(f"   - Training overhead: ~{overhead}GB")
        print(f"   - Total estimated: ~{total_estimated:.1f}GB")
        
        if total_estimated <= 85:
            print("   ✅ Should fit in 85GB")
            return True
        else:
            print("   ⚠️  May exceed 85GB")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in estimation: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Standalone Configuration Test - 85GB")
    print("=" * 50)
    
    tests = [
        ("YAML Loading", test_yaml_loading),
        ("Required Parameters", test_required_parameters),
        ("Memory Estimation", test_memory_estimation)
    ]
    
    all_results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, list):
                # For yaml loading test
                success = all(r[1] for r in result)
                all_results.append((test_name, success))
            else:
                all_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            all_results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    all_passed = True
    for test_name, result in all_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("✅ Configuration is ready for training!")
        print("\n🚀 You can run:")
        print("   python train.py --config-path=cfgs/run_cfg --config-name=teacher_rlt_mono_85gb")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 