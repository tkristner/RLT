# RLT - Mono-GPU Version

This is a refactored version of the RLT (Reinforcement Learning Teachers) project optimized for **single GPU** usage.

## 🎯 Key Changes from Original

### Dependencies Removed
- ✅ **DeepSpeed** - Multi-GPU distributed training framework
- ✅ **Ray[serve]** - Distributed computing framework
- ✅ **accelerate_configs/** - DeepSpeed configurations
- ✅ **trainers/custom_ray/** - Ray-based generation modules

### Dependencies Kept
- ✅ **Accelerate** - Still useful for single GPU optimization
- ✅ **vLLM** - Local inference engine (no server mode)
- ✅ **All other core dependencies**

## 🚀 Quick Start


### 1. Training

#### SFT Pre-training (Mono-GPU)
```bash
# Use standard launch script with mono-GPU config
./launch.sh 1 cfgs/run_cfg/teacher_sft.yaml output_dir=path/to/save/pre_rl_model
```

#### RL Training (Mono-GPU)
```bash
# Use simplified mono-GPU launcher
./launch_mono.sh cfgs/run_cfg/teacher_rlt_mono.yaml model_name_or_path=path/of/saved/pre_rl_model
```

## ⚙️ Mono-GPU Configuration

### New Configuration Files

1. **`cfgs/trainer_cfg/grpo_mono.yaml`** - GRPO trainer optimized for single GPU
2. **`cfgs/run_cfg/teacher_rlt_mono.yaml`** - Complete run configuration for mono-GPU

### Key Parameter Adjustments

| Parameter | Original | Mono-GPU | Reason |
|-----------|----------|----------|---------|
| `train_batch_size` | 1024 | 32 | Memory constraints |
| `num_generations` | 64 | 8 | Reduced parallelism |
| `max_completion_length` | 16384 | 4096 | Memory efficiency |
| `vllm_gpu_memory_utilization` | 0.9 | 0.7 | Leave room for training |
| `offload_untrained_models` | false | true | CPU offloading for memory |
| `use_vllm_server` | true | false | Local vLLM instead |

## 🔧 Memory Optimization Features

### Enabled Optimizations
- ✅ **CPU Offloading** - Reference and reward models
- ✅ **Prefix Caching** - Improved vLLM efficiency 
- ✅ **Smaller Context Windows** - Reduced memory footprint
- ✅ **Simplified Accumulation** - Single-step accumulation
- ✅ **Debugging Logs** - Better monitoring for single GPU

### vLLM Configuration
- **Local Mode**: No server/client architecture
- **Memory Usage**: 70% GPU memory allocation
- **Prefix Caching**: Enabled for efficiency
- **Eager Mode**: Disabled for better performance

## 🎮 Usage Examples

### Basic Training
```bash
./launch_mono.sh teacher_rlt_mono.yaml
```

### With Custom Parameters
```bash
./launch_mono.sh teacher_rlt_mono.yaml \
  learning_rate=0.00001 \
  max_steps=100 \
  num_generations=4
```

### With Custom Model
```bash
./launch_mono.sh teacher_rlt_mono.yaml \
  model_name_or_path=microsoft/DialoGPT-medium \
  student_model=microsoft/DialoGPT-small
```

## 📊 Expected Performance

### Memory Requirements
- **Minimum**: 12GB VRAM (with CPU offloading)
- **Recommended**: 16GB+ VRAM
- **System RAM**: 32GB+ recommended for offloading

### Training Speed
- **Throughput**: ~4-8 generations per step
- **Steps/hour**: Depends on model size and GPU
- **Convergence**: May require more steps than multi-GPU

## 🐛 Troubleshooting

### Common Issues

1. **OOM (Out of Memory)**
   - Reduce `max_completion_length` to 2048
   - Reduce `num_generations` to 4
   - Enable `offload_untrained_models: true`

2. **vLLM Issues**
   - Reduce `vllm_gpu_memory_utilization` to 0.5
   - Enable `enforce_eager: true`
   - Disable `enable_prefix_caching` if problems persist

3. **Slow Training**
   - Increase `generation_aggregation_steps`
   - Reduce `logging_steps` frequency
   - Use smaller student model

### Debug Mode
```bash
./launch_mono.sh teacher_rlt_mono.yaml activate_debugging_logs=true
```

## 📁 File Structure Changes

```
RLT/
├── cfgs/
│   ├── trainer_cfg/
│   │   └── grpo_mono.yaml          # NEW: Mono-GPU trainer config
│   └── run_cfg/
│       └── teacher_rlt_mono.yaml   # NEW: Mono-GPU run config
├── launch_mono.sh                  # NEW: Simplified launcher
├── README_MONO_GPU.md             # NEW: This documentation
├── requirements.txt               # MODIFIED: Removed multi-GPU deps
└── [removed]
    ├── accelerate_configs/        # REMOVED: DeepSpeed configs
    └── trainers/custom_ray/       # REMOVED: Ray modules
```

## 🔄 Migration from Multi-GPU

If you have existing multi-GPU checkpoints:

1. **Checkpoint Compatibility**: Most checkpoints should work directly
2. **Config Migration**: Use `teacher_rlt_mono.yaml` as base
3. **Batch Size Adjustment**: Scale down batch sizes appropriately
4. **Memory Settings**: Adjust vLLM memory allocation

## 💡 Tips for Best Results

1. **Start Small**: Begin with smaller models and contexts
2. **Monitor Memory**: Use `activate_debugging_logs=true`
3. **Iterative Scaling**: Gradually increase parameters
4. **Save Frequently**: Use smaller `save_steps` intervals
5. **Experiment**: Try different `num_generations` values

## 📞 Support

For mono-GPU specific issues, check:
1. Memory usage with `nvidia-smi`
2. Debug logs for detailed error messages
3. vLLM configuration settings
4. Batch size compatibility

Original RLT documentation: [README.md](README.md) 