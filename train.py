import logging
import os
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import bitsandbytes as bnb
from peft import LoraConfig
from torch.utils.data import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
import wandb

from custom_data.sft_data import load_formatted_sft_dataset
from custom_data.utils import get_special_token_values, make_masked_sft_collator
from custom_data.debug_utils import debug_print_examples

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    wandb_run = wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    total_devices = get_total_devices()

    div = per_device_train_batch_size*total_devices
    steps = train_batch_size/div
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path="cfgs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True

    if OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        accumulation_steps = compute_accumulation_steps(
            train_batch_size=cfg.train_batch_size,
            per_device_train_batch_size=cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps = accumulation_steps

    logger.info(f"Accumulation steps {cfg.gradient_accumulation_steps} ----")

    using_wandb = False
    if isinstance(cfg.report_to, str):
        using_wandb = cfg.report_to == 'wandb'
    elif cfg.report_to is not None:
        for v in cfg.report_to:
            using_wandb = using_wandb or (v == 'wandb')

    if using_wandb and is_main_process:
        wandb = wandb_init(
            cfg=cfg,
            group_name=cfg.wandb_group_name,
            run_name=cfg.wandb_run_name,
            log_dir=cfg.output_dir,
        )

    # Manually instantiate tokenizer and model to allow for resizing embeddings.
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # Convert model_args from DictConfig to a standard dict for kwargs
    model_kwargs = OmegaConf.to_container(cfg.model_args, resolve=True)
    model_kwargs.pop('_target_', None)  # Remove hydra's target key
    model_kwargs.pop('use_peft', None)  # PEFT is handled by SFTTrainer
    model_path = model_kwargs.pop('model_name_or_path')
    revision = model_kwargs.pop('model_revision', None)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        **model_kwargs
    )

    # Resize model embeddings to match tokenizer size, preventing IndexError.
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if cfg.do_sft:
        datasets = hydra.utils.instantiate(
            cfg.make_dataset_fn, tokenizer=tokenizer)
        # Debug print SFT examples
        debug_print_examples(datasets['train_dataset'])
    else:
        datasets = hydra.utils.instantiate(
            cfg.make_dataset_fn, tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "train_dataset": datasets.get('train_dataset'),
        "eval_dataset": datasets.get('val_dataset'),
        "data_collator": datasets.get('collator'),
    }
    if cfg.use_peft:
        # Instantiate LoraConfig manually, removing Hydra's _target_ key
        peft_kwargs = OmegaConf.to_container(cfg.peft_config, resolve=True)
        peft_kwargs.pop('_target_', None) # Remove the _target_ key
        peft_config = LoraConfig(**peft_kwargs)
        trainer_kwargs["peft_config"] = peft_config
    else:
        trainer_kwargs["peft_config"] = None

    # Check if we're doing SFT or RL training
    if cfg.do_sft:
        # SFT Training - use SFTConfig and SFTTrainer
        sft_args_dict = OmegaConf.to_container(cfg.trainer_args, resolve=True)
        sft_args_dict.pop('_target_', None) # Remove hydra's key
        
        # Remove RL-specific arguments that don't belong in SFTConfig
        rl_specific_keys = [
            'max_prompt_length', 'num_generations', 'max_completion_length',
            'shuffle_generation_inputs', 'ds3_gather_for_generation', 'temperature',
            'top_p', 'top_k', 'min_p', 'repetition_penalty', 'generation_aggregation_steps',
                    'use_vllm', 'vllm_device', 'vllm_gpu_memory_utilization', 'vllm_dtype',
        'vllm_max_model_len', 'vllm_quantization', 'use_ray', 'ray_share_training_devices',
            'ray_tensor_parallelism', 'ray_data_parallelism', 'ray_no_memory_duplication',
            'enable_prefix_caching', 'enforce_eager', 'vllm_sleep_level',
            'use_vllm_server', 'vllm_host', 'vllm_port', 'vllm_group_port',
            'num_vllm_clients', 'beta', 'reward_weights', 'sync_ref_model',
            'ref_model_mixup_alpha', 'ref_model_sync_steps', 'log_completions',
            'save_completions_probability', 'artificial_epochs',
            'backprop_accumulation_steps', 'backprop_accumulation_micro_batch_size',
            'offload_untrained_models', 'unbias_log_probabilities',
            'activate_debugging_logs', 'model_init_kwargs', 'remove_unused_columns'
        ]
        
        for key in rl_specific_keys:
            sft_args_dict.pop(key, None)
        
        sft_args = SFTConfig(**sft_args_dict)

        trainer = SFTTrainer(
            model=trainer_kwargs["model"],
            args=sft_args,
            train_dataset=trainer_kwargs["train_dataset"],
            eval_dataset=trainer_kwargs["eval_dataset"],
            data_collator=trainer_kwargs["data_collator"],
            peft_config=trainer_kwargs["peft_config"],
        )
    else:
        # RL Training - use the trainer from hydra configuration
        # Create PEFT config manually to avoid OmegaConf Union issues
        if cfg.use_peft:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        else:
            peft_config = None
            
        # Remove arguments that GRPOTrainer doesn't accept
        rl_trainer_kwargs = {
            "model": trainer_kwargs["model"],
            "train_dataset": trainer_kwargs["train_dataset"],
            "eval_dataset": trainer_kwargs["eval_dataset"],
        }
        if peft_config is not None:
            rl_trainer_kwargs["peft_config"] = peft_config
            
        # Create trainer config manually to avoid OmegaConf issues
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer_config.pop('_target_', None)
        
        # Remove parameters that we're passing explicitly to avoid duplicates
        trainer_config.pop('model', None)
        trainer_config.pop('train_dataset', None) 
        trainer_config.pop('eval_dataset', None)
        trainer_config.pop('peft_config', None)
        trainer_config.pop('reward_funcs', None)  # This will be passed separately
        trainer_config.pop('args', None)  # Remove args to avoid conflict with grpo_args
        
        # Extract trainer args and create GRPOConfig
        trainer_args_config = OmegaConf.to_container(cfg.trainer_args, resolve=True)
        trainer_args_config.pop('_target_', None)
        
        # Import and create GRPOConfig
        from trainers.grpo_config import GRPOConfig
        grpo_args = GRPOConfig(**trainer_args_config)
        
        # Import and instantiate the trainer class directly
        from trainers import TeacherGRPOTrainer
        
        # Extract reward_funcs from configuration
        reward_funcs = hydra.utils.instantiate(cfg.reward_fns)
        
        trainer = TeacherGRPOTrainer(
            model=trainer_kwargs["model"],
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=trainer_kwargs["train_dataset"],
            eval_dataset=trainer_kwargs["eval_dataset"],
            peft_config=peft_config,
            **trainer_config  # Pass remaining config parameters
        )

    print('Model initialized!!!')

    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if cfg.save_final_model:
        logger.info(f"Saving final model at {cfg.output_dir}")
        trainer.model.config.use_cache = True
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        logger.info(f"Done saving {datetime.now()}")

    if is_main_process and cfg.push_to_hub:
        tags = cfg.tags if cfg.tags is not None else []
        trainer.create_model_card({"tags": tags})
    if cfg.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    if is_main_process and cfg.call_post_training is not None:

        hydra.utils.instantiate(cfg.call_post_training)


if __name__ == "__main__":
    main()
