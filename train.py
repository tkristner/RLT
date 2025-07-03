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

from custom_data.sft_data import load_formatted_sft_dataset
from custom_data.utils import get_special_token_values, make_masked_sft_collator
from custom_data.debug_utils import debug_print_examples

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb
    from omegaconf import OmegaConf

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

    # Instantiate SFTConfig and SFTTrainer directly
    sft_args_dict = OmegaConf.to_container(cfg.trainer_args, resolve=True)
    sft_args_dict.pop('_target_', None) # Remove hydra's key
    sft_args = SFTConfig(**sft_args_dict)

    trainer = SFTTrainer(
        model=trainer_kwargs["model"],
        args=sft_args,
        train_dataset=trainer_kwargs["train_dataset"],
        eval_dataset=trainer_kwargs["eval_dataset"],
        data_collator=trainer_kwargs["data_collator"],
        peft_config=trainer_kwargs["peft_config"],
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
