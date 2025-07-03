from datasets import load_dataset, concatenate_datasets
from .utils import make_masked_sft_collator, get_special_token_values
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData)


def add_indices(ds):
    if "__index" not in ds.column_names:
        ds = ds.map(lambda x, i: {"__index": i}, with_indices=True)
    return ds


def get_process_line_fn(dataset_id_or_path, model_name=None, tokenizer=None):
    data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
    system_prompt = data.system_prompt
    # Get the correct response template for the model (Qwen2, Llama, etc)
    response_template = None
    if model_name is not None and tokenizer is not None:
        _, _, response_template = get_special_token_values(tokenizer, model_name)
    else:
        # fallback: Qwen2 default
        response_template = "<|im_start|>assistant\n"

    def process_line_fn(line):
        user_message = line.get(data.user_message_field)
        assistant_message = line.get(data.assistant_message_field)

        # ** FIX: Filter out empty/None examples **
        if not user_message or not assistant_message:
            return None

        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question_content,
            },
            {
                "role": "assistant",
                # Ajoute explicitement la balise attendue
                "content": response_template + thought_process_and_solution,
            }
        ]
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        return {"text": line_text, "original_prompt": question_content}
    return process_line_fn


def load_formatted_sft_dataset(
        tokenizer,
        dataset_id_or_path,
        dataset_local_directory=None,
        train_split='train',
        val_split=None,
        process_line_fn=None,
        model_name_or_path=None,
        completion_only_training=True,
        custom_start_of_response=None,
        keep_columns=None,
        add_dataset_indices=False,
        artificial_epochs=None,
        padding='max_length',
        truncation=True,
        num_proc=None,
        **dataset_loading_kwargs,
):

    if dataset_local_directory is None:
        dataset_local_directory = dataset_id_or_path
    dataset = load_dataset(dataset_local_directory, **dataset_loading_kwargs)
    train_dataset = dataset[train_split]
    if add_dataset_indices:
        train_dataset = add_indices(train_dataset)
    
    # Set default num_proc based on CPU count if not specified
    if num_proc is None:
        import os
        num_proc = min(8, os.cpu_count() or 4)
    
    if process_line_fn is not None:
        if isinstance(process_line_fn, (list, tuple)):
            processed_train_datasets = []
            for fn in process_line_fn:
                processed = train_dataset.map(
                    lambda x, func=fn: func(x, tokenizer=tokenizer))
                processed_train_datasets.append(processed)
            train_dataset = concatenate_datasets(
                processed_train_datasets)
        else:
            print('not loading from cache')
            train_dataset = train_dataset.map(
                lambda x: process_line_fn(x, tokenizer=tokenizer),
                num_proc=num_proc,  # Use configurable parallel processes
                desc="Processing training data"
            )
    # ** FIX: Filter out the None values returned by the new process_line_fn **
    # Only filter if process_line_fn was used and 'text' column exists
    if process_line_fn is not None and 'text' in train_dataset.column_names:
        train_dataset = train_dataset.filter(
            lambda x: x is not None and x.get('text') is not None,
            num_proc=num_proc,  # Parallelize filtering too
            desc="Filtering empty examples"
        )
    if val_split is None:
        val_dataset = None
    else:
        val_dataset = dataset[val_split]
        if add_dataset_indices:
            val_dataset = add_indices(val_dataset)
        if process_line_fn is not None:
            if isinstance(process_line_fn, (list, tuple)):
                processed_val_datasets = []
                for fn in process_line_fn:
                    processed = val_dataset.map(
                        lambda x, func=fn: func(x, tokenizer=tokenizer))
                    processed_val_datasets.append(processed)
                val_dataset = concatenate_datasets(
                    processed_val_datasets)
            else:
                val_dataset = val_dataset.map(
                    lambda x: process_line_fn(x, tokenizer=tokenizer),
                    num_proc=num_proc,  # Parallelize validation processing
                    desc="Processing validation data"
                )
        # Filter validation dataset too if needed
        if 'text' in val_dataset.column_names:
            val_dataset = val_dataset.filter(
                lambda x: x is not None and x.get('text') is not None,
                num_proc=num_proc,  # Parallelize validation filtering
                desc="Filtering validation examples"
            )
    
    if truncation:
        # Optimize tokenization with batching and caching
        def tokenize_batch(examples):
            return tokenizer(
                examples['text'], 
                truncation=True, 
                padding=padding, 
                max_length=min(8192, tokenizer.model_max_length),
                return_tensors=None  # Keep as lists for efficiency
            )
        
        train_dataset = train_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,  # Process in larger batches
            num_proc=num_proc,
            desc="Tokenizing training data",
            load_from_cache_file=True  # Enable caching
        )
        if val_dataset:
            val_dataset = val_dataset.map(
                tokenize_batch,
                batched=True,
                batch_size=1000,
                num_proc=num_proc,
                desc="Tokenizing validation data",
                load_from_cache_file=True
            )

    if keep_columns is not None:
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names
             if col not in keep_columns])

    if artificial_epochs is not None:
        assert artificial_epochs == 1, (
            'Artificial epoch, moved to GRPO to avoid shuffling samples between'
            ' different epochs.')

        train_dataset = concatenate_datasets(
            [train_dataset]*artificial_epochs)
    out_data = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if completion_only_training:
        out_data['data_collator'] = make_masked_sft_collator(
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            custom_start_of_response=custom_start_of_response,
        )

    return out_data
