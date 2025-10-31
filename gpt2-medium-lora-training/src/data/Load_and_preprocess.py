"""
Dataset loading and preprocessing for GPT-2 LoRA training.
"""
import os
from typing import Dict, List, Optional, Union, Any

import datasets
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_column_name: str = "text",
) -> Dict[str, List[Any]]:
    """
    Tokenize the input text.

    Args:
        examples: Dictionary containing the text data
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length
        text_column_name: Name of the text column in the dataset

    Returns:
        Dictionary with tokenized inputs
    """
    return tokenizer(
        examples[text_column_name],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

def group_texts(
    examples: Dict[str, List[List[int]]],
    block_size: int,
    keys: Optional[List[str]] = None,
) -> Dict[str, List[List[int]]]:
    """
    Group texts into chunks of fixed size.

    Args:
        examples: Dictionary containing tokenized inputs
        block_size: Size of each text chunk
        keys: Keys to process in the examples dictionary

    Returns:
        Dictionary with grouped texts
    """
    if keys is None:
        keys = ["input_ids", "attention_mask"]
        
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in keys}
    total_length = len(concatenated_examples[keys[0]])
    
    # Drop the small remainder
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Create labels for language modeling
    result["labels"] = result["input_ids"].copy()
    return result

def get_dataset(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load and preprocess the dataset.

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer to use for preprocessing
        cache_dir: Directory to cache the dataset

    Returns:
        Processed dataset
    """
    data_config = config.get("data", {})
    
    # Load dataset
    if data_config.get("dataset_name"):
        dataset = load_dataset(
            data_config["dataset_name"],
            data_config.get("dataset_config_name"),
            cache_dir=cache_dir,
            use_auth_token=config.get("hub", {}).get("hub_token"),
        )
    else:
        raise ValueError("Dataset name must be provided in config")
    
    # Process the dataset
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(
            examples,
            tokenizer=tokenizer,
            max_length=data_config.get("max_seq_length", 1024),
            text_column_name=data_config.get("text_column_name", "text"),
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=data_config.get("preprocessing_num_workers", 1),
    )
    
    # Group texts into chunks
    block_size = data_config.get("max_seq_length", 1024)
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, block_size=block_size),
        batched=True,
        batch_size=1000,
        num_proc=data_config.get("preprocessing_num_workers", 1),
    )
    
    # Split into train and validation if needed
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=data_config.get("validation_split_percentage", 5) / 100.0,
            seed=config.get("training", {}).get("seed", 42),
        )
    
    return lm_datasets

def preprocess_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_column_name: str = "text",
) -> Dict[str, List[int]]:
    """
    Preprocess function for text generation.

    Args:
        examples: Dictionary containing the text data
        tokenizer: Tokenizer to use for preprocessing
        max_length: Maximum sequence length
        text_column_name: Name of the text column in the dataset

    Returns:
        Dictionary with preprocessed inputs
    """
    return tokenizer(
        examples[text_column_name],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
