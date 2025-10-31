"""
Tests for data loading and preprocessing functionality.
"""
import os
from unittest import TestCase, mock
from typing import Dict, List, Any

import datasets
import numpy as np
import pytest
from transformers import AutoTokenizer

from src.data.Load_and_preprocess import (
    tokenize_function,
    group_texts,
    get_dataset,
    preprocess_function,
)

class TestDataLoading(TestCase):
    """Test cases for data loading and preprocessing."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.sample_texts = [
            "This is a test sentence.",
            "Another test sentence for the dataset.",
            "One more example to ensure everything works.",
        ]
        self.sample_dataset = datasets.Dataset.from_dict({"text": self.sample_texts})
        self.config = {
            "dataset_name": "wikitext",
            "dataset_config_name": "wikitext-2-raw-v1",
            "max_length": 128,
            "block_size": 64,
            "text_column_name": "text",
        }

    def test_tokenize_function(self):
        """Test the tokenize_function correctly tokenizes input text."""
        examples = {"text": ["This is a test.", "Another test sentence."]}
        
        result = tokenize_function(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=10,
            text_column_name="text"
        )
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(len(result["input_ids"]), 2)  # Two input examples
        self.assertTrue(all(isinstance(ids, list) for ids in result["input_ids"]))

    def test_group_texts(self):
        """Test the group_texts function correctly groups tokenized texts."""
        examples = {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15]
            ]
        }
        block_size = 4
        
        result = group_texts(examples, block_size=block_size)
        
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        # Check that all chunks have length <= block_size
        self.assertTrue(all(len(chunk) <= block_size for chunk in result["input_ids"]))

    @mock.patch("datasets.load_dataset")
    def test_get_dataset(self, mock_load_dataset):
        """Test the get_dataset function with mocked dataset loading."""
        # Setup mock dataset
        mock_dataset = datasets.Dataset.from_dict({
            "text": ["Sample text 1", "Sample text 2"],
            "split": ["train", "validation"],
        })
        mock_dataset_dict = datasets.DatasetDict({
            "train": mock_dataset,
            "validation": mock_dataset,
        })
        mock_load_dataset.return_value = mock_dataset_dict
        
        # Call the function
        dataset = get_dataset(self.config, self.tokenizer)
        
        # Assertions
        self.assertIsInstance(dataset, datasets.DatasetDict)
        self.assertIn("train", dataset)
        self.assertIn("validation", dataset)
        mock_load_dataset.assert_called_once()

    def test_preprocess_function(self):
        """Test the preprocess_function correctly processes examples."""
        examples = {"text": ["Sample text for testing.", "Another example."]}
        
        result = preprocess_function(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=10,
            text_column_name="text"
        )
        
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("labels", result)
        self.assertEqual(len(result["input_ids"]), 2)  # Two input examples

    def test_tokenize_function_empty_input(self):
        """Test tokenize_function with empty input."""
        with self.assertRaises(ValueError):
            tokenize_function(
                examples={"text": [""]},
                tokenizer=self.tokenizer,
                max_length=10,
                text_column_name="text"
            )

    def test_group_texts_smaller_than_block_size(self):
        """Test group_texts with inputs smaller than block size."""
        examples = {
            "input_ids": [
                [1, 2, 3],
                [4, 5, 6, 7]
            ]
        }
        result = group_texts(examples, block_size=10)
        self.assertEqual(len(result["input_ids"]), 2)  # Should keep original examples

if __name__ == "__main__":
    pytest.main()
