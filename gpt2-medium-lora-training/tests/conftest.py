"""
Configuration file for pytest.
Defines fixtures and other test configurations.
"""
import os
import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture that returns the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def tokenizer():
    """Fixture that provides a tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture(scope="session")
def sample_dataset():
    """Fixture that provides a small sample dataset for testing."""
    import datasets
    
    return datasets.Dataset.from_dict({
        "text": [
            "This is a test sentence.",
            "Another test sentence for the dataset.",
            "One more example to ensure everything works.",
        ]
    })

@pytest.fixture(scope="function")
def temp_dir():
    """Fixture that provides a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
