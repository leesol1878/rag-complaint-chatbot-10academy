import pytest
import pandas as pd
import numpy as np

def test_pandas_import():
    """Test that pandas imports correctly"""
    assert pd.__version__ is not None

def test_numpy_import():
    """Test that numpy imports correctly"""
    assert np.__version__ is not None

def test_dataframe_creation():
    """Test basic pandas functionality"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    assert df.shape == (3, 2)
    assert df.columns.tolist() == ['A', 'B']

class TestTextCleaning:
    """Tests for text cleaning functions"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        from src.text_cleaning import clean_text  # We'll create this later
        text = "Hello!!! This is a test."
        result = clean_text(text)
        assert result is not None
        
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        from src.text_cleaning import clean_text
        result = clean_text("")
        assert result == ""