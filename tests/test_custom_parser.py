import pytest
import os
import pandas as pd
import importlib.util
import sys

# Add custom_parsers to sys.path so the parser module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'custom_parsers'))

# Get target bank from environment variable, default to 'icici'
TARGET = os.getenv("TEST_TARGET", "icici")

@pytest.fixture
def pdf_path():
    """Return the path to the sample PDF file."""
    return f"../data/{TARGET}/{TARGET}_sample.pdf"

@pytest.fixture
def csv_path():
    """Return the path to the expected CSV file."""
    return f"../data/{TARGET}/{TARGET}_sample.csv"

def test_parse(pdf_path, csv_path):
    """Test that the parser's output matches the expected CSV."""
    parser_path = f"{TARGET}_parser.py"
    # Check if parser file exists
    if not os.path.exists(parser_path):
        pytest.fail(f"Parser file {parser_path} does not exist.")
    
    # Load the parser module dynamically
    spec = importlib.util.spec_from_file_location(f"{TARGET}_parser", parser_path)
    if spec is None:
        pytest.fail(f"Failed to load module spec for {parser_path}")
    if spec.loader is None:
        pytest.fail(f"No loader available for {parser_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run the parser and compare with expected CSV
    parsed_df = module.parse(pdf_path)
    expected_df = pd.read_csv(csv_path)
    assert parsed_df.equals(expected_df), (
        f"Parsed DataFrame does not match expected CSV.\n"
        f"Parsed shape: {parsed_df.shape}, Expected shape: {expected_df.shape}\n"
        f"Parsed head:\n{parsed_df.head()}\nExpected head:\n{expected_df.head()}"
    )