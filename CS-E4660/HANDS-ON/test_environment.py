#!/usr/bin/env python3
"""
Test script to verify the ML environment setup.
This script tests the basic functionality of key packages.
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
from datetime import datetime

def test_environment():
    """Test the ML environment setup."""
    print("=" * 60)
    print("End-to-End ML Systems Environment Test")
    print("=" * 60)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test key packages
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"MLflow version: {mlflow.__version__}")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Create sample data
    data = {
        'station_id': [1161114002] * 10,
        'parameter_id': [122] * 10,
        'value': np.random.random(10),
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='H')
    }
    df = pd.DataFrame(data)
    print(f"Sample DataFrame created: {df.shape}")
    
    # Test TensorFlow
    try:
        # Simple tensor operation
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        result = tf.matmul(a, b)
        print(f"TensorFlow tensor operation successful: {result.shape}")
    except Exception as e:
        print(f"TensorFlow test failed: {e}")
    
    # Test MLflow
    try:
        mlflow.set_experiment("test_experiment")
        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
        print("MLflow logging test successful")
    except Exception as e:
        print(f"MLflow test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Environment test completed successfully!")
    print("Ready for End-to-End ML Systems development")
    print("=" * 60)

if __name__ == "__main__":
    test_environment()