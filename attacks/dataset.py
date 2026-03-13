"""
Dataset loading for VCFP (Voice Command Fingerprinting) attack.

This module provides functions to load and prepare datasets from burst
CSV files for training and testing classifiers.
"""

import os
import numpy as np
from collections import defaultdict

from services.data_loader import load_burst_csv, get_label_from_filename
from attacks.features import compute_bayes_feature, compute_vngpp_feature, compute_jaccard_feature


def load_dataset_from_bursts(csv_dir, feature_method='bayes', interval=50):
    """
    Load dataset from burst CSV directory.
    
    Args:
        csv_dir: Directory containing burst CSV files
        feature_method: 'bayes', 'vngpp', or 'jaccard'
        interval: Interval for feature extraction (used for bayes and vngpp)
    
    Returns:
        X: Feature matrix
        y: Labels (as integers)
        label_map: Mapping of label strings to integer indices
    """
    X = []
    y = []
    label_map = defaultdict(int)
    label_count = 0
    
    for filename in os.listdir(csv_dir):
        if not filename.endswith('.csv'):
            continue
        
        csv_path = os.path.join(csv_dir, filename)
        
        # Load the burst CSV
        df = load_burst_csv(csv_path)
        
        if len(df) == 0:
            continue
        
        # Get label from filename
        label = get_label_from_filename(csv_path)
        
        # Get label index
        if label not in label_map:
            label_map[label] = label_count
            label_count += 1
        
        # Compute features based on method
        if feature_method == 'bayes':
            features = compute_bayes_feature(df, interval)
        elif feature_method == 'vngpp':
            features = compute_vngpp_feature(df, interval)
        elif feature_method == 'jaccard':
            features = compute_jaccard_feature(df)
        else:
            raise ValueError(f"Unknown feature method: {feature_method}")
        
        X.append(features)
        y.append(label_map[label])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples with {len(label_map)} classes")
    print(f"Feature dimension: {X.shape[1] if len(X) > 0 else 0}")
    
    return X, y, dict(label_map)
