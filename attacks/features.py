"""
Feature extraction functions for VCFP (Voice Command Fingerprinting) attack.

This module provides functions to extract features from network traffic bursts
for training classifiers to identify voice commands.
"""

import numpy as np


def compute_bayes_feature(df, interval=50):
    """
    Compute features for Bayes classifier using packet size histograms.
    
    Args:
        df: DataFrame with traffic data (must have 'length' column)
        interval: Bucket size for histogram
    
    Returns:
        Feature vector (histogram of packet sizes)
    """
    # Create size histogram
    start, end, step = -1500, 1501, interval
    ranges = list(range(start, end, step))
    features = [0] * len(ranges)
    
    for size in df['length']:
        idx = int((size - start) / step)
        if 0 <= idx < len(features):
            features[idx] += 1
    
    return features


def compute_vngpp_feature(df, interval=5000):
    """
    Compute VNG++ features using cumulative packet sizes.
    
    Args:
        df: DataFrame with traffic data (must have 'length' column)
        interval: Bucket size for cumulative sum
    
    Returns:
        Feature vector (histogram of cumulative packet sizes)
    """
    # Compute cumulative sum of sizes
    cumsum = df['length'].cumsum().values
    
    # Create histogram
    start, end, step = -400000, 400001, interval
    ranges = list(range(start, end, step))
    features = [0] * len(ranges)
    
    for val in cumsum:
        idx = int((val - start) / step)
        if 0 <= idx < len(features):
            features[idx] += 1
    
    return features


def compute_jaccard_feature(df):
    """
    Compute Jaccard similarity features.
    
    Args:
        df: DataFrame with traffic data (must have 'length' column)
    
    Returns:
        Set of packet sizes
    """
    return set(df['length'].values)
