"""
Data loading utilities for voice command fingerprinting project.

This module provides functions for loading burst CSV data, extracting labels
from filenames, and generating synthetic burst traffic data for testing.
"""

import os
import re
import numpy as np
import pandas as pd


# Directory paths
CAPTURED_FILES_DIR = 'data/captured_files/'
TRACE_CSV_DIR = 'data/trace_csv_files/'
SEPARATED_BURSTS_DIR = 'data/seperated_bursts_files/'
BUFLO_OUTPUT_DIR = 'data/buflo/'


def load_burst_csv(csv_path):
    """
    Load burst CSV data into a DataFrame.
    
    Args:
        csv_path: Path to the burst CSV file
        
    Returns:
        DataFrame with columns: packet_id, timestamp, length, direction, 
        burst_id, gap_before, gap_after, rel_time, viz_bytes, cumulative_sum
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Handle inf values in gap_before and gap_after columns
    df['gap_before'] = df['gap_before'].replace('inf', float('inf'))
    df['gap_after'] = df['gap_after'].replace('inf', float('inf'))
    
    return df


def get_label_from_filename(csv_path):
    """
    Extract label (query name) from CSV filename using regex pattern.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Label string (query name)
        
    Examples:
        - "good_morning_1_bursts.csv" -> "good_morning"
        - "what_is_the_weather_3_bursts.csv" -> "what_is_the_weather"
        - "play_music_2.csv" -> "play_music"
    """
    filename = os.path.basename(csv_path)
    
    # Extract query name from filename pattern like: name_bursts.csv or name_1.csv
    # For burst files: remove _bursts suffix and number suffix
    # For numbered files: remove _number suffix
    
    # Remove .csv extension first
    name = filename[:-4] if filename.endswith('.csv') else filename
    
    # Handle cases: name_bursts, name_number_bursts, name_number
    
    # If ends with _bursts, remove it
    if name.endswith('_bursts'):
        name = name[:-7]  # Remove '_bursts'
        # If name now ends with _number, remove that too
        if '_' in name:
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                name = parts[0]
        return name
    
    # If ends with _number, remove it
    if '_' in name:
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    
    return name


def generate_synthetic_burst_data(output_dir, n_queries, packets_per_query):
    """
    Generate synthetic burst traffic data for testing.
    
    Args:
        output_dir: Directory to save synthetic CSV files
        n_queries: Number of query samples to generate
        packets_per_query: Average number of packets per query
        
    Returns:
        List of generated file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predefined query templates matching real voice commands
    query_templates = [
        'what_is_the_weather',
        'tell_me_a_joke',
        'play_music',
        'set_alarm',
        'good_morning'
    ]
    
    generated_files = []
    
    for i in range(n_queries):
        query_name = query_templates[i % len(query_templates)]
        filename = f"{query_name}_{i//len(query_templates)+1}_bursts.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Generate random packet data with query-specific patterns
        np.random.seed(i)
        n_packets = packets_per_query + np.random.randint(-10, 10)
        
        # Different packet size distributions for different queries
        base_size = 100 + (i % len(query_templates)) * 50
        sizes = np.random.poisson(base_size, n_packets)
        
        # Time intervals
        times = np.cumsum(np.random.exponential(0.1, n_packets))
        
        # Directions (1 = outgoing, -1 = incoming)
        directions = np.random.choice([1, -1], n_packets)
        
        # Create DataFrame with new column names
        df = pd.DataFrame({
            'packet_id': range(n_packets),
            'timestamp': times,
            'length': sizes,
            'direction': directions,
            'burst_id': 1
        })
        
        # Add additional columns that separate_bursts produces
        df['gap_before'] = df['timestamp'].diff().fillna(float('inf'))
        df['gap_after'] = df['timestamp'].diff(periods=-1).abs().fillna(float('inf'))
        df['rel_time'] = df['timestamp'] - df['timestamp'].min()
        df['viz_bytes'] = df.apply(lambda x: x['length'] if x['direction'] == 1 else -x['length'], axis=1)
        df['cumulative_sum'] = df['viz_bytes'].cumsum()
        
        df.to_csv(filepath, index=False)
        generated_files.append(filepath)
    
    print(f"Generated {n_queries} synthetic burst CSV files in {output_dir}")
    return generated_files


def load_all_burst_files(data_dir=None, file_pattern='*.csv'):
    """
    Load all burst CSV files from a directory.
    
    Args:
        data_dir: Directory containing burst CSV files (default: SEPARATED_BURSTS_DIR)
        file_pattern: File pattern to match (default: '*.csv')
        
    Returns:
        List of tuples (DataFrame, label) for each file
    """
    if data_dir is None:
        data_dir = SEPARATED_BURSTS_DIR
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            df = load_burst_csv(filepath)
            label = get_label_from_filename(filepath)
            data.append((df, label))
    
    return data
