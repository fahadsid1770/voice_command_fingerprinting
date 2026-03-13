"""
BuFLO (Bursting Flow) defense implementation for voice command fingerprinting.

BuFLO is a defensive mechanism that:
- Sends fixed-length packets at regular intervals
- Pads small packets to a fixed size
- Chops large packets into fixed-size chunks
- Adds dummy packets to meet minimum transmission time requirements
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

from services.data_loader import BUFLO_OUTPUT_DIR, load_burst_csv


def apply_buflo_to_burst(df_burst, d=600, f=50, t=20):
    """
    Apply BuFLO countermeasure to a single burst.
    
    Args:
        df_burst: DataFrame with burst data (columns: timestamp, length, direction, burst_id)
        d: Fixed packet size (default: 600 bytes)
        f: Frequency of packet transmission (packets per second)
        t: Minimum transmission time (in seconds)
    
    Returns:
        DataFrame with columns: index, timestamp, length, direction, overhead, status, burst_id
        Status values: 'padded', 'chopped', 'dummy'
    """
    if len(df_burst) == 0:
        return pd.DataFrame()
    
    buflo_data = []
    start_t = df_burst['timestamp'].min()
    end_time = df_burst['timestamp'].max()
    
    index = 0
    total_packet = int(t * f)  # Minimum packets needed
    total_overhead = 0
    
    # Process each packet in the burst
    for _, row in df_burst.iterrows():
        size = row['length']
        direction = row['direction']
        
        if size <= d:
            # Pad small packets
            overhead = d - size
            total_overhead += overhead
            new_p = {
                'index': index,
                'timestamp': round(start_t + index * (1 / f), 6),
                'length': d,
                'direction': direction,
                'overhead': overhead,
                'status': 'padded'
            }
            buflo_data.append(new_p)
            index += 1
        else:
            # Chop large packets
            remaining = size
            while remaining > 0:
                chunk_size = min(d, remaining)
                overhead = d - chunk_size if remaining <= d else 0
                total_overhead += overhead
                new_p = {
                    'index': index,
                    'timestamp': round(start_t + index * (1 / f), 6),
                    'length': chunk_size,
                    'direction': direction,
                    'overhead': overhead,
                    'status': 'chopped'
                }
                buflo_data.append(new_p)
                remaining -= chunk_size
                index += 1
    
    # Add dummy packets to meet minimum time requirement
    if index < total_packet:
        for i in range(index, total_packet):
            seed = -1 + 2 * random.random()
            direction = float(np.sign(seed))
            dummy_packet = {
                'index': i,
                'timestamp': round(start_t + (i + 1) * (1 / f), 6),
                'length': d,
                'direction': direction,
                'overhead': d,
                'status': 'dummy'
            }
            total_overhead += d
            buflo_data.append(dummy_packet)
    
    # Create DataFrame
    df_buflo = pd.DataFrame(buflo_data)
    
    time_delay = df_buflo['timestamp'].max() - end_time if len(df_buflo) > 0 else 0
    
    print(f"  Time delay: {time_delay:.2f}s")
    print(f"  Total overhead: {total_overhead} bytes")
    print(f"  Original packets: {len(df_burst)}, BuFLO packets: {len(df_buflo)}")
    
    return df_buflo


def apply_buflo_to_file(csv_path, d=600, f=50, t=20, output_dir=None):
    """
    Apply BuFLO to all bursts in a CSV file.
    
    Args:
        csv_path: Path to the burst CSV file
        d: Fixed packet size (default: 600 bytes)
        f: Frequency of packet transmission (packets per second)
        t: Minimum transmission time (in seconds)
        output_dir: Output directory (default: BUFLO_OUTPUT_DIR from services.data_loader)
    
    Returns:
        Path to the defended CSV file
    """
    if output_dir is None:
        output_dir = BUFLO_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the burst CSV
    df = load_burst_csv(csv_path)
    
    if len(df) == 0:
        print("Error: No data in CSV file")
        return None
    
    # Get base name
    base_name = Path(csv_path).stem
    
    # Get unique burst IDs
    burst_ids = df['burst_id'].unique()
    
    all_buflo_data = []
    
    print(f"Applying BuFLO to {len(burst_ids)} bursts...")
    
    for burst_id in sorted(burst_ids):
        df_burst = df[df['burst_id'] == burst_id].copy()
        df_buflo = apply_buflo_to_burst(df_burst, d, f, t)
        df_buflo['burst_id'] = burst_id
        all_buflo_data.append(df_buflo)
    
    # Combine all bursts
    df_combined = pd.concat(all_buflo_data, ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{base_name}_buflo.csv")
    df_combined.to_csv(output_path, index=False)
    
    print(f"BuFLO applied: {base_name}")
    print(f"  Output: {output_path}")
    
    return output_path
