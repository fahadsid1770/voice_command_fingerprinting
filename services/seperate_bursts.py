import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


GAP_THRESHOLD = 1.0 
MIN_PACKETS_PER_BURST = 10

def separate_bursts(csv_file, gap_threshold):
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['gap_before'] = df['timestamp'].diff().fillna(float('inf')) # gap_before[i] = time since previous packet
    df['gap_after'] = df['timestamp'].diff(periods=-1).abs().fillna(float('inf')) # gap_after[i] = time until next packet
    
    #IDing the bursts
    burst_id = 0
    burst_ids = []
    in_burst = False
    
    for idx, row in df.iterrows():
        gb = row['gap_before']
        ga = row['gap_after']
        
        if gb >= gap_threshold and ga < gap_threshold: #gap before > 1s, gap after < 1s
            burst_id += 1
            in_burst = True
            burst_ids.append(burst_id)
            
        elif gb < gap_threshold and ga >= gap_threshold: #gap before < 1s, gap after > 1s
            if in_burst:
                burst_ids.append(burst_id)
                in_burst = False
            else:
                burst_ids.append(0)
                
        elif gb >= gap_threshold and ga >= gap_threshold: #Gap > 1s on both sides
            in_burst = False
            burst_ids.append(0)
            
        #gap > 1s on both sides
        else: 
            if in_burst: 
                burst_ids.append(burst_id)
            else:
                burst_ids.append(0)
                
    df['burst_id'] = burst_ids
    df_clean = df[df['burst_id'] > 0].copy() # noice filtering
    
    #filtering Keep-alives
    burst_counts = df_clean['burst_id'].value_counts()
    valid_bursts = burst_counts[burst_counts >= MIN_PACKETS_PER_BURST].index
    df_clean = df_clean[df_clean['burst_id'].isin(valid_bursts)].copy()
    
    df_clean['rel_time'] = df_clean.groupby('burst_id')['timestamp'].transform(lambda x: x - x.min()) # adding a relative time column
    
    df_clean['viz_bytes'] = df_clean.apply(lambda x: x['length'] if x['direction'] == 1 else -x['length'], axis=1) #adding a cumulative bytes
    df_clean['cumulative_sum'] = df_clean.groupby('burst_id')['viz_bytes'].cumsum()

    print(f"Found {df_clean['burst_id'].nunique()} valid voice commands.")
    return df_clean





# df_bursts = separate_bursts(INPUT_FILE, GAP_THRESHOLD)
# df_bursts.head()
# list_of_burst = df_bursts["burst_id"].unique().tolist()
# print(f"There are {len(list_of_burst)} bursts, and The ids are : {', '.join(map(str, list_of_burst))}")
# df_bursts.to_csv("working_bursts.csv")