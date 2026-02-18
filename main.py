import sys
import os
from services.pcap_to_csv import TrafficLoader
from services.seperate_bursts import separate_bursts

def find_pcap_file(filename):
    paths = [
        f"data/captured_files/{filename}",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def cmd_pcap_to_csv(target_ip, pcap_filename):
    pcap_path = find_pcap_file(pcap_filename)
    if not pcap_path:
        print(f"PCAP file not found: {pcap_filename}")
        return
    base_name = os.path.splitext(pcap_filename)[0]
    trace_csv = f"data/trace_csv_files/{base_name}.csv"
    bursts_csv = f"data/seperated_bursts_files/{base_name}_bursts.csv"
    loader = TrafficLoader(target_ip)
    loader.pcap_to_csv(pcap_path, trace_csv)
    df = separate_bursts(trace_csv, gap_threshold=1.0)
    burst_counts = df['burst_id'].value_counts().sort_index()
    for burst_id, count in burst_counts.items():
        print(f"burst_id: {burst_id}, count: {count}")
    df.to_csv(bursts_csv, index=False)
    print(f"Saved bursts to {bursts_csv}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <command> [args]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "pcap_to_csv":
        if len(sys.argv) != 4:
            print("Usage: python3 main.py pcap_to_csv <target_ip> <pcap_filename>")
            sys.exit(1)
        cmd_pcap_to_csv(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

