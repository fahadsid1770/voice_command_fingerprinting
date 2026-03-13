"""
Main orchestration file for voice command fingerprinting project.

This module integrates all components from services/, attacks/, and defences/
modules to provide a complete workflow for voice command fingerprinting
attack and defense experiments.
"""

import sys
import os
import argparse

from services.pcap_to_csv import TrafficLoader
from services.seperate_bursts import separate_bursts
from services.data_loader import (
    SEPARATED_BURSTS_DIR, 
    BUFLO_OUTPUT_DIR,
    generate_synthetic_burst_data,
    load_all_burst_files
)
from defences.buflo import apply_buflo_to_file
from attacks.dataset import load_dataset_from_bursts
from attacks.cross_validation import n_fold_cross_validation
from attacks.classifiers import train_bayes, train_svm


def find_pcap_file(filename):
    """
    Find PCAP file in data directories.
    
    Args:
        filename: Name of the PCAP file to find
        
    Returns:
        Full path to the PCAP file if found, None otherwise
    """
    paths = [
        f"data/captured_files/{filename}",
        f"data/{filename}",
        filename,
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def compare_results(results):
    """
    Print comparison of undefended vs defended attack results.
    
    Args:
        results: Dictionary containing 'undefended' and/or 'defended' results
                 Each result should have 'accuracy' and 'rank' keys
    """
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    if 'undefended' in results:
        print(f"\nUndefended Data:")
        print(f"  Accuracy: {results['undefended']['accuracy']:.4f}")
        print(f"  Rank: {results['undefended']['rank']:.4f}")
    
    if 'defended' in results:
        print(f"\nBuFLO Defended Data:")
        print(f"  Accuracy: {results['defended']['accuracy']:.4f}")
        print(f"  Rank: {results['defended']['rank']:.4f}")
    
    if 'undefended' in results and 'defended' in results:
        acc_diff = results['defended']['accuracy'] - results['undefended']['accuracy']
        rank_diff = results['defended']['rank'] - results['undefended']['rank']
        print(f"\nDefense Impact:")
        print(f"  Accuracy Change: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
        print(f"  Rank Change: {rank_diff:+.4f}")
    
    print("=" * 60)


def cmd_pcap_to_csv(target_ip, pcap_filename):
    """
    Convert PCAP to burst-separated CSV.
    
    Args:
        target_ip: Target IP address to filter traffic
        pcap_filename: Name of the PCAP file
    """
    pcap_path = find_pcap_file(pcap_filename)
    if not pcap_path:
        print(f"PCAP file not found: {pcap_filename}")
        return
    
    base_name = os.path.splitext(pcap_filename)[0]
    trace_csv = f"data/trace_csv_files/{base_name}.csv"
    bursts_csv = f"data/seperated_bursts_files/{base_name}_bursts.csv"
    
    print(f"Processing PCAP file: {pcap_path}")
    print(f"Target IP: {target_ip}")
    
    # Convert PCAP to CSV
    loader = TrafficLoader(target_ip)
    loader.pcap_to_csv(pcap_path, trace_csv)
    
    # Separate bursts
    print(f"\nSeparating bursts...")
    df = separate_bursts(trace_csv, gap_threshold=1.0)
    burst_counts = df['burst_id'].value_counts().sort_index()
    for burst_id, count in burst_counts.items():
        print(f"  burst_id: {burst_id}, packet count: {count}")
    
    df.to_csv(bursts_csv, index=False)
    print(f"\nSaved bursts to {bursts_csv}")


def cmd_apply_buflo(input_csv, packet_size=600, frequency=50, min_time=20):
    """
    Apply BuFLO defense to a burst CSV file.
    
    Args:
        input_csv: Path to input burst CSV file
        packet_size: Fixed packet size (d parameter)
        frequency: Packet transmission frequency (f parameter)
        min_time: Minimum transmission time (t parameter)
    """
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return
    
    print(f"Applying BuFLO defense to: {input_csv}")
    print(f"  Packet size (d): {packet_size}")
    print(f"  Frequency (f): {frequency}")
    print(f"  Min time (t): {min_time}")
    
    output_path = apply_buflo_to_file(
        input_csv, 
        d=packet_size, 
        f=frequency, 
        t=min_time
    )
    
    if output_path:
        print(f"\nBuFLO defense applied successfully!")
        print(f"Output saved to: {output_path}")


def cmd_run_attack(data_dir, method='bayes', n_folds=5, interval=50):
    """
    Run VCFP attack on burst data.
    
    Args:
        data_dir: Directory containing burst CSV files
        method: Classification method ('bayes', 'svm', 'jaccard')
        n_folds: Number of cross-validation folds
        interval: Interval for feature extraction
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Map classifier method to feature method
    # SVM uses bayes features, jaccard uses jaccard features, bayes uses bayes features
    if method == 'svm':
        feature_method = 'bayes'
    else:
        feature_method = method
    
    print(f"Running VCFP attack...")
    print(f"  Data directory: {data_dir}")
    print(f"  Method: {method}")
    print(f"  Cross-validation folds: {n_folds}")
    print(f"  Feature interval: {interval}")
    
    # Load dataset
    X, y, label_map = load_dataset_from_bursts(data_dir, feature_method, interval)
    
    if len(X) == 0:
        print("Error: No data loaded")
        return
    
    print(f"\nDataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Classes: {len(label_map)}")
    print(f"  Labels: {list(label_map.keys())}")
    
    # Run cross-validation
    print(f"\nRunning {n_folds}-fold cross-validation...")
    accuracy, rank, fold_accuracies = n_fold_cross_validation(
        X, y, n_folds=n_folds, method=method
    )
    
    print("\n" + "=" * 60)
    print("ATTACK RESULTS")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Average Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Rank: {rank:.4f}")
    print(f"Fold accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
    print("=" * 60)


def cmd_workflow(data_dir, method='bayes', n_folds=5, use_buflo=False, 
                 packet_size=600, frequency=50, min_time=20):
    """
    Run complete attack-defense workflow.
    
    Args:
        data_dir: Directory containing burst CSV files
        method: Classification method ('bayes', 'svm', 'jaccard')
        n_folds: Number of cross-validation folds
        use_buflo: Whether to apply BuFLO defense
        packet_size: BuFLO packet size parameter
        frequency: BuFLO frequency parameter
        min_time: BuFLO minimum time parameter
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    results = {}
    
    # Run attack on undefended data
    print("=" * 60)
    print("PHASE 1: Running attack on UNDEFENDED data")
    print("=" * 60)
    
    # Map classifier method to feature method
    if method == 'svm':
        feature_method = 'bayes'
    else:
        feature_method = method
    
    X, y, label_map = load_dataset_from_bursts(data_dir, feature_method)
    
    if len(X) == 0:
        print("Error: No data loaded from undefended directory")
        return
    
    print(f"Dataset: {len(X)} samples, {len(label_map)} classes")
    accuracy, rank, _ = n_fold_cross_validation(X, y, n_folds=n_folds, method=method)
    results['undefended'] = {'accuracy': accuracy, 'rank': rank}
    
    # Run attack on defended data if requested
    if use_buflo:
        print("\n" + "=" * 60)
        print("PHASE 2: Applying BuFLO defense")
        print("=" * 60)
        print(f"BuFLO parameters: d={packet_size}, f={frequency}, t={min_time}")
        
        # Apply BuFLO to all CSV files in the directory
        defended_dir = BUFLO_OUTPUT_DIR
        os.makedirs(defended_dir, exist_ok=True)
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            input_path = os.path.join(data_dir, csv_file)
            print(f"\nProcessing: {csv_file}")
            apply_buflo_to_file(input_path, packet_size, frequency, min_time)
        
        print("\n" + "=" * 60)
        print("PHASE 3: Running attack on DEFENDED data")
        print("=" * 60)
        
        # Run attack on defended data
        X_def, y_def, label_map_def = load_dataset_from_bursts(defended_dir, method)
        
        if len(X_def) == 0:
            print("Error: No data loaded from defended directory")
        else:
            print(f"Dataset: {len(X_def)} samples, {len(label_map_def)} classes")
            accuracy_def, rank_def, _ = n_fold_cross_validation(
                X_def, y_def, n_folds=n_folds, method=method
            )
            results['defended'] = {'accuracy': accuracy_def, 'rank': rank_def}
    
    # Compare results
    compare_results(results)


def cmd_generate_data(n_queries=10, packets=50):
    """
    Generate synthetic test data.
    
    Args:
        n_queries: Number of query samples to generate
        packets: Average number of packets per query
    """
    output_dir = "data/synthetic_bursts"
    
    print(f"Generating synthetic burst data...")
    print(f"  Number of queries: {n_queries}")
    print(f"  Packets per query: {packets}")
    print(f"  Output directory: {output_dir}")
    
    generated_files = generate_synthetic_burst_data(output_dir, n_queries, packets)
    
    print(f"\nGenerated {len(generated_files)} files:")
    for f in generated_files[:5]:
        print(f"  - {f}")
    if len(generated_files) > 5:
        print(f"  ... and {len(generated_files) - 5} more")


def cmd_list_data():
    """List available data files in all data directories."""
    print("Available data files:")
    print("=" * 60)
    
    directories = {
        "Captured PCAP files": "data/captured_files",
        "Trace CSV files": "data/trace_csv_files",
        "Separated bursts": "data/seperated_bursts_files",
        "BuFLO output": "data/buflo",
    }
    
    for desc, directory in directories.items():
        print(f"\n{desc} ({directory}):")
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if files:
                for f in sorted(files):
                    filepath = os.path.join(directory, f)
                    size = os.path.getsize(filepath)
                    print(f"  - {f} ({size:,} bytes)")
            else:
                print("  (empty)")
        else:
            print("  (directory does not exist)")


def main():
    """Main entry point for the CLI."""
    # Handle --help/-h flags before checking command
    if len(sys.argv) >= 2 and sys.argv[1] in ['--help', '-h']:
        print("Usage: python3 main.py <command> [args]")
        print("\nCommands:")
        print("  pcap_to_csv    - Convert PCAP to burst-separated CSV")
        print("  apply_buflo    - Apply BuFLO defense to burst CSV")
        print("  run_attack     - Run VCFP attack on burst data")
        print("  workflow       - Run complete attack-defense workflow")
        print("  generate_data  - Generate synthetic test data")
        print("  list_data      - List available data files")
        print("\nFor help on a specific command, run:")
        print("  python3 main.py <command> --help")
        sys.exit(0)
    
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <command> [args]")
        print("\nCommands:")
        print("  pcap_to_csv    - Convert PCAP to burst-separated CSV")
        print("  apply_buflo    - Apply BuFLO defense to burst CSV")
        print("  run_attack     - Run VCFP attack on burst data")
        print("  workflow       - Run complete attack-defense workflow")
        print("  generate_data  - Generate synthetic test data")
        print("  list_data      - List available data files")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Create parsers for each command
    if command == "pcap_to_csv":
        parser = argparse.ArgumentParser(description='Convert PCAP to burst-separated CSV')
        parser.add_argument('target_ip', help='Target IP address')
        parser.add_argument('pcap_filename', help='PCAP filename')
        args = parser.parse_args(sys.argv[2:])
        cmd_pcap_to_csv(args.target_ip, args.pcap_filename)
    
    elif command == "apply_buflo":
        parser = argparse.ArgumentParser(description='Apply BuFLO defense to burst CSV')
        parser.add_argument('input_csv', help='Input burst CSV file')
        parser.add_argument('--d', dest='packet_size', type=int, default=600,
                           help='Fixed packet size (default: 600)')
        parser.add_argument('--f', dest='frequency', type=int, default=50,
                           help='Packet transmission frequency (default: 50)')
        parser.add_argument('--t', dest='min_time', type=int, default=20,
                           help='Minimum transmission time in seconds (default: 20)')
        args = parser.parse_args(sys.argv[2:])
        cmd_apply_buflo(args.input_csv, args.packet_size, args.frequency, args.min_time)
    
    elif command == "run_attack":
        parser = argparse.ArgumentParser(description='Run VCFP attack on burst data')
        parser.add_argument('data_dir', help='Directory containing burst CSV files')
        parser.add_argument('--method', choices=['bayes', 'svm', 'jaccard'], 
                           default='bayes', help='Classification method (default: bayes)')
        parser.add_argument('--folds', type=int, default=5,
                           help='Number of cross-validation folds (default: 5)')
        parser.add_argument('--interval', type=int, default=50,
                           help='Feature extraction interval (default: 50)')
        args = parser.parse_args(sys.argv[2:])
        cmd_run_attack(args.data_dir, args.method, args.folds, args.interval)
    
    elif command == "workflow":
        parser = argparse.ArgumentParser(description='Run complete attack-defense workflow')
        parser.add_argument('data_dir', help='Directory containing burst CSV files')
        parser.add_argument('--method', choices=['bayes', 'svm', 'jaccard'],
                           default='bayes', help='Classification method (default: bayes)')
        parser.add_argument('--folds', type=int, default=5,
                           help='Number of cross-validation folds (default: 5)')
        parser.add_argument('--buflo', action='store_true',
                           help='Apply BuFLO defense')
        parser.add_argument('--d', dest='packet_size', type=int, default=600,
                           help='BuFLO packet size (default: 600)')
        parser.add_argument('--f', dest='frequency', type=int, default=50,
                           help='BuFLO frequency (default: 50)')
        parser.add_argument('--t', dest='min_time', type=int, default=20,
                           help='BuFLO minimum time (default: 20)')
        args = parser.parse_args(sys.argv[2:])
        cmd_workflow(args.data_dir, args.method, args.folds, 
                    args.buflo, args.packet_size, args.frequency, args.min_time)
    
    elif command == "generate_data":
        parser = argparse.ArgumentParser(description='Generate synthetic test data')
        parser.add_argument('--n_queries', type=int, default=10,
                           help='Number of query samples (default: 10)')
        parser.add_argument('--packets', type=int, default=50,
                           help='Average packets per query (default: 50)')
        args = parser.parse_args(sys.argv[2:])
        cmd_generate_data(args.n_queries, args.packets)
    
    elif command == "list_data":
        cmd_list_data()
    
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
