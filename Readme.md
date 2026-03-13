# Voice Command Fingerprinting - Attack and Defense

This project implements the VCFP (Voice Command Fingerprint) attack and BuFLO defense for analyzing network traffic from voice assistants. It demonstrates how traffic analysis can be used to identify voice commands from encrypted network packets and how the BuFLO (Brute-force Load Balancing Obfuscation) defense can mitigate such attacks.

## Overview

The VCFP attack analyzes packet timing and size patterns in network traffic to identify which voice command was issued to a smart speaker. The BuFLO defense normalizes traffic patterns to prevent such attacks by enforcing fixed packet sizes and transmission rates.

## Installation/Setup

### Requirements

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:

- **scapy** - Packet manipulation and PCAP file processing
- **pandas** - Data manipulation and CSV processing
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning classifiers
- **plotly** - Visualization (for analysis)

## Project Structure

```
voice_command_fingerprinting/
├── main.py                 # Main orchestration file
├── requirements.txt        # Python dependencies
├── attacks/                # VCFP attack implementation
│   ├── classifiers.py      # Bayes, SVM classifiers
│   ├── cross_validation.py # N-fold cross-validation
│   ├── dataset.py         # Dataset loading utilities
│   └── features.py        # Feature extraction
├── defences/               # Defense implementations
│   └── buflo.py           # BuFLO defense
├── services/               # Utility services
│   ├── pcap_to_csv.py     # PCAP to CSV conversion
│   ├── seperate_bursts.py # Burst separation
│   └── data_loader.py     # Data loading utilities
└── data/                   # Data directories
    ├── captured_files/     # Original PCAP files
    ├── trace_csv_files/   # Processed CSV traces
    ├── seperated_bursts_files/ # Burst-separated data
    ├── buflo/             # BuFLO output
    └── synthetic_bursts/ # Synthetic test data
```

## Usage/Commands

### 1. Convert PCAP to CSV

Convert a PCAP file to burst-separated CSV format for analysis.

```bash
python3 main.py pcap_to_csv <target_ip> <pcap_file>
```

**Example:**

```bash
python3 main.py pcap_to_csv 192.168.86.40 how_deep_is_the_indian_ocean_5_30s.pcap
```

This command:
- Filters traffic by target IP address
- Converts packet data to CSV format
- Separates traffic into bursts (based on 1-second gap threshold)
- Outputs to `data/seperated_bursts_files/`

### 2. Apply BuFLO Defense

Apply BuFLO defense to a burst CSV file to normalize traffic patterns.

```bash
python3 main.py apply_buflo <input_csv> [--d packet_size] [--f frequency] [--t time]
```

**Parameters:**

| Flag | Parameter | Description | Default |
|------|-----------|-------------|---------|
| `--d` | packet_size | Fixed packet size in bytes | 600 |
| `--f` | frequency | Packet transmission frequency (ms interval) | 50 |
| `--t` | min_time | Minimum transmission time (seconds) | 20 |

**Example:**

```bash
# Using default parameters
python3 main.py apply_buflo data/seperated_bursts_files/good_morning_1_bursts.csv

# Using custom parameters
python3 main.py apply_buflo data/seperated_bursts_files/good_morning_1_bursts.csv --d 1000 --f 100 --t 30
```

### 3. Run Attack

Run the VCFP attack on a directory of burst data.

```bash
python3 main.py run_attack <data_dir> [--method bayes|svm|jaccard] [--folds n] [--interval n]
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--method` | Classification method: `bayes`, `svm`, or `jaccard` | bayes |
| `--folds` | Number of cross-validation folds | 5 |
| `--interval` | Feature extraction interval | 50 |

**Examples:**

```bash
# Run attack with Naive Bayes
python3 main.py run_attack data/seperated_bursts_files --method bayes --folds 5

# Run attack with SVM
python3 main.py run_attack data/seperated_bursts_files --method svm --folds 10

# Run attack with Jaccard similarity
python3 main.py run_attack data/seperated_bursts_files --method jaccard --folds 5
```

### 4. Complete Workflow

Run the complete attack-defense workflow, comparing undefended vs. BuFLO-protected traffic.

```bash
python3 main.py workflow <data_dir> [--method method] [--buflo] [--folds n] [--d size] [--f freq] [--t time]
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--method` | Classification method (`bayes`, `svm`, `jaccard`) | bayes |
| `--buflo` | Apply BuFLO defense before attack | false |
| `--folds` | Number of cross-validation folds | 5 |
| `--d` | BuFLO packet size | 600 |
| `--f` | BuFLO frequency | 50 |
| `--t` | BuFLO minimum time | 20 |

**Examples:**

```bash
# Run attack without defense
python3 main.py workflow data/seperated_bursts_files --method bayes --folds 5

# Run attack with BuFLO defense
python3 main.py workflow data/seperated_bursts_files --method bayes --folds 5 --buflo
```

### 5. Generate Synthetic Data

Generate synthetic burst data for testing purposes.

```bashpython3 main.py generate_data [--n_queries n] [--packets p]
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--n_queries` | Number of query samples to generate | 10 |
| `--packets` | Average number of packets per query | 50 |

**Example:**

```bash
python3 main.py generate_data --n_queries 20 --packets 100
```

### 6. List Available Data

List all available data files in the project.

```bash
python3 main.py list_data
```

## BuFLO Parameters Explained

BuFLO (Brute-force Load Balancing Obfuscation) is a traffic analysis defense that normalizes network traffic patterns. The key parameters are:

### 1. Packet Size (d)
- **Flag:** `--d`
- **Description:** Fixed packet size in bytes
- **Effect:** All packets are padded or truncated to this size, removing information about original packet sizes
- **Default:** 600 bytes
- **Higher values** provide stronger defense but consume more bandwidth

### 2. Frequency (f)
- **Flag:** `--f`
- **Description:** Packet transmission interval in milliseconds
- **Effect:** Packets are sent at fixed intervals, removing timing information
- **Default:** 50 (one packet every 50ms = 20 packets/second)
- **Lower values** (more frequent) provide stronger defense but more overhead

### 3. Minimum Time (t)
- **Flag:** `--t`
- **Description:** Minimum transmission time in seconds
- **Effect:** Transmission continues for at least this duration, adding dummy traffic if needed
- **Default:** 20 seconds
- **Higher values** provide stronger defense but increase latency and bandwidth

## Attack Methods

The VCFP attack supports three classification methods:

### 1. Naive Bayes (`bayes`)
- Uses probabilistic classification based on packet timing features
- Fast and works well with limited training data
- Best for: Quick experiments, baseline comparisons

### 2. Support Vector Machine (`svm`)
- Uses SVM with RBF kernel for classification
- Better at capturing non-linear patterns
- Best for: Higher accuracy with larger datasets

### 3. Jaccard Similarity (`jaccard`)
- Uses set-based similarity between packet bursts
- Treats packets as categorical items
- Best for: Commands with distinct packet patterns

## Data Directories

The `data/` folder contains several subdirectories:

### `data/captured_files/`
Contains original PCAP (packet capture) files from voice assistant traffic.

### `data/trace_csv_files/`
Contains processed CSV files with filtered network traces (one file per PCAP).

### `data/seperated_bursts_files/`
Contains burst-separated data where network traffic is divided into bursts (clusters of packets with small inter-packet gaps).

### `data/buflo/`
Contains output from BuFLO defense - processed traffic after applying the defense.

### `data/synthetic_bursts/`
Contains synthetically generated test data for experimentation.

## Example Workflow

Here's a complete example workflow:

```bash
# 1. List available data
python3 main.py list_data

# 2. Run attack on existing data
python3 main.py run_attack data/seperated_bursts_files --method bayes --folds 5

# 3. Run complete workflow with BuFLO defense
python3 main.py workflow data/seperated_bursts_files --method bayes --buflo --folds 5
```

## Output Interpretation

The attack results include:

- **Accuracy**: Percentage of correctly classified voice commands
- **Rank**: Average rank of correct command in sorted predictions (lower is better)
- **Fold accuracies**: Individual fold results for cross-validation stability

The workflow comparison shows:
- **Undefended Data**: Attack accuracy on original traffic
- **BuFLO Defended Data**: Attack accuracy after applying BuFLO
- **Defense Impact**: Difference in accuracy/rank between defended and undefended

## References

This implementation is based on the research paper:
- "I Can Hear Your Alexa Voice Command Fingerprinting on Smart Home Speakers" 
- See `docs/` folder for the full paper

## License

This project is for educational and research purposes.
