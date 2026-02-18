

import pandas as pd
from scapy.layers.inet import IP
from scapy.utils import PcapReader

class TrafficLoader:
    def __init__(self, target_ip):
        self.target_ip = target_ip

    def pcap_to_csv(self, pcap_file, output_csv):
        traces = []
        start_time = None
        print(f"Starting to Processing {pcap_file} for IP: {self.target_ip}")
        with PcapReader(pcap_file) as packets:
            for pkt in packets:
                if IP in pkt:
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    
                    if self.target_ip == src_ip or self.target_ip == dst_ip:
                        current_time = float(pkt.time)
                        
                        if start_time is None:
                            start_time = current_time
                        
                        t = current_time - start_time
                        l = len(pkt)
                        
                        if src_ip == self.target_ip:
                            b = 1
                        else:
                            b = -1
                        
                        traces.append((t, l, b))

        if not traces:
            print("\n No packets found for the specified IP. Check your Target IP.")
            return None

    
        df = pd.DataFrame(traces, columns=['timestamp', 'length', 'direction'])
        
        df.to_csv(output_csv, index_label="packet_id")
        
        print(f"Successfully saved {len(traces)} packets to {output_csv}")
        print(f"Trace duration: {df['timestamp'].iloc[-1]:.4f} seconds")
        return df


# if __name__ == "__main__":
#     TARGET_IP = "192.168.86.40"  
#     INPUT_PCAP = "how_deep_is_the_indian_ocean_5_30s.pcap"
#     OUTPUT_CSV = "trace.csv"

#     loader = TrafficLoader(TARGET_IP)
    
#     try:
#         loader.pcap_to_csv(INPUT_PCAP, OUTPUT_CSV)
#     except FileNotFoundError:
#         print("Error: The specified .pcap file was not found.")