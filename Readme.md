# conversion: pcap file -> filtered trace of ip=192.168.86.40 --> seperated_bursts.csv

command: python3 main.py pcap_to_csv 192.168.86.40 how_deep_is_the_indian_ocean_5_30s.pcap

# train the doc2vec model with quora dataset

download the data from : https://www.kaggle.com/datasets/sambit7/first-quora-dataset and paste the csv file into the data/ folder.

command:

cd services
python training_doc2vec_model.py
