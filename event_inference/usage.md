# 0. Input files
## generate input: full path to PCAPs
To generate a list of file paths for the PCAP files in your dataset, use the following command in your terminal:
```
find /your-dataset-path > inputs/202x/xxx.txt 
```

# 1. Decoding
## hostname-IP mapping extraction from DNS and TLS 
We extract domain names from DNS and TLS handshakes.
To generate the hosename-IP mapping files, run `pipeline/s1_decode_dns_tls.py list_of_PCAPs.txt`. 
`list_of_PCAPs.txt` contains the list of PCAPs with  DNS and TLS handshake packets.  
```
python3 pipeline/s1_decode_dns_tls.py inputs/2021/idle_dns.txt
python3 pipeline/s1_decode_dns_tls.py inputs/2021/activity_dns.txt
```
## run decoding
This steps convert traffic in PCAPs into flow-bursts (defined in the paper) for each device. 
```
python3 pipeline/s1_decode_idle.py inputs/2021/idle-2021.txt data/idle-2021-decoded/ 8
python3 pipeline/s1_decode_activity.py inputs/2021/train.txt data/train-decoded/ 8
python3 pipeline/s1_decode_activity.py inputs/2021/test.txt data/test-decoded/ 8
```

# 2. Feature extraction
Extract the features (defined in the paper) from the decoded traffic.
```
python3 pipeline/s2_get_features.py data/idle-2021-decoded/ data/idle-2021-features/
python3 pipeline/s2_get_features.py data/train-decoded/ data/train-features/
python3 pipeline/s2_get_features.py data/test-decoded/ data/test-features/
```

## routine dataset
```
python3 pipeline/s1_decode_dns_tls.py inputs/2021/routine_dns.txt
python3 pipeline/s1_decode_activity.py inputs/2021/routine-dataset.txt data/routine-decoded/ 
python3 pipeline/s2_get_features.py data/routine-decoded/ data/routine-features/
```
## uncontrolled dataset. 
*Note that uncontrolled dataset is not included in our public datasets due to IRB constraints*)
```
python3 pipeline/s1_decode_dns_tls.py inputs/2022/uncontrolled_dns.txt
python3 pipeline/s1_decode_unctrl.py inputs/2022/uncontrolled_dataset.txt data/uncontrolled_decoded/  
```

# 3. Periodic traffic extraction
Extract the periods of periodic background traffic from idle datasets. 
```
cd period_extraction
python3 periodicity_inference.py
python3 fingerprint_generation.py
cd ..
```

# 4. Preprocessing
Preprocessing for ML-based inference.
```
python3 pipeline/s4_preprocess_feature_new.py -i data/idle-2021-features/ -o data/idle/
```
## preprocessing transform-only on uncontrolled datasets
```
python3 pipeline/s4_preprocess_feature_applyonly.py -i data/uncontrolled-features/ -o data/uncontrolled/
```

# 5. Periodic event inference and filtering 
## train
```
python3 pipeline/s5_periodic_filter.py -i data/idle-2021-train-std/ -o model/filter_apr20
```
## activity dataset 
Filter out periodic events from activity datasets for better user event classificaition performance.
```
python3 pipeline/s5_filter_by_periodic.py -i train -o model/filter
python3 pipeline/s5_filter_by_periodic.py -i test -o model/filter
```
## routine and uncontrolled dataset
Filters for routine and uncontrolled datasets. It uses both timing information and trained ML models for filtering
```
python3 pipeline/s5_periodic_time_filter.py -i data/routines-std/ -o model/time_filter
python3 pipeline/s5_filter_by_periodic_after_time.py -i routines -o model/filter
# python3 pipeline/s5_filter_by_periodic_after_time.py -i uncontrolled -o model/filter_may1
# python3 pipeline/s5_filter_by_periodic_after_time.py -i uncontrolled02 -o model/filter_may1
```

# 6. Activity (user event) inference
We've provided two options for user event inference: with and without hostnames. Based on our observations in the paper, the hostnames remain unchanged for most user events. However, there could be exceptions due to behavior changes or incomplete hostname-IP mappings. Therefore, we've implemented both methods. 
```
python3 pipeline/s6_activity_fingerprint.py -i data/train-filtered-std/ -o model/fingerprint/
python3 pipeline/s6_binary_model_whostname.py -i data/train-filtered-std/ -o model/binary
python3 pipeline/s6_binary_predict_whostname.py -i data/routines-filtered-std/ -o model/binary
```
or
```
python3 pipeline/s6_binary_model.py -i data/train-filtered-std/ -o model/binary
python3 pipeline/s6_binary_predict.py -i data/routines-filtered-std/ -o model/binary
```
# Periodic model score
## generate periodic event (background traffic) deviation scores from datasets
```
python3 pipeline/periodic_deviation_score.py -i data/idle-half-train-std/ -o model/time_score_newT_train_idle
python3 pipeline/periodic_score_analysis.py model/time_score_newT_train_idle model/time_score_newT_test_idle
```
