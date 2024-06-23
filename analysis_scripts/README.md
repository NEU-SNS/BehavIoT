## Destination analysis per event type.

1. run 1_get_domains_from_feature_csv to get domain name lists
2. run 2_getorg to get org and party for each domain
3. run 3_getStats to collectively save numbers in one file
4. plotting 

Please download the non-essential and essential destination domain list from IoTrimmer paper. 

## Notes
2_getorg relies on Whois. However, Whois and its python wrapper are not reliable all the time. Manual effort is often needed. 
periods_imc_analysis.ipynb: get statistics of peirodic models 
