# Event Inferences

## Usage:
[usage](usage.md)

## Folder structure of event inference
### pipeline
Contains all scripts for processing data and machine learning model training and testing.

### inputs
The inputs that need to be generated before executing scripts

### period_extraction
Contains scripts for extracting period from idle traffic.

### model
A directory to store the trained machine learning models.

### ip_hosts
Stores ip-host mappings extracted from DNS or TLS traffic.

### data
Stores processed data from PCAPs 

### logs
Designated for any logging files created

### results
Results generated

## Requirements
The scripts are tested on Linux version 5.4.0-169-generic (Ubuntu 9.4.0-1ubuntu1~20.04.2) with python 3.7.6
```
pip3 install numpy pandas scipy ipaddress statsmodels sklearn matplotlib
```
## Acknowledgement
We thank the following students for their contributions to the project: Abhijit Menon, Derek Ng, Shu Zhang
