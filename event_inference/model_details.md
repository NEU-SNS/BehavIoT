# Detailed Descriptions for Content Analysis Models and Scripts

Below is a detailed description about the machine learning models, the files, and the directories in this section.

## Machine Learning Models to Detect Device Activity

### Machine Learning

During evaluation, we use following algorithms:
- RF:  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (supervised)
- DBSCAN: [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) (unsupervised)

## Steps
See [usage](usage.md)

## Scripts


Information about the contents of each of these files and directories can be found below.

### s1_decode_xxx.py

#### Usage

The script decodes data in pcap files (whose filenames are listed in a text file) into human-readable text files using tshark.

#### Input

`exp_list` - The text file that contains paths to input pcap files to generate the models. To see the format of this text file, please see the [traffic/](#traffic) section below.

`out_imd_dir` - The path to the directory where the script will create and put decoded pcap files. If this directory current does not exist, it will be generated.

`num_proc` - The number of processes to use to decode the pcaps. Default is 1.

#### Output

A plain-text file will be produced for every input pcap (.pcap) file. Each output file contains a translation of some of the fields in the raw input file into human-readable form. The raw data output is tab-delimited and is stored in a text (.txt) file at `{out_imd_dir}/{device}/{activity}/{filename}.txt` (see the [traffic/](#traffic) section below for an explanation of `device` and `activity`.


### s2_get_features.py

#### Usage

The script uses the decoded pcap data output to perform data analysis to get features.

#### Input

`in_dec_dir` - The path to a directory containing text files of human-readable raw pcap data.

`out_features_dir` - The path to the directory to write the analyzed CSV files. If this directory current does not exist, it will be generated.

`num_proc` - The number of processes to use to generate the feature files.

#### Output

Each valid input text (.txt) file in the input directory will be analyzed, and a CSV file containing statistical analysis will be produced in a `cache/` directory in `out_features_dir`. The name of this file is a sanitized version of the input file path. After each input file is processed, all the CSV files of each device will be concatenated together in a separate CSV file, named {device}.csv, which will be placed in `out_features_dir`.

### src/s4_xxx.py

### src/s5_xxx.py

### src/s6_xxx.py

The script trains analyzed pcap data and generates one or more models that can be used to predict device activity.
