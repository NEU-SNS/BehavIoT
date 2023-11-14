import sys
import os

#script paths
PATH = sys.argv[0]
MODEL_DIR = os.path.dirname(PATH)
if MODEL_DIR == "":
    MODEL_DIR = "."
SRC_DIR = MODEL_DIR + "/src/"
RANDOM_STATE = 422

#output paths
OUT_DIR = "results/"
for i, arg in enumerate(sys.argv):
    if arg == "-o" and i + 1 < len(sys.argv):
        OUT_DIR = sys.argv[i + 1]
        break

#basics
RED = "\033[31;1m"
BLUE = "\033[36;1m"
END = "\033[0m"
BEG = RED + PATH + ": Error: "

#basic errors
NON_POS = BEG + "The number of processes must be a positive integer. Received \"%s\"." + END
WRONG_NUM_ARGS = BEG + "%d arguments required. %d arguments found." + END
MISSING = BEG + "The \"%s\" %s is missing.\n"\
          "    Please make sure it is in the \"%s\" directory." + END
NO_PERM = BEG + "The %s \"%s\" does not have %s permission." + END
INVAL = BEG + "%s \"%s\" is not a %s." + END
WRONG_EXT = BEG + "%s must be a %s file. Received \"%s\"" + END



#eval_model.py errors
NO_FEAT_DIR = BEG + "Features directory (-i) required." + END
NO_MOD_DIR = BEG + "Model directory (-o) required." + END

#s1_decocde_xxx.py usage
DEC_RAW_USAGE = """
Usage: python3 {prog_name} exp_list out_dec_dir [num_proc]

Decodes raw pcap data into human-readable text files.

Example: python3 {prog_name} inputs/2021/trace-dataset.txt data/trace-decoded/ 4

Arguments:
  exp_list:    a text file containing the file paths to pcap files to decode; pcap
                 paths must be formatted as .../{{device}}/{{activity}}/{{filename}}.pcap
  out_dec_dir: path to the directory to place the decoded output; directory will be
                 generated if it does not already exist
  num_proc:    number of processes to use to decode the pcaps (Default = 1)

For more information, see model_details.md.""".format(prog_name=PATH)

#s2_get_features.py usage
GET_FEAT_USAGE = """
Usage: python3 {prog_name} in_dec_dir out_features_dir [num_proc]

Performs statistical analysis on decoded pcap files to generate feature files.

Example: python3 {prog_name} decoded/us/ features/us/ 4

Arguments:
  in_dec_dir:   path to a directory containing text files of decoded pcap data
  out_feat_dir: path to the directory to write the analyzed CSV files;
                  directory will be generated if it does not already exist
  num_proc:     number of processes to use to generate feature files
                  (Default = 1)

For more information, see the README or model_details.md.""".format(prog_name=PATH)

#s4_xxx.py usage
PREPRO_USAGE = """
Preprocessing the extracted feature files.
Usage: python3 {prog_name} -i IN_FEATURES_DIR -o OUT_MODELS_DIR 
Example: python3 pipeline/s4_preprocess_feature_new.py -i data/idle-2021-features/ -o data/idle/""".format(prog_name=PATH)


#s5_xxx.py usage
PERIODIC_MOD_USAGE = """
Periodic event inference and filtering
Usage: python3 {prog_name} -i IN_FEATURES_DIR -o OUT_MODELS_DIR 
Example: see usage.md""".format(prog_name=PATH)

#s6_xxx.py usage
PREDICT_MOD_USAGE = """
User event inference
Usage: python3 {prog_name} -i IN_FEATURES_DIR -o OUT_MODELS_DIR 
Example: see usage.md""".format(prog_name=PATH)
