# PFSM building and PFSM-based deviation analysis

## Requirement: install Synoptic for FSM construction
```
Ivan Beschastnikh, Yuriy Brun, Sigurd Schneider, Michael Sloan, and Michael D Ernst. 2011. Leveraging existing instrumentation to automatically infer invariantconstrained models. In Proceedings of the 19th ACM SIGSOFT symposium and the 13th European conference on Foundations of software engineering. 267–277.
```
The scripts are tested on Linux version 5.4.0-169-generic (Ubuntu 9.4.0-1ubuntu1~20.04.2) with python 3.7.6.
```
pip install networkx pydot numpy scipy matplotlib
```
## Event traces
- input: traces/log_routines_xx/
- log processor: convertor_new.py: generate log files for Synoptic - uncontrolled dataset
- convertor_trace_dataset.py: generate synthetic (for evaluting deviation scores) datasets and 5-fold training and testing sets


## Build PFSM
`bash synoptic.sh -o output/pfsm --dumpInvariants=True -r '(?<TYPE>.+),(?<DTIME>.+)$' -s '^------$' --outputCountLabels=True --outputProbLabels=True --outputSupportCount=True --ignoreNFbyInvs=True --supportCountThreshold=2 logs/trace_may1`

Invariants: AlwaysFollowedBy and AlwaysPrecedes

## Deviation measurement and detection
```
python3 state_machine_builder/state_machine_read.py output/pfsm traces/trace_may1
python3 state_machine_builder/state_machine_read.py output/pfsm traces/trace_train_may1
python3 state_machine_builder/state_machine_read.py output/pfsmtraces/trace_test_may1
```
output/pfsm: the generated FSM file from Synoptic

## Others
- synthetic_fn_analysis.py: edit event traces on routine dataset (add or alter)
- train_test_split.py: convert routine logs to training and testing sets

