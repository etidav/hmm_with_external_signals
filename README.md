# HERMES: Discrete hidden Markov model with inclusion of external signals

Authors: Etienne DAVID, Sylvain LE CORFF and Jean BELLOT

Paper link: 

### Abstract
> 

## Code Organisation

This repository provides the code of the hidden Markov model with the inclusion of external signals and a simple code base to reproduce the results presented in this [paper](). The repository is organized as follow:

 - [hmm_model](hmm_model\): Directory gathering the code of the HMMs and all the available variations (hmm, shmm, arhmm, arshmm, hmmes, shmmes, arhmmes, arshmmes)
 - [run](run\): Directory gathering the different scipt tro reproduced the result of the paper.
 - [data](data\): Directory gathering the 10 fashion time series introduced in this paper [paper]() and used in the experimental section of the HMM with external signals paper.
 - [docker](docker\): directory gathering the code to reproduce a docker so as to recover the exact result of the paper.  

## Reproduce benchmark results

First, you should build the docker. In the docker folder, run
```bash
make ...
```

To reproduce the result on simulated sequences :


- [run_simulated_sequence.py](run\simulated_sequence\..) (make sure you are in the docker environement.

```bash
python benchmark.py --help # display the default parameters and their description
python benchmark.py # run the benchmark on an 
python benchmark --dataset-path DATASET_PATH --model-names snaive ... # run the benchmark on DATASET_PATH with only snaive
python benchmark --dataset-path DATASET_PATH --model-names ets ... # run the benchmark on DATASET_PATH with only ets
python benchmark --dataset-path DATASET_PATH --model-names snaive --model-names ets ... # run the benchmark on DATASET_PATH with ets and snaive
```
- [run_simulated_sequence.py](run\simulated_sequence\..) (make sure you are in the docker environement.

```bash
python benchmark.py --help # display the default parameters and their description
python benchmark.py # run the benchmark on an 
python benchmark --dataset-path DATASET_PATH --model-names snaive ... # run the benchmark on DATASET_PATH with only snaive
python benchmark --dataset-path DATASET_PATH --model-names ets ... # run the benchmark on DATASET_PATH with only ets
python benchmark --dataset-path DATASET_PATH --model-names snaive --model-names ets ... # run the benchmark on DATASET_PATH with ets and snaive
```


## HMM with external signals paper results

The following tabs summarize the results that can be reproduced with this code:


 - on simulated sequences:
  - with seasonal hmm with external signals:

| Model         | Mase        | Accuracy    |
| :-------------| :-----------| :-----------|
| snaive        | 0.881       | 0.357       |
| ets           | 0.807       | 0.449       |

  - with simple hmm:

| Model         | Mase        | Accuracy    |
| :-------------| :-----------| :-----------|
| snaive        | 0.881       | 0.357       |
| ets           | 0.807       | 0.449       |

 - on fashion sequences:

| Model         | Mase        | Accuracy    |
| :-------------| :-----------| :-----------|
| snaive        | 0.881       | 0.357       |
| ets           | 0.807       | 0.449       |

