# Discrete hidden Markov model with inclusion of external signals

Authors: Étienne David, Jean Bellot, Sylvain Le Corff and Luc Lehéricy

Paper link: 

### Abstract
> In this paper, we consider a bivariate process Xt,Yt which, conditionally on a signal Wt, is a hidden Markov model whose transition and emission kernels depend on Wt. The resulting process (Xt,Yt,Wt) is referred to as an input-output hidden Markov model or hidden Markov model with external signals. We prove that this model is identifiable and that the associated maximum likelihood estimator is consistent. Introducing an Expectation Maximization-based algorithm, we train and evaluate the performance of this model in several frameworks. In addition to learning dependencies between (Xt,Yt) and Wt, our approach based on hidden Markov models with external signals also outperforms state-of-the-art algorithms on real-world fashion sequences.

## Code Organisation

This repository provides the code of the hidden Markov model with the inclusion of external signals and a simple code base to reproduce results presented in this [paper](). The repository is organized as follow:

 - [model/](model/): Directory gathering the code of the benchmarks and all the HMMs variations (hmm, shmm, ar-hmm, ar-shmm, hmm-es, shmm-es, ar-hmm-es, ar-shmm-es)
 - [run/](run/): Directory gathering the different scripts to reproduce the result of the paper.
 - [data/](data/): Directory gathering the 10 fashion time series introduced in the HERMES [paper](https://arxiv.org/pdf/2202.03224.pdf) and used in Section 4.
 - [docker/](docker/): directory gathering the code to reproduce a docker so as to recover the exact result of the paper.  

## Reproduce benchmark results

First, you should build, run and enter into the docker. In the main folder, run
```bash
make build run enter
```

To reproduce the result on the synthetic sequence simulated with a shmm-es:
- [shmmes_simulated_sequences.py](run/simulated_sequences/shmmes_simulated_sequences.py)
run
```bash
python run/simulated_sequences/shmmes_simulated_sequence.py --help # display the default parameters and their description
python run/simulated_sequences/shmmes_simulated_sequence.py --model_folder result/shmmes_simulated_sequence # run a hmm, shmm, hmm-es and shmm-es model on a synthetic sequence simulated with a shmm-es model and save the results in the dir result/shmmes_simulated_sequence
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list all --nb_iteration_per_epoch 1 # commande to recover the exact result of the Table 1 of the HMM with external signals paper.
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 1000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list shmmes --nb_iteration_per_epoch 1 # commande to reproduce the result of the Table 4 of the HMM with external signals paper. (set --train_length 10000 and --train_length 100000 to recovered the result of column 2 and 3)
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list shmmes --nb_iteration_per_epoch 1 # commande to reproduce the result of the Table 5 of the HMM with external signals paper. (set --percentage_of_variation 1 and --percentage_of_variation 2 to recovered the result of column 2 and 3) 
```
To reproduce the result on HMM simulated sequences :
- [hmm_simulated_sequences.py](run/simulated_sequences/hmm_simulated_sequences.py)
run
```bash
python run/simulated_sequences/hmm_simulated_sequence.py --help # display the default parameters and their description
python run/simulated_sequences/hmm_simulated_sequence.py --model_folder result/hmm_simulated_sequence # train a hmm, shmm, hmm-es and shmm-es model on a synthetic sequence simulated with a hmm model and save the results in the dir result/hmm_simulated_sequence
python3 run/simulated_sequences/hmm_simulated_sequences.py --model_folder result/hmm_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 1 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list all --nb_iteration_per_epoch 1 # commande to recover the exact result of the Table 6 of the HMM with external signals paper.
```
To reproduce the HMM-based models results on the 10 fashion sequences:
- [hmm_fashion_sequences.py](run/fashion_sequences/hmm_fashion_sequences.py)
run
```bash
python run/fashion_sequences/hmm_fashion_sequences.py --help # display the default parameters and their description 
python run/fashion_sequences/hmm_fashion_sequences.py --main_folder result/hmm_fashion_sequences --trend_name eu_female_top_325 --nb_em_epoch 10 --nb_iteration_per_epoch 5 --nb_em_execution 10 --nb_repetition 10 --nb_simulation 1000 --learning_rate 0.5 # train all the hmm variations on the fashion sequence eu_female_top_325 and provide the result of Table 2. 
python run/fashion_sequences/hmm_fashion_sequences.py --main_folder result/hmm_fashion_sequences --trend_name br_female_shoes_262 --nb_em_epoch 10 --nb_iteration_per_epoch 5 --nb_em_execution 10 --nb_repetition 10 --nb_simulation 1000 --learning_rate 0.5 # train all the hmm variations on the fashion sequence br_female_shoes_262 and provide the first column of Table 3 (ts1). Run the same command with the following --trend_name arguments to recover the full results of table 3 : br_female_texture_59, br_female_texture_82, eu_female_outerwear_177, eu_female_top_325, eu_female_top_394, eu_female_texture_80, us_female_outerwear_171, us_female_shoes_76, us_female_top_79
```
To reproduce the benchmarks results on the 10 fashion sequences (except hermes and lstm models):
- [benchmark_fashion_sequences.py](run/fashion_sequences/benchmark_fashion_sequences.py)
run
```bash
python run/fashion_sequences/benchmark_fashion_sequences.py --help # display the default parameters and their description
python run/fashion_sequences/benchmark_fashion_sequences.py --main_folder result/benchmark_fashion_sequences # train benchmark models on all the fashion sequence and provide the benchmarks results of Table 2 and Table 3.
```

## HMM with external signals paper results

The following tabs summarizes some results that can be reproduced with this code:


 - on SHMMES simulated sequences:

| Model         | Mase        | Mae         | Mse         |
| :-------------| :-----------| :-----------| :-----------|
| hmm           | 1.354       | 4.833       | 31.124      |
| shmm          | 0.903       | 3.222       | 15.582      |
| hmmes         | 1.245       | 4.446       | 26.346      |
| shmmes        | 0.737       | 2.630       | 14.102      |

 - on the eu_female_top_325 fashion sequences:

| Model         | Mase        | Mae         | Mse         |
| :-------------| :-----------| :-----------| :-----------|
| thetam        | 1.73        | 0.87        | 1.04        |
| ets           | 1.59        | 0.80        | 0.89        |
| tbats         | 1.25        | 0.63        | 0.68        |
| snaive        | 1.09        | 0.55        | 0.51        |
| lstm-ws       | 0.97        | 0.49        | 0.50        |
| lstm          | 0.78        | 0.39        | 0.28        |
| hermes        | 0.70        | 0.35        | 0.23        |
| hermes-ws     | 0.67        | 0.34        | 0.22        |
| hmm           | 1.99        | 1.01        | 1.78        |
| shmm          | 1.95        | 0.99        | 1.61        |
| hmm-es        | 0.98        | 0.60        | 0.52        |
| ar-hmm        | 0.95        | 0.58        | 0.62        |
| ar-hmm-es     | 0.80        | 0.49        | 0.40        |
| ar-shmm       | 0.77        | 0.47        | 0.43        |
| shmm-es       | 0.62        | 0.38        | 0.24        |
| ar-shmm-es    | 0.56        | 0.35        | 0.24        |
