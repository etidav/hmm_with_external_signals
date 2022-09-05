# Discrete hidden Markov model with inclusion of external signals

Authors: Etienne DAVID, Sylvain LE CORFF and Jean BELLOT

Paper link: 

### Abstract
> In this paper, we consider a bivariate process (Xt,Yt) whose transition and emission kernels depend on an additional process Wt. The resulting model (Xt,Yt,Wt) is referred to as a hidden Markov model with external signals and we prove that under several assumptions, it is identifiable and its maximum likelihood estimator is  consistent. Using the Expectation Maximization algorithm, we train and evaluate the performance of this new approach on several sequences. In addition to learning dependencies between the main signal and the external signal, hidden Markov model using external signals also outperforms state-of-the-art models on real-world sequences while showing theoretical guarantees and interpretability.
## Code Organisation

This repository provides the code of the hidden Markov model with the inclusion of external signals and a simple code base to reproduce the results presented in this [paper](). The repository is organized as follow:

 - [model/](model/): Directory gathering the code of the benchmarks and all the HMMs variation (hmm, shmm, arhmm, arshmm, hmmes, shmmes, arhmmes, arshmmes)
 - [run/](run/): Directory gathering the different scipt tro reproduced the result of the paper.
 - [data/](data/): Directory gathering the 10 fashion time series introduced in this paper [paper](https://arxiv.org/pdf/2202.03224.pdf) and used in the experimental section of the HMM with external signals paper.
 - [docker/](docker/): directory gathering the code to reproduce a docker so as to recover the exact result of the paper.  

## Reproduce benchmark results

First, you should build, run and enter into the docker. In the main folder, run
```bash
make build run enter
```

To reproduce the result on SHMMES simulated sequences :
- [shmmes_simulated_sequences.py](run/simulated_sequences/shmmes_simulated_sequences.py)
run
```bash
python run/simulated_sequences/shmmes_simulated_sequence.py --help # display the default parameters and their description
python run/simulated_sequences/shmmes_simulated_sequence.py --model_folder result/shmmes_simulated_sequence # run a hmm, shmm, hmmes and shmmes model on a simulated sequence using a shmmes model and save the results in the dir result/shmmes_simulated_sequence
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list all --nb_iteration_per_epoch 1 # commande to recover the exact result of the Table 1 of the HMM with external signals paper.
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 1000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list shmmes --nb_iteration_per_epoch 1 # commande to reproduce the result of the Table 4 of the HMM with external signals paper. (set --train_length 10000 and --train_length 100000 to recovered the result of column 2 and 3)
python3 run/simulated_sequences/shmmes_simulated_sequences.py --model_folder result/shmmes_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 10 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list shmmes --nb_iteration_per_epoch 1 # commande to reproduce the result of the Table 5 of the HMM with external signals paper. (set --percentage_of_variation 1 and --percentage_of_variation 2 to recovered the result of column 2 and 3) 
```
To reproduce the result on HMM simulated sequences :
- [hmm_simulated_sequences.py](run/simulated_sequences/hmm_simulated_sequences.py)
run
```bash
python run/simulated_sequences/hmm_simulated_sequence.py --help # display the default parameters and their description
python run/simulated_sequences/hmm_simulated_sequence.py --model_folder result/hmm_simulated_sequence # train a hmm, shmm, hmmes and shmmes model on a simulated sequence using a hmm model and save the results in the dir result/hmm_simulated_sequence
python3 run/simulated_sequences/hmm_simulated_sequences.py --model_folder result/hmm_simulated_sequence --train_length 10000 --test_length 250 --nb_em_epoch 1000 --nb_repetition 1 --init_with_true_parameter 1 --percentage_of_variation 0.5 --learning_rate 0.5 --nb_test_simulation 1000 --model_name_list all --nb_iteration_per_epoch 1 # commande to recover the exact result of the Table 6 of the HMM with external signals paper.
```
To reproduce the HMM-based models results on the 10 fashion sequences:
- [hmm_fashion_sequences.py](run/fashion_sequences/hmm_fashion_sequences.py)
run
```bash
python run/fashion_sequences/hmm_fashion_sequences.py --help # display the default parameters and their description 
python run/fashion_sequences/hmm_fashion_sequences.py --main_folder result/hmm_fashion_sequences --trend_name eu_female_top_325 --nb_em_epoch 10 --nb_iteration_per_epoch 5 --nb_em_execution 10 --nb_repetition 10 --nb_simulation 1000 --learning_rate 0.5 # train all the hmm variations on the fashion sequence eu_female_top_325 and provide the HMM result of the Table 2 of the HMM with external signals paper. 
python run/fashion_sequences/hmm_fashion_sequences.py --main_folder result/hmm_fashion_sequences --trend_name br_female_shoes_262 --nb_em_epoch 10 --nb_iteration_per_epoch 5 --nb_em_execution 10 --nb_repetition 10 --nb_simulation 1000 --learning_rate 0.5 # train all the hmm variations on the fashion sequence br_female_shoes_262 and provide the first column of the Table 3 (ts1). Run the same command with the following --trend_name arguments to recover the full HMMs results of table 3 : br_female_texture_59, br_female_texture_82, eu_female_outerwear_177, eu_female_top_325, eu_female_top_394, eu_female_texture_80, us_female_outerwear_171, us_female_shoes_76, us_female_top_79
```
To reproduce the benchmarks results on the 10 fashion sequences (except hermes and lstm models):
- [benchmark_fashion_sequences.py](run/fashion_sequences/benchmark_fashion_sequences.py)
run
```bash
python run/fashion_sequences/benchmark_fashion_sequences.py --help # display the default parameters and their description
python run/fashion_sequences/benchmark_fashion_sequences.py --main_folder result/benchmark_fashion_sequences # train benchmark models on all the fashion sequence and provide the benchmarks results of Table 2 and Table 3 of the HMM with external signals paper.
```

## HMM with external signals paper results

The following tabs summarize some results that can be reproduced with this code:


 - on SHMMES simulated sequences:

| Model         | Mase        | Mae         | Mse         |
| :-------------| :-----------| :-----------| :-----------|
| hmm           | 1.354       | 4.833       | 31.124      |
| shmm          | 0.903       | 3.222       | 15.582      |
| hmm-es        | 1.245       | 4.446       | 26.246      |
| shmm-es       | 0.737       | 2.630       | 14.102      |

 - on the eu_female_top_325 fashion sequences:

| Model         | Mase        | Mae         | Mse         |
| :-------------| :-----------| :-----------| :-----------|
| thetam        | 1.73        | 0.87        | 1.04        |
| ets           | 1.59        | 0.80        | 0.89        |
| tbats         | 1.25        | 0.63        | 0.68        |
| snaive        | 1.09        | 0.55        | 0.51        |
| lstm-ws       | 0.97        | 0.49        | 0.50        |
| lstm          | 0.78        | 0.39        | 0.28        |
| hermes        | 0.70        | 0.25        | 0.23        |
| hermes-ws     | 0.67        | 0.34        | 0.22        |
| hmm           |             |             |             |
| shmm          |             |             |             |
| hmm-es        |             |             |             |
| ar-hmm        |             |             |             |
| ar-hmm-es     |             |             |             |
| ar-shmm-es    |             |             |             |
| shmm-es       |             |             |             |
| ar-shmm-es    |             |             |             |
