import argparse
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from run.utils import (
    read_json,
    read_pickle,
    read_yaml,
    write_json,
    write_pickle,
    write_yaml,
)

import matplotlib.patches as patches
import matplotlib.dates as mdates
from datetime import datetime
from pandas.plotting import table
import matplotlib.pyplot as plt
import tensorflow as tf
from model.HMM import HMM
from model.SHMM import SHMM
from model.HMMES import HMMES_window
from model.SHMMES import SHMMES_window
from model.ARHMM import ARHMM_window
from model.ARSHMM import ARSHMM_window
from model.ARSHMMES import ARSHMMES_window
from model.ARHMMES import ARHMMES_window

def run_model(model_folder, model, y_train, y_test, w_train, w_test, nb_em_epoch, nb_execution, nb_simulation, learning_rate, norm):
    
    
    for i in range(10):
        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = 52,
            nb_em_epoch =nb_em_epoch,
            nb_execution =nb_execution,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=1,
            return_param_evolution=True
        )
        model.assign_param(*all_param[best_exec][0])

        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = 52,
            nb_em_epoch =100,
            nb_execution =1,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=0,
            return_param_evolution=True
        )
        if run_mse < np.inf:
            break
    write_pickle(model.get_param(), os.path.join(model_folder, "final_param.pkl"))
    write_pickle(all_param, os.path.join(model_folder, "param_evolution.pkl"))
    write_pickle(run_mse, os.path.join(model_folder, "run_mse.pkl"))
    model_eval = model.evaluate(y_train, y_test, w_train, w_test, len(y_train), len(y_test), nb_simulation, norm)
    write_pickle(model_eval, os.path.join(model_folder, "model_eval.pkl"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulated data using simple HMM")
    parser.add_argument(
        "--main_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument("--trend_name", type=str, help="", default="eu_female_top_325")
    parser.add_argument("--nb_em_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_em_execution", type=int, help="", default=1)
    parser.add_argument("--nb_repetition", type=int, help="", default=1)
    parser.add_argument("--nb_simulation", type=int, help="", default=1)
    parser.add_argument("--learning_rate", type=float, help="", default=0.05)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    np.random.seed(1)
    tf.random.set_seed(1)
    
    main_folder = args.main_folder
    nb_em_epoch = args.nb_em_epoch
    nb_em_execution = args.nb_em_execution
    nb_repetition = args.nb_repetition
    nb_simulation = args.nb_simulation
    learning_rate = args.learning_rate
    trend_list = [args.trend_name]
    if main_folder[-1] != "/":
        main_folder += "/"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        
    config_dict = vars(args)
    write_json(config_dict,os.path.join(main_folder, "config.json"))
    
    all_y_data = pd.read_csv('/hmm_with_external_signals/data/f1_main_10_sequences.csv',index_col=0)
    all_w_data = pd.read_csv('/hmm_with_external_signals/data/f1_fashion_forward_10_sequences.csv',index_col=0)
    
    for trend in trend_list:
        y_data = all_y_data[trend].values
        w_data = all_w_data[trend].rolling(8,min_periods=0).mean().values
        y_train = y_data[:-52]/y_data[:52].mean()
        y_test = y_data[-52:]
        w_train = w_data[:-52]/w_data[:52].mean()
        w_test = np.zeros_like(w_data[-52:])
        
        trend_folder = os.path.join(main_folder,trend)
        os.makedirs(trend_folder)
        
        model_dict = {}
        model_dict['hmm'] = HMM()
        model_dict['shmm'] = SHMM(season=52)
        model_dict['hmmes'] =  HMMES_window(past_dependency=52, season=52, past_dependency0 = 1)
        model_dict['shmmes'] = SHMMES_window(past_dependency=52, season=52, past_dependency0 = 1)
        model_dict['arshmm'] = ARSHMM_window(past_dependency=52, season=52, past_dependency0 = 1)
        model_dict['arhmm'] = ARHMM_window(past_dependency=52, season=52, past_dependency0 = 1)
        model_dict['arhmmes'] = ARHMMES_window(past_dependency=52, season=52, past_dependency0 = 1)
        model_dict['arshmmes'] = ARSHMMES_window(past_dependency=52, season=52, past_dependency0 = 1)
        
        for model_name in model_dict.keys():
            model = model_dict[model_name]
            model_folder = os.path.join(trend_folder,model_name)
            os.makedirs(model_folder)
            for i in range(nb_repetition):
                exec_folder = os.path.join(model_folder,f'exec{i}')
                os.makedirs(exec_folder)
                run_model(exec_folder, model, y_train, y_test, w_train, w_test, nb_em_epoch, nb_em_execution, nb_simulation, learning_rate, y_data[:52].mean())
