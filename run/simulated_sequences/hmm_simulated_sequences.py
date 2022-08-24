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
from model.HMMES import HMMES
from model.SHMMES import SHMMES


def init_model_param(
    model_name, original_model, init_with_true_parameter, percentage_of_variation
):
    if init_with_true_parameter:
        init_param = list(original_model.get_param())
        init_param[1] = init_param[1] + np.random.normal(
            0, percentage_of_variation, init_param[1].flatten().shape[0]
        ).reshape(init_param[1].shape)
        init_param[2] = init_param[2] + np.random.normal(
            0, percentage_of_variation, init_param[2].flatten().shape[0]
        ).reshape(init_param[2].shape)
        init_param[2][init_param[2] < 0] = 0.1
        init_param[3] = init_param[3] + np.random.normal(
            0, percentage_of_variation, init_param[3].flatten().shape[0]
        ).reshape(init_param[3].shape)
        if model_name.lower() == "hmm":
            model = HMM()
            model.assign_param(*init_param)
        elif model_name.lower() == "shmm":
            model = SHMM(season=52)
            shmm_param = list(model.get_param())
            shmm_param[0] = init_param[0]
            shmm_param[1][:, 0] = init_param[1].flatten()
            shmm_param[1][:, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(shmm_param[1][:, 1:].flatten()))
            ).reshape(
                shmm_param[1][:, 1:].shape
            )
            shmm_param[2] = init_param[2]
            shmm_param[3][:, :, :1] = init_param[3]
            shmm_param[3][:, :, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(shmm_param[3][:, :, 1:].flatten()))
            ).reshape(
                shmm_param[3][:, :, 1:].shape
            )
            model.assign_param(*shmm_param)
        elif model_name.lower() == "hmmes":
            model = HMMES()
            hmmes_param = list(model.get_param())
            hmmes_param[0] = init_param[0]
            hmmes_param[1][:, 0] = init_param[1].flatten()
            hmmes_param[1][:, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(hmmes_param[1][:, 1:].flatten()))
            ).reshape(
                hmmes_param[1][:, 1:].shape
            )
            hmmes_param[2] = init_param[2]
            hmmes_param[3][:, :, :1] = init_param[3]
            hmmes_param[3][:, :, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(hmmes_param[3][:, :, 1:].flatten()))
            ).reshape(
                hmmes_param[3][:, :, 1:].shape
            )
            model.assign_param(*hmmes_param)
        elif model_name.lower() == "shmmes":
            model = SHMMES(season=52)
            shmmes_param = list(model.get_param())
            shmmes_param[0] = init_param[0]
            shmmes_param[1][:, 0] = init_param[1].flatten()
            shmmes_param[1][:, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(shmmes_param[1][:, 1:].flatten()))
            ).reshape(
                shmmes_param[1][:, 1:].shape
            )
            shmmes_param[2] = init_param[2]
            shmmes_param[3][:, :, :1] = init_param[3]
            shmmes_param[3][:, :, 1:] = np.random.normal(
                0, percentage_of_variation
            ) * np.power(
                -1, np.random.randint(0, 2, len(shmmes_param[3][:, :, 1:].flatten()))
            ).reshape(
                shmmes_param[3][:, :, 1:].shape
            )
            model.assign_param(*shmmes_param)
        else:
            raise ValueError(
                "The model name {} is not in ['hmm','shmm','hmmes','shmmes'].".format(
                    model_name.lower()
                )
            )
        init_param = False
    else:
        init_param = True

    return model, init_param


def train_model(
    model_name,
    y_train,
    y_test,
    w_train,
    w_test,
    train_length,
    test_length,
    nb_test_simulation,
    learning_rate,
    init_with_true_parameter,
    percentage_of_variation,
    nb_em_epoch,
    nb_iteration_per_epoch,
    nb_repetition,
    model_folder,
    original_model,
):

    for i in range(nb_repetition):
        current_execution_model_folder = os.path.join(
            model_folder, model_name.lower(), "execution" + str(i), ""
        )
        os.makedirs(current_execution_model_folder)

        model, init_param = init_model_param(
            model_name,
            original_model,
            init_with_true_parameter,
            percentage_of_variation,
        )
        all_param, run_mse, best_exec = model.fit(
            y=tf.constant(y_train),
            w=tf.constant(w_train[1:]),
            eval_size=test_length,
            nb_em_epoch=nb_em_epoch,
            nb_iteration_per_epoch=nb_iteration_per_epoch,
            nb_execution=1,
            optimizer_name="Adam",
            learning_rate=learning_rate,
            init_param=init_param,
            return_param_evolution=True,
        )

        model.save_weights(current_execution_model_folder)
        write_pickle(
            all_param,
            os.path.join(current_execution_model_folder, "param_evolution.pkl"),
        )
        write_pickle(
            run_mse, os.path.join(current_execution_model_folder, "run_mse.pkl")
        )
        model_eval = model.evaluate(
            y_train,
            y_test,
            w_train,
            w_test,
            train_length,
            test_length,
            nb_test_simulation,
        )
        write_pickle(
            model_eval, os.path.join(current_execution_model_folder, "model_eval.pkl")
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulated data using simple HMM")
    parser.add_argument(
        "--model_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument(
        "--model_name_list", type=str, nargs="+", help="model to to fit", default="all"
    )
    parser.add_argument("--train_length", type=int, help="", default=1000)
    parser.add_argument("--test_length", type=int, help="", default=100)
    parser.add_argument("--nb_test_simulation", type=int, help="", default=1000)
    parser.add_argument("--nb_em_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_iteration_per_epoch", type=int, help="", default=1)
    parser.add_argument("--nb_repetition", type=int, help="", default=1)
    parser.add_argument("--init_with_true_parameter", type=int, help="", default=0)
    parser.add_argument("--percentage_of_variation", type=float, help="", default=0.0)
    parser.add_argument("--learning_rate", type=float, help="", default=0.05)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    np.random.seed(42)
    tf.random.set_seed(42)

    model_folder = args.model_folder
    model_name_list = args.model_name_list
    if model_name_list == ["all"]:
        model_name_list = ["hmm", "shmm", "hmmes", "shmmes"]
    train_length = args.train_length
    test_length = args.test_length
    nb_test_simulation = args.nb_test_simulation
    nb_em_epoch = args.nb_em_epoch
    nb_iteration_per_epoch = args.nb_iteration_per_epoch
    nb_repetition = args.nb_repetition
    init_with_true_parameter = args.init_with_true_parameter
    percentage_of_variation = args.percentage_of_variation
    learning_rate = args.learning_rate
    if model_folder[-1] != "/":
        model_folder += "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        if os.path.isdir(model_folder) and len(os.listdir(model_folder)) > 0:
            raise ValueError(
                "The model folder {} already exists and isn't empty. you need to use an other model folder".format(
                    model_folder
                )
            )
    config_dict = vars(args)
    write_json(config_dict, os.path.join(model_folder, "config.json"))
    original_model = HMM(nb_hidden_states=2)
    pi = np.array([0.0, 1.0])
    mu = np.array([-1.0, 2.0]).reshape(2, 1)
    sigma = np.array([1.0, 0.25])
    tp = np.array([-0.8, -1.4]).reshape(2, 1, 1)
    original_model.assign_param(pi, mu, sigma, tp)
    result = original_model.simulate_xy(train_length + test_length, 0)
    x_train = result[0][:train_length]
    x_test = result[0][train_length:]
    y_train = result[1][:train_length]
    y_test = result[1][train_length:]

    external_signal = pd.read_csv(
        "/hmm_with_external_signals/data/f1_fashion_forward_10_sequences.csv", index_col=0
    )["eu_female_top_325"]
    external_signal = external_signal.rolling(8, min_periods=0).mean()
    external_signal = external_signal.values.flatten()
    external_signal /= external_signal[:52].mean()
    external_signal = np.concatenate(
        [
            external_signal
            for i in range(int((train_length + test_length) / len(external_signal)) + 1)
        ]
    )
    w_train = external_signal[-train_length-test_length:-test_length]
    w_test = external_signal[-test_length:]

    write_pickle(y_train, os.path.join(model_folder, "data_y_train.pkl"))
    write_pickle(y_test, os.path.join(model_folder, "data_y_test.pkl"))
    write_pickle(x_train, os.path.join(model_folder, "data_x_train.pkl"))
    write_pickle(x_test, os.path.join(model_folder, "data_x_test.pkl"))
    write_pickle(w_train, os.path.join(model_folder, "data_w_train.pkl"))
    write_pickle(w_test, os.path.join(model_folder, "data_w_test.pkl"))

    for model_name in model_name_list:
        train_model(
            model_name,
            y_train,
            y_test,
            w_train,
            w_test,
            train_length,
            test_length,
            nb_test_simulation,
            learning_rate,
            init_with_true_parameter,
            percentage_of_variation,
            nb_em_epoch,
            nb_iteration_per_epoch,
            nb_repetition,
            model_folder,
            original_model,
        )
