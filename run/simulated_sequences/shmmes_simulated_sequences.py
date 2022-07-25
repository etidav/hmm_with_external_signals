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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulated data using simple HMM")
    parser.add_argument(
        "--model_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument("--train_length", type=int, help="", default=1000)
    parser.add_argument("--test_length", type=int, help="", default=100)
    parser.add_argument("--nb_em_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_execution", type=int, help="", default=1)
    parser.add_argument("--init_with_true_parameter", type=int, help="", default=0)
    parser.add_argument("--percentage_of_variation", type=float, help="", default=0.)
    parser.add_argument("--learning_rate", type=float, help="", default=0.05)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model_folder = args.model_folder
    train_length = args.train_length
    test_length = args.test_length
    nb_em_epoch = args.nb_em_epoch
    nb_execution = args.nb_execution
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
    write_json(config_dict,os.path.join(model_folder, "config.json"))
    
    external_signal = pd.read_csv('/hmm_with_external_signals/data/f1_main_10_sequences.csv',index_col=0)['eu_female_top_325']
    external_signal = external_signal.rolling(8,min_periods=0).mean()
    external_signal = external_signal.values.flatten()
    external_signal /= external_signal[:52].mean()
    external_signal = np.concatenate([external_signal for i in range(int((train_length+test_length)/len(external_signal)) + 1)])
    w_train = external_signal[-train_length-test_length:-test_length]
    w_test = external_signal[-test_length:]
    
    
    original_model = SHMMES(nb_hidden_states=2, season=52)
    pi = np.array([0.,1.])
    sigma = np.array([0.25,0.5])
    delta = np.array([[3.,0.8,2.5,4.],[-1.1,-0.1,-1.5, 3.5]])
    omega = np.array([[[0.5,.9,0.7,0.5]],[[-2,-0.2,-0.6, 0.7]]])
    original_model.assign_param(pi,delta, sigma, omega)
    result = original_model.simulate_xy(horizon = train_length+test_length,w=external_signal[-train_length-test_length:])
    x_train = result[0][:train_length]
    x_test = result[0][train_length:]
    y_train = result[1][:train_length]
    y_test = result[1][train_length:]
    
    write_pickle(y_train,os.path.join(model_folder, "data_y_train.pkl"))
    write_pickle(y_test,os.path.join(model_folder, "data_y_test.pkl"))
    write_pickle(x_train,os.path.join(model_folder, "data_x_train.pkl"))
    write_pickle(x_test,os.path.join(model_folder, "data_x_test.pkl"))
    write_pickle(w_train,os.path.join(model_folder, "data_w_train.pkl"))
    write_pickle(w_test,os.path.join(model_folder, "data_w_test.pkl"))
    
    model = HMM()
    for i in range(nb_execution):
        current_execution_model_folder = os.path.join(model_folder,'hmm','execution'+str(i),"")
        os.makedirs(current_execution_model_folder)
        
        if init_with_true_parameter:
            hmm_param = list(model.get_param())
            init_param = list(original_model.get_param())
            init_param[1] = init_param[1] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[1].flatten()))).reshape(init_param[1].shape)
            init_param[2] = init_param[2] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[2].flatten()))).reshape(init_param[2].shape)
            init_param[2][init_param[2]<0] = 0.1
            init_param[3] = init_param[3] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[3].flatten()))).reshape(init_param[3].shape)
            hmm_param[0] = init_param[0]
            hmm_param[1] = init_param[1][:,0].reshape((-1,1))
            hmm_param[2] = init_param[2]
            hmm_param[3] = init_param[3][:,0].reshape((-1,1))
            model.assign_param(*hmm_param)
            init_param = False
        else:
            init_param = True
    
    
        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = test_length,
            nb_em_epoch =nb_em_epoch,
            nb_execution =1,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=init_param,
            return_param_evolution=True
        )
    
        model.save_weights(current_execution_model_folder)
        write_pickle(all_param, os.path.join(current_execution_model_folder, "param_evolution.pkl"))
        write_pickle(run_mse, os.path.join(current_execution_model_folder, "run_mse.pkl"))
        model_eval = model.evaluate(y_train, y_test, w_train, w_test, train_length, test_length, 1000)
        write_pickle(model_eval, os.path.join(current_execution_model_folder, "model_eval.pkl"))
        
    model = SHMM(season=52)
    for i in range(nb_execution):
        current_execution_model_folder = os.path.join(model_folder,'shmm','execution'+str(i),"")
        os.makedirs(current_execution_model_folder)
        
        if init_with_true_parameter:
            hmm_param = list(model.get_param())
            init_param = list(original_model.get_param())
            init_param[1] = init_param[1] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[1].flatten()))).reshape(init_param[1].shape)
            init_param[2] = init_param[2] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[2].flatten()))).reshape(init_param[2].shape)
            init_param[2][init_param[2]<0] = 0.1
            init_param[3] = init_param[3] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[3].flatten()))).reshape(init_param[3].shape)
            hmm_param[0] = init_param[0]
            hmm_param[1][:,0] = init_param[1][:,0]
            hmm_param[1][:,1:] = init_param[1][:,2:]
            hmm_param[2] = init_param[2]
            hmm_param[3][:,0] = init_param[3][:,0]
            hmm_param[3][:,1:] = init_param[3][:,2:]
            model.assign_param(*hmm_param)
            init_param = False
        else:
            init_param = True
    
    
        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = test_length,
            nb_em_epoch =nb_em_epoch,
            nb_execution =1,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=init_param,
            return_param_evolution=True
        )
    
        model.save_weights(current_execution_model_folder)
        write_pickle(all_param, os.path.join(current_execution_model_folder, "param_evolution.pkl"))
        write_pickle(run_mse, os.path.join(current_execution_model_folder, "run_mse.pkl"))
        model_eval = model.evaluate(y_train, y_test, w_train, w_test, train_length, test_length, 1000)
        write_pickle(model_eval, os.path.join(current_execution_model_folder, "model_eval.pkl"))
        
    model = HMMES()
    for i in range(nb_execution):
        current_execution_model_folder = os.path.join(model_folder,'hmmes','execution'+str(i),"")
        os.makedirs(current_execution_model_folder)
        
        if init_with_true_parameter:
            hmmes_param = list(model.get_param())
            init_param = list(original_model.get_param())
            init_param[1] = init_param[1] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[1].flatten()))).reshape(init_param[1].shape)
            init_param[2] = init_param[2] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[2].flatten()))).reshape(init_param[2].shape)
            init_param[2][init_param[2]<0] = 0.1
            init_param[3] = init_param[3] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[3].flatten()))).reshape(init_param[3].shape)
            hmmes_param[0] = init_param[0]
            hmmes_param[1] = init_param[1][:,:2]
            hmmes_param[2] = init_param[2]
            hmmes_param[3] = init_param[3][:,:2]
            model.assign_param(*hmmes_param)
            init_param = False
        else:
            init_param = True
    
        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = test_length,
            nb_em_epoch =nb_em_epoch,
            nb_execution =1,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=init_param,
            return_param_evolution=True
        )
    
        model.save_weights(current_execution_model_folder)
        write_pickle(all_param, os.path.join(current_execution_model_folder, "param_evolution.pkl"))
        write_pickle(run_mse, os.path.join(current_execution_model_folder, "run_mse.pkl"))
        model_eval = model.evaluate(y_train, y_test, w_train, w_test, train_length, test_length, 1000)
        write_pickle(model_eval, os.path.join(current_execution_model_folder, "model_eval.pkl"))
        
    model = SHMMES(season=52)
    for i in range(nb_execution):
        current_execution_model_folder = os.path.join(model_folder,'shmmes','execution'+str(i),"")
        os.makedirs(current_execution_model_folder)
        
        if init_with_true_parameter:
            init_param = list(original_model.get_param())
            init_param[1] = init_param[1] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[1].flatten()))).reshape(init_param[1].shape)
            init_param[2] = init_param[2] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[2].flatten()))).reshape(init_param[2].shape)
            init_param[2][init_param[2]<0] = 0.1
            init_param[3] = init_param[3] + np.random.normal(0,percentage_of_variation)*np.power(-1,np.random.randint(0,2,len(init_param[3].flatten()))).reshape(init_param[3].shape)
            model.assign_param(*init_param)
            init_param = False
        else:
            init_param = True
    
    
        all_param, run_mse, best_exec = model.fit(
            y = tf.constant(y_train),
            w = tf.constant(w_train),
            eval_size = test_length,
            nb_em_epoch =nb_em_epoch,
            nb_execution =1,
            optimizer_name = 'Adam',
            learning_rate=learning_rate,
            init_param=init_param,
            return_param_evolution=True
        )
    
        model.save_weights(current_execution_model_folder)
        write_pickle(all_param, os.path.join(current_execution_model_folder, "param_evolution.pkl"))
        write_pickle(run_mse, os.path.join(current_execution_model_folder, "run_mse.pkl"))
        model_eval = model.evaluate(y_train, y_test, w_train, w_test, train_length, test_length, 1000)
        write_pickle(model_eval, os.path.join(current_execution_model_folder, "model_eval.pkl"))
        
    ####EVALUATION####

    def plot_param_evolution(hmm_param_evolution,hmm_true_parameter, model_name, folder_dir):

        for i in hmm_param_evolution.columns:
            plot_path_out = os.path.join(folder_dir,model_name,i+'.png')
            fig, ax1 = plt.subplots(figsize=(15, 6))
            size = 16
            lns2 = ax1.plot(hmm_param_evolution[i].index,np.repeat(hmm_true_parameter[i],hmm_param_evolution[i].shape[0]) , color = 'red', label = f'true_{i}')
            lns4 = ax1.plot(hmm_param_evolution[i].index, hmm_param_evolution[i], color = 'black', label = i)
            lns = [lns2[0],lns4[0]]
            ax1.set_xlabel('epoch',fontsize = size)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.legend(fontsize='large')
            ax1.tick_params(labelsize=size)
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc=0,fontsize='large')
            ax1.set_title(model_name + ': ' + i, fontsize=15)
            fig.tight_layout()
            fig.savefig(plot_path_out, format='png')
            plt.close(fig)

    def plot_model_prediction(experiment_dir_path,model_name, folder_dir):

        all_run_mse = []
        for i in os.listdir(os.path.join(experiment_dir_path,model_name.lower())):
            run_mse = read_pickle(os.path.join(experiment_dir_path,model_name.lower(), i,'run_mse.pkl'))
            all_run_mse.append(run_mse)
        exec_number = all_run_mse.index(max(all_run_mse))
        hmm_result = read_pickle(os.path.join(experiment_dir_path,f'{model_name.lower()}/execution{exec_number}/model_eval.pkl'))
        data_train = read_pickle(os.path.join(experiment_dir_path,'data_y_train.pkl'))
        data_test = read_pickle(os.path.join(experiment_dir_path,'data_y_test.pkl'))
        y_pred_q1 = hmm_result[0]['y_pred_q1']
        y_pred_q9 = hmm_result[0]['y_pred_q9']
        y_pred_mean = hmm_result[0]['y_pred_mean']

        plot_path_out = os.path.join(folder_dir,f'{model_name}_prediction.png')
        fig, ax1 = plt.subplots(figsize=(15, 6))
        size = 16
        lns1 = ax1.fill_between(range(train_length,train_length+test_length), y_pred_q9, y_pred_q1, color = 'grey', alpha=0.2, label = 'CI 90%')
        lns2 = ax1.plot(range(train_length,train_length+test_length), y_pred_mean, color = 'red', label = 'y_mean_pred')
        lns4 = ax1.plot(range(train_length,train_length+test_length), data_test, color = 'black', label = 'y_test')

        lns = [lns1,lns2[0],lns4[0]]

        ax1.set_xlabel('Time',fontsize = size)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(fontsize='large')
        ax1.tick_params(labelsize=size)
        ax1.set_title(model_name.upper(), fontsize=15)
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0,fontsize='large')
        fig.tight_layout()
        fig.savefig(plot_path_out, format='png')
        plt.close(fig)

    result_folder = os.path.join(model_folder,'result')

    if result_folder[-1] != "/":
        result_folder += "/"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        else:
            if os.path.isdir(result_folder) and len(os.listdir(result_folder)) > 0:
                raise ValueError(
                    "The model folder {} already exists and isn't empty. you need to use an other model folder".format(
                        result_folder
                    )
                )

    # final param
    hmm_all_param = {}
    for i in range(nb_execution):
        param = read_pickle(os.path.join(model_folder,f'hmm/execution{i}/param_evolution.pkl'))
        hmm_all_param[i] = param[0][-1]
    shmm_all_param = {}
    for i in range(nb_execution):
        param = read_pickle(os.path.join(model_folder,f'shmm/execution{i}/param_evolution.pkl'))
        shmm_all_param[i] = param[0][-1]
    hmmes_all_param = {}
    for i in range(nb_execution):
        param = read_pickle(os.path.join(model_folder,f'hmmes/execution{i}/param_evolution.pkl'))
        hmmes_all_param[i] = param[0][-1]
    shmmes_all_param = {}
    for i in range(nb_execution):
        param = read_pickle(os.path.join(model_folder,f'shmmes/execution{i}/param_evolution.pkl'))
        shmmes_all_param[i] = param[0][-1]


    nb_param = len(np.concatenate([i.flatten() for i in hmm_all_param[0]]))
    hmm_final_param = np.array([0., 1., 3., -1.1, 0.25, 0.5, 0.5, -2.]).reshape((1,nb_param))
    for nb_exec in hmm_all_param.keys():
        model_param = np.concatenate([i.flatten() for i in hmm_all_param[nb_exec]])
        hmm_final_param = np.concatenate([hmm_final_param,[model_param]], axis=0)
    index = ['true_param']
    for i in hmm_all_param.keys():
        index.append(f'exec_{i}')
    columns = ['pi1', 'pi2', 'delta11', 'delta21', 'sigma1', 'sigma2', 'omega11', 'omega21']
    hmm_final_param = pd.DataFrame(hmm_final_param, index=index, columns=columns)
    
    nb_param = len(np.concatenate([i.flatten() for i in shmm_all_param[0]]))
    shmm_final_param = np.array([0., 1., 3., 2.5, 4., -1.1, -1.5, 3.5, 0.25, 0.5, 0.5, 0.7, 0.5,-2.,-0.6,0.7]).reshape((1,nb_param))
    for nb_exec in shmm_all_param.keys():
        model_param = np.concatenate([i.flatten() for i in shmm_all_param[nb_exec]])
        shmm_final_param = np.concatenate([shmm_final_param,[model_param]], axis=0)
    index = ['true_param']
    for i in shmm_all_param.keys():
        index.append(f'exec_{i}')
    columns = ['pi1', 'pi2', 'delta11','delta13','delta14', 'delta21','delta23','delta24', 'sigma1', 'sigma2', 'omega11', 'omega13','omega14','omega21','omega23','omega24']
    shmm_final_param = pd.DataFrame(shmm_final_param, index=index, columns=columns)

    nb_param = len(np.concatenate([i.flatten() for i in hmmes_all_param[0]]))
    hmmes_final_param = np.array([0., 1., 3., 0.8, -1.1, -0.1, 0.25, 0.5, 0.5, 0.9, -2., -0.2]).reshape((1,nb_param))
    for nb_exec in hmmes_all_param.keys():
        model_param = np.concatenate([i.flatten() for i in hmmes_all_param[nb_exec]])
        hmmes_final_param = np.concatenate([hmmes_final_param,[model_param]], axis=0)
    index = ['true_param']
    for i in hmmes_all_param.keys():
        index.append(f'exec_{i}')
    columns = ['pi1', 'pi2', 'delta11', 'delta12','delta21', 'delta22', 'sigma1', 'sigma2', 'omega11', 'omega12','omega21', 'omega22']
    hmmes_final_param = pd.DataFrame(hmmes_final_param, index=index, columns=columns)
    
    nb_param = len(np.concatenate([i.flatten() for i in shmmes_all_param[0]]))
    shmmes_final_param = np.array([0., 1., 3., 0.8, 2.5, 4., -1.1, -0.1, -1.5, 3.5, 0.25, 0.5, 0.5, 0.9, 0.7, 0.5,-2.,-0.2,-0.6,0.7]).reshape((1,nb_param))
    for nb_exec in shmmes_all_param.keys():
        model_param = np.concatenate([i.flatten() for i in shmmes_all_param[nb_exec]])
        shmmes_final_param = np.concatenate([shmmes_final_param,[model_param]], axis=0)
    index = ['true_param']
    for i in shmmes_all_param.keys():
        index.append(f'exec_{i}')
    columns = ['pi1','pi2','delta11','delta12','delta13','delta14','delta21','delta22','delta23','delta24','sigma1','sigma2','omega11','omega12','omega13','omega14','omega21','omega22','omega23','omega24']
    shmmes_final_param = pd.DataFrame(shmmes_final_param, index=index, columns=columns)


    write_pickle(hmm_final_param,os.path.join(result_folder,'hmm_final_parameter.pkl'))
    write_pickle(hmmes_final_param,os.path.join(result_folder,'hmmes_final_parameter.pkl'))
    write_pickle(shmmes_final_param,os.path.join(result_folder,'shmmes_final_parameter.pkl'))

    #param evolution

    all_run_mse = []
    for i in os.listdir(os.path.join(model_folder,'hmm')):
        run_mse = read_pickle(os.path.join(model_folder,'hmm', i,'run_mse.pkl'))
        all_run_mse.append(run_mse)
    exec_number = all_run_mse.index(max(all_run_mse))
    param = read_pickle(os.path.join(model_folder,f'hmm/execution{exec_number}/param_evolution.pkl'))
    nb_param = len(np.concatenate([i.flatten() for i in param[0][0]]))
    hmm_param_evolution = np.zeros((1,nb_param))
    for epoch_param in param[0]:
        model_param = np.concatenate([i.flatten() for i in epoch_param])
        hmm_param_evolution = np.concatenate([hmm_param_evolution,[model_param]], axis=0)
    columns = ['pi1', 'pi2', 'delta11', 'delta21', 'sigma1', 'sigma2', 'omega11', 'omega21']
    hmm_param_evolution = pd.DataFrame(hmm_param_evolution[1:], columns=columns)
    
    all_run_mse = []
    for i in os.listdir(os.path.join(model_folder,'shmm')):
        run_mse = read_pickle(os.path.join(model_folder,'shmm', i,'run_mse.pkl'))
        all_run_mse.append(run_mse)
    exec_number = all_run_mse.index(max(all_run_mse))
    param = read_pickle(os.path.join(model_folder,f'shmm/execution{exec_number}/param_evolution.pkl'))
    nb_param = len(np.concatenate([i.flatten() for i in param[0][0]]))
    shmm_param_evolution = np.zeros((1,nb_param))
    for epoch_param in param[0]:
        model_param = np.concatenate([i.flatten() for i in epoch_param])
        shmm_param_evolution = np.concatenate([shmm_param_evolution,[model_param]], axis=0)
    columns = ['pi1','pi2','delta11','delta13','delta14','delta21','delta23','delta24','sigma1','sigma2','omega11','omega13','omega14','omega21','omega23','omega24']
    shmm_param_evolution = pd.DataFrame(shmm_param_evolution[1:], columns=columns)

    all_run_mse = []
    for i in os.listdir(os.path.join(model_folder,'hmmes')):
        run_mse = read_pickle(os.path.join(model_folder,'hmmes', i,'run_mse.pkl'))
        all_run_mse.append(run_mse)
    exec_number = all_run_mse.index(max(all_run_mse))
    param = read_pickle(os.path.join(model_folder,f'hmmes/execution{exec_number}/param_evolution.pkl'))
    nb_param = len(np.concatenate([i.flatten() for i in param[0][0]]))
    hmmes_param_evolution = np.zeros((1,nb_param))
    for epoch_param in param[0]:
        model_param = np.concatenate([i.flatten() for i in epoch_param])
        hmmes_param_evolution = np.concatenate([hmmes_param_evolution,[model_param]], axis=0)
    columns = ['pi1', 'pi2', 'delta11', 'delta12','delta21', 'delta22', 'sigma1', 'sigma2', 'omega11', 'omega12','omega21', 'omega22']
    hmmes_param_evolution = pd.DataFrame(hmmes_param_evolution[1:], columns=columns)

    all_run_mse = []
    for i in os.listdir(os.path.join(model_folder,'shmmes')):
        run_mse = read_pickle(os.path.join(model_folder,'shmmes', i,'run_mse.pkl'))
        all_run_mse.append(run_mse)
    exec_number = all_run_mse.index(max(all_run_mse))
    param = read_pickle(os.path.join(model_folder,f'shmmes/execution{exec_number}/param_evolution.pkl'))
    nb_param = len(np.concatenate([i.flatten() for i in param[0][0]]))
    shmmes_param_evolution = np.zeros((1,nb_param))
    for epoch_param in param[0]:
        model_param = np.concatenate([i.flatten() for i in epoch_param])
        shmmes_param_evolution = np.concatenate([shmmes_param_evolution,[model_param]], axis=0)
    columns = ['pi1','pi2','delta11','delta12','delta13','delta14','delta21','delta22','delta23','delta24','sigma1','sigma2','omega11','omega12','omega13','omega14','omega21','omega22','omega23','omega24']
    shmmes_param_evolution = pd.DataFrame(shmmes_param_evolution[1:], columns=columns)


    hmm_true_parameter = np.array([0., 1., 3., 0.8, 2.5, 4., -1.1, -0.1, -1.5, 3.5, .25, 0.5, 0.5, 0.9, 0.7, 0.5,-2.,-0.2,-0.6,0.7])
    hmm_true_parameter = pd.DataFrame(hmm_true_parameter,index = columns).T

    write_pickle(hmm_param_evolution,os.path.join(result_folder,'hmm_parameter_evolution.pkl'))
    write_pickle(shmm_param_evolution,os.path.join(result_folder,'shmm_parameter_evolution.pkl'))
    write_pickle(hmmes_param_evolution,os.path.join(result_folder,'hmmes_parameter_evolution.pkl'))
    write_pickle(shmmes_param_evolution,os.path.join(result_folder,'shmmes_parameter_evolution.pkl'))


    os.makedirs(os.path.join(result_folder,'parameter_evolution_plot'))
    os.makedirs(os.path.join(result_folder,'parameter_evolution_plot','HMM'))
    os.makedirs(os.path.join(result_folder,'parameter_evolution_plot','SHMM'))
    os.makedirs(os.path.join(result_folder,'parameter_evolution_plot','HMMES'))
    os.makedirs(os.path.join(result_folder,'parameter_evolution_plot','SHMMES'))
    plot_param_evolution(hmm_param_evolution,hmm_true_parameter, 'HMM', os.path.join(result_folder,'parameter_evolution_plot'))
    plot_param_evolution(shmm_param_evolution,hmm_true_parameter, 'SHMM', os.path.join(result_folder,'parameter_evolution_plot'))
    plot_param_evolution(hmmes_param_evolution,hmm_true_parameter, 'HMMES', os.path.join(result_folder,'parameter_evolution_plot'))
    plot_param_evolution(shmmes_param_evolution,hmm_true_parameter, 'SHMMES', os.path.join(result_folder,'parameter_evolution_plot'))

    #model result
    mase_model_result = []
    mae_model_result = []
    mse_model_result = []
    for i in range(nb_execution):
        hmm_result = read_pickle(os.path.join(model_folder,f'hmm/execution{i}/model_eval.pkl'))
        shmm_result = read_pickle(os.path.join(model_folder,f'shmm/execution{i}/model_eval.pkl'))
        hmmes_result = read_pickle(os.path.join(model_folder,f'hmmes/execution{i}/model_eval.pkl'))
        shmmes_result = read_pickle(os.path.join(model_folder,f'shmmes/execution{i}/model_eval.pkl'))
        mase_model_result.append([hmm_result[0]['mase_y_pred_mean'],shmm_result[0]['mase_y_pred_mean'],hmmes_result[0]['mase_y_pred_mean'],shmmes_result[0]['mase_y_pred_mean']])
        mae_model_result.append([hmm_result[0]['mae_y_pred_mean'],shmm_result[0]['mae_y_pred_mean'],hmmes_result[0]['mae_y_pred_mean'],shmmes_result[0]['mae_y_pred_mean']])
        mse_model_result.append([hmm_result[0]['mse_y_pred_mean'],shmm_result[0]['mse_y_pred_mean'],hmmes_result[0]['mse_y_pred_mean'],shmmes_result[0]['mse_y_pred_mean']])
    model_mase_result = pd.DataFrame(mase_model_result, columns = ['HMM_mase','SHMM_mase','HMMES_mase','SHMMES_mase'])
    model_mae_result = pd.DataFrame(mae_model_result, columns = ['HMM_mae','SHMM_mae','HMMES_mae','SHMMES_mae'])
    model_mse_result = pd.DataFrame(mse_model_result, columns = ['HMM_mse','SHMM_mse','HMMES_mse','SHMMES_mse'])

    model_result = pd.concat([model_mase_result,model_mae_result,model_mse_result],axis=1)
    write_pickle(model_result,os.path.join(result_folder,'accuracy_metric.pkl'))
    plot_model_prediction(model_folder,'HMM',result_folder)
    plot_model_prediction(model_folder,'SHMM',result_folder)
    plot_model_prediction(model_folder,'HMMES',result_folder)
    plot_model_prediction(model_folder,'SHMMES',result_folder)
