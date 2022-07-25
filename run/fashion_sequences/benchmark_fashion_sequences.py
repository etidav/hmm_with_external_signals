import argparse
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from tbats import TBATS
from tqdm import tqdm
from model.benchmark_model import predict, eval_model
from run.utils import (
    read_json,
    read_pickle,
    read_yaml,
    write_json,
    write_pickle,
    write_yaml,
)

WEEKS_IN_A_YEAR = 52

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulated data using simple HMM")
    parser.add_argument(
        "--main_folder", type=str, help="where to store the model files", required=True
    )
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    np.random.seed(1)
    
    main_folder = args.main_folder
    if main_folder[-1] != "/":
        main_folder += "/"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    else:
        if os.path.isdir(main_folder) and len(os.listdir(main_folder)) > 0:
            raise ValueError(
                "The model folder {} already exists and isn't empty. you need to use an other model folder".format(
                    main_folder
                )
            )
    config_dict = vars(args)
    write_json(config_dict,os.path.join(main_folder, "config.json"))
    
    all_y_data = pd.read_csv('data/f1_main_10_sequences.csv',index_col=0)
    trend_list = [
        'br_female_shoes_262',
        'br_female_texture_59',
        'br_female_texture_82',
        'eu_female_outerwear_177',
        'eu_female_top_325',
        'eu_female_top_394',
        'eu_female_texture_80',
        'us_female_outerwear_171',
        'us_female_shoes_76',
        'us_female_top_79'
    ]
    data = all_y_data[trend_list]
    paper_result = {}
    paper_result['MASE'] = []
    paper_result['MAE'] = []
    paper_result['MSE'] = []
    for model_name in tqdm(['snaive','ets','tbats','thetam']):
        model_pred = predict(data, model_name, time_split='2018-12-31')
        model_result = eval_model(data, model_pred, model_name,freq=52)
        paper_result['MASE'].append(model_result[0])
        paper_result['MAE'].append(model_result[1])
        paper_result['MSE'].append(model_result[2])
        
    paper_result['MASE'] = pd.concat(paper_result['MASE'])
    paper_result['MAE'] = pd.concat(paper_result['MAE'])
    paper_result['MSE'] = pd.concat(paper_result['MSE'])
    write_pickle(paper_result['MASE'], os.path.join(main_folder,'benchmark_result_mase.pkl'))
    write_pickle(paper_result['MAE'], os.path.join(main_folder,'benchmark_result_mae.pkl'))
    write_pickle(paper_result['MSE'], os.path.join(main_folder,'benchmark_result_mse.pkl'))
    
