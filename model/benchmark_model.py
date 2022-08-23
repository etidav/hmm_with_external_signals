from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from tbats import TBATS
from tqdm import tqdm

WEEKS_IN_A_YEAR = 52


def fit_predict_single_ts(ts: pd.Series, model_name: str) -> Dict[str, np.array]:
    """
    This method is a method to fit and compute the prediction of a statistical method on a single time series.
    Two statistical methods are enabled with this function, the exponential smoothing (ets) and tbats model (tbats)
    Ets forecast is computed using the 'statsmodels' package.
    Tbats forecast is computed using the 'tbats' package.
    
    Arguments:
    
    - *ts*: a pd.Series containing the value of a single time series.
        
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Only two possibilities : 'ets' for exponential smoothing `tbats` for the tbats model.
        
    Returns:
    
    - *stat_model*: A dict associating the time series name to its statistical forecast. 
    """
    ts_name = ts.name
    if model_name == "ets":
        model = ExponentialSmoothing(
            ts.values, seasonal_periods=WEEKS_IN_A_YEAR, seasonal="add"
        )
        fitted_model = model.fit()
    elif model_name == "thetam":
        model = ThetaModel(ts.values, period=WEEKS_IN_A_YEAR, method="add")
        fitted_model = model.fit()
    elif model_name == "tbats":
        model = TBATS(seasonal_periods=[WEEKS_IN_A_YEAR], n_jobs=1)
        fitted_model = model.fit(ts.values)

    return fitted_model.forecast(WEEKS_IN_A_YEAR)


def fit_predict(data: pd.DataFrame, model_name: str) -> Dict[str, np.array]:
    """
    This method is the method to compute a forecast for each time series present in a pd.DataFrame for
    methods that need to be fit.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
        
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Only two possibilities : 'ets' for exponential smoothing 'tbats' for the tbats model.
        
    Returns:
    
    - *model_prediction*: A dict linking the time series names to their associated statistical forecasts. 
    """

    model_prediction = {}
    for ts_name in data:
        single_pred = fit_predict_single_ts(data[ts_name], model_name)
        model_prediction[ts_name] = single_pred

    return model_prediction


def compute_snaive_prediction(data: pd.DataFrame) -> Dict[str, np.array]:
    """
    This method is the method to compute the `naive` forecast for each time series present in a pd.DataFrame.
    As the `snaive` forecast is a model that only replicated the past year of data, no train is needed.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
  
    Returns:
    
    - *model_prediction*: A dict linking the time series names to their associated statistical forecasts. 
    """

    model_prediction = {}
    for ts_name in data:
        model_prediction[ts_name] = data[ts_name].values[-WEEKS_IN_A_YEAR:]

    return model_prediction


def predict(
    data: pd.DataFrame, model_name: str, time_split: str = None,
) -> pd.DataFrame:
    """
    This method is the main method to compute the forecast for each time series present in a pd.DataFrame.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
    
    - *model_name*: Use this parameter to define what statistical model you want to use.
        Three possibilities : 'snaive' for the naive forecats,
        'ets' for exponential smoothing and 'tbats' for the tbats model.
    
    - *time_split*: a str with the followinf format 'YYYY-MM-DD'. It delimits where stop each time series 
        and start computing a 1 year forecast.
  
    Returns:
    
    - *final_prediction*: A pd.DataFrame with the model predictions.
        In column the time series names.
        In index the time steps.
    """
    if time_split is not None:
        data = data.loc[:time_split]

    if model_name in ["ets", "thetam", "tbats"]:
        prediction = fit_predict(data, model_name)
    elif model_name == "snaive":
        prediction = compute_snaive_prediction(data)
    else:
        raise NotImplementedError(model_name)

    delta = pd.to_datetime(data.index[-1]) - pd.to_datetime(data.index[-2])
    prediction_index = [
        str((pd.to_datetime(data.index[-1]) + delta * (i + 1)).date())
        for i in range(WEEKS_IN_A_YEAR)
    ]
    final_prediction = pd.DataFrame(prediction)
    final_prediction.index = prediction_index

    return final_prediction


def compute_mase(
    y_true: np.array, y_pred: np.array, y_histo: np.array, freq: int = WEEKS_IN_A_YEAR
) -> Tuple[list, int]:
    """
    This method is the method to compute the seasonal Mean Absolute Scaled Errror (MASE).
    
    Arguments:
    
    - *y_true*: a matrix with all the ground truth for each time series 
    
    - *y_pred*: a matrix with all the sequence predictions.
    
    - *y_histo*: a matrix with the all the past historical data for each time series.
        
    - *freq*: By default set to 52. If you change this value to 1, the simple MASE will be computed.
  
    Returns:
    
    - *final_mase*: a float representing the final mase. 
        The average mase is computed on all the seqences.
    """
    denominator = np.mean(np.abs(y_histo[freq:] - y_histo[:-freq]), axis=0)
    numerator = np.mean(np.abs(y_true - y_pred), axis=0)
    all_mase = numerator / denominator
    final_mase = all_mase.mean()
    return all_mase, final_mase


def compute_mse(y_true: np.array, y_pred: np.array) -> Tuple[list, int]:
    """
    This method is the method to compute the seasonal Mean Absolute Scaled Errror (MASE).
    
    Arguments:
    
    - *y_true*: a matrix with all the ground truth for each time series 
    
    - *y_pred*: a matrix with all the sequence predictions.

    Returns:
    
    - *final_mse*: a float representing the final mase. 
        The average mse is computed on all the seqences.
    """
    all_mse = np.mean(np.square(y_true - y_pred), axis=0)
    final_mse = all_mse.mean()
    return all_mse, final_mse


def compute_mae(y_true: np.array, y_pred: np.array) -> Tuple[list, int]:
    """
    This method is the method to compute the seasonal Mean Absolute Scaled Errror (MASE).
    
    Arguments:
    
    - *y_true*: a matrix with all the ground truth for each time series 
    
    - *y_pred*: a matrix with all the sequence predictions.

    Returns:
    
    - *final_mae*: a float representing the final mase. 
        The average mae is computed on all the seqences.
    """
    all_mse = np.mean(np.abs(y_true - y_pred), axis=0)
    final_mse = all_mse.mean()
    return all_mse, final_mse


def eval_model(
    data: pd.DataFrame,
    prediction: pd.DataFrame,
    model_name: str,
    freq: int = WEEKS_IN_A_YEAR,
) -> Dict[str, int]:
    """
    This method is the main method to compute the MASE and the Accuracy on a dataset.
    
    Arguments:
    
    - *data*: A pd.DataFrame gathering single or multiple time series.
        Times series names are provided in columns and the index represents the time steps.
    
    - *final_prediction*: A pd.DataFrame with the model predictions.
        In column the time series names.
        In index the time steps.
    
    - *y_histo*: a matrix with the past 52 historical data for each time series.  
        
    - *freq*: By default set to 52. With a value set to 52, the seasonal MASE will be computed.
        If you change this value to 1, the simple MASE will be computed.
    
    - *threshold*: Threshold that defines the yoy classification rule.
        yoy <= -0.5 -> decreasing time series
        -0.5 <=yoy <= 0.5 -> flat time series
        yoy <= -0.5 -> increasing time series
  
    Returns:
    
    - *paper_result*: a dict with the two final evaluation metric given in the HERMES paper: MASE and Accuracy
    """
    time_split = prediction.index[0]
    ground_truth = data.loc[time_split:].iloc[:freq]
    histo_ground_truth = data.loc[:time_split].iloc[:-1]

    all_mase, final_mase = compute_mase(
        ground_truth.values, prediction.values, histo_ground_truth.values, freq=freq
    )
    all_mae, final_mae = compute_mae(ground_truth.values, prediction.values)
    all_mse, final_mse = compute_mse(ground_truth.values, prediction.values)
    model_result_mase = pd.DataFrame(
        all_mase, index=data.columns, columns=[model_name]
    ).T
    model_result_mase["total"] = final_mase
    model_result_mae = pd.DataFrame(all_mae, index=data.columns, columns=[model_name]).T
    model_result_mae["total"] = final_mae
    model_result_mse = pd.DataFrame(all_mse, index=data.columns, columns=[model_name]).T
    model_result_mse["total"] = final_mse

    return model_result_mase, model_result_mae, model_result_mse
