import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

import IntermittentAlignmentError as iae_mod

# IAE class
import importlib
importlib.reload(iae_mod)
IAE = iae_mod.IAE

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"

class ARIMAmodel:
    def __init__(self, data_dir: str, feature_series):
        # folder path
        self.data_dir = data_dir

        # RandomForest folder path
        self.hold_folder = os.path.join(self.data_dir, "ARIMA")
        if not os.path.exists(self.hold_folder):
            os.mkdir(self.hold_folder)

        # feature series data
        self.feature_series = feature_series

    
    # unified model: regression model based on comprehensive data
    def regression_comprehensive_feature_comparison(self, 
                                                    start_date, 
                                                    train_period, 
                                                    TARGET):
        # dataframe to build model
        df_model = self.feature_series.copy()

        reg_metrics={}
        dealer_part_list = []
        for (dealer, part), g in df_model.groupby(["dealer_id", "part_type"], observed=True):
            dealer_part_list.append([dealer,part])
            series = df_model.loc[(df_model['dealer_id']==dealer) & 
                      (df_model['part_type']==part),TARGET]
            # print(series)

            #-- target data splitting --#
            train = series.loc[(series.index > start_date) & (series.index < train_period)]
            test  = series.loc[series.index >= train_period]

            #─ 日次頻度に変換 ─#
            train = train.asfreq('D').fillna(0)

            #─ 予測ホライズンと開始日 ─#
            horizon = 365
            start_for_pred = train_period 
            test_period_dates = pd.date_range(start=start_for_pred, periods=horizon, freq='D')

            #─ ARIMA モデル ─#
            p, d, q = 1, 1, 1
            order = (p, d, q)
            model = ARIMA(train, order=order)
            fit   = model.fit()

            #─ 予測 ─#
            forecast_steps = horizon
            pred = fit.forecast(steps=forecast_steps)

            #─ 結果を Series / DataFrame に格納 ─#
            pred_series   = pd.Series(pred, index=test_period_dates, name="forecast")
            actual_series = series.reindex(test_period_dates).rename("actual")

            df = pd.concat([actual_series, pred_series], axis=1)
            df.index.name = 'date'
            
            # prediction metric
            mae  = mean_absolute_error(actual_series, pred_series)
            rmse = float(np.sqrt(mean_squared_error(actual_series, pred_series)))
            mape = mean_absolute_percentage_error(actual_series, pred_series)
            R2 = r2_score(actual_series, pred_series)
            sum_demand = np.sum(actual_series)
            iae = IAE(actual_series, pred_series)
            result_dict = iae.intermittent_alignment_error()
            reg_metrics[dealer+part] = {}
            reg_metrics[dealer+part]["MAE"] = mae
            reg_metrics[dealer+part]["RMSE"] = rmse
            reg_metrics[dealer+part]["MAPE"] = mape
            reg_metrics[dealer+part]["R2"] = R2
            reg_metrics[dealer+part]["IAE"] = result_dict['intermittent_alignment_error']
            reg_metrics[dealer+part]["SUM_DEMAND"] = sum_demand

            # actual/forecasted demand
            self.write_actual_forecasted_demand(df, dealer, part)

        self.write_regression_evaluation_data(reg_metrics, dealer_part_list)

        self.write_dealer_part_info(dealer_part_list)
            
        # self.write_regression_evaluation_data(metric_results, feas_name_list, prediction_results)

        # self.write_dealer_part_info(feas_name_list, prediction_results)

        # self.write_actual_forecasted_demand(prediction_results, feas_name_list, TARGET)

        # self.feas_name_list = feas_name_list

        # dealer_part = []
        # for (dealer, part), g in prediction_results.groupby(['dealer_id', 'part_type']):
        #     dealer_part.append([dealer, part])
        # self.dealer_part = dealer_part


    # function: evaluate regression accuracy for each feature
    def evaluate_regression_accuracy_each_feature(self, test):
        reg_metrics={}
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"], observed=True):
            mae  = mean_absolute_error(g['failure'], g['prediction'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction'])))
            mape = mean_absolute_percentage_error(g['failure'], g['prediction'])
            R2 = r2_score(g['failure'], g['prediction'])
            sum_demand = np.sum(g['failure'])
            iae = IAE(g['failure'], g['prediction'])
            result_dict = iae.intermittent_alignment_error()
            reg_metrics[dealer+part] = {}
            reg_metrics[dealer+part]["MAE"] = mae
            reg_metrics[dealer+part]["RMSE"] = rmse
            reg_metrics[dealer+part]["MAPE"] = mape
            reg_metrics[dealer+part]["R2"] = R2
            reg_metrics[dealer+part]["IAE"] = result_dict['intermittent_alignment_error']
            reg_metrics[dealer+part]["SUM_DEMAND"] = sum_demand
        return reg_metrics

    # method: write regression accuracy data
    def write_regression_evaluation_data(self, metric_results,  dealer_part_list):
        # creat data frame
        df_feas = pd.DataFrame(columns=["dealer_id","part_type","MAE","RMSE","R2","IAE","SUM_DEMAND"])
        for arr in dealer_part_list:
            dealer, part = arr[0], arr[1]
            MAE = metric_results[dealer+part]["MAE"]
            RMSE = metric_results[dealer+part]["RMSE"]
            R2 = metric_results[dealer+part]["R2"]
            IAE = metric_results[dealer+part]["IAE"]
            SUM_D = metric_results[dealer+part]["SUM_DEMAND"]
            df_feas.loc[len(df_feas)] = [dealer, part, MAE, RMSE, R2, IAE, SUM_D]

            # write data to csv file
            datapath = os.path.join(self.hold_folder, "evaluation_metric.csv")
            df_feas.to_csv(datapath, index=False)

    # method: record dealer and part information
    def write_dealer_part_info(self, dealer_part_list):
        # creat data frame
        df = pd.DataFrame(columns=["dealer_id","part_type"])
        for arr in dealer_part_list:
            dealer, part = arr[0], arr[1]
            df.loc[len(df)] = [dealer, part]

        # write data to csv file
        datapath = os.path.join(self.hold_folder, "dealer_part_info.csv")
        df.to_csv(datapath, index=False)
    
    # method: record actual and forecasted demand
    def write_actual_forecasted_demand(self, df, dealer, part):
            # df = pd.DataFrame(columns=["actual","forecast"])
            # df["actual"] = series.copy()
            # df["forecast"] = pred
            # print(df)

            # write data to csv file
            datapath = os.path.join(self.hold_folder, "demand_"+str(dealer)+"_"
                                    +str(part)+".csv")
            df.to_csv(datapath)
            
    

