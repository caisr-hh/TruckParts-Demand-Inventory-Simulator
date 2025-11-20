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

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"

class Noisemodel:
    def __init__(self, demand_dataname: str, noise: int):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")
        self.demand_dir = os.path.join(self.data_dir, "demand")

        # noise folder path
        self.hold_folder = os.path.join(self.data_dir, "Noise_"+str(noise))
        if not os.path.exists(self.hold_folder):
            os.mkdir(self.hold_folder)
        self.inter_data_dir = os.path.join(self.hold_folder, "intermediate")
        if not os.path.exists(self.inter_data_dir):
            os.mkdir(self.inter_data_dir)

        # read data
        self.df = pd.read_csv(os.path.join(self.demand_dir, demand_dataname), parse_dates=['date'])
        self.df = self.df.set_index('date')
        self.df.index = pd.to_datetime(self.df.index)

        # noise parameter
        self.noise = noise

    
    # unified model: regression model based on comprehensive data
    def regression_comprehensive_feature_comparison(self, start_date, train_period, 
                                TARGET, DEALER_PART=[]):
        # dataframe to build model
        df_model = self.df.copy()

        # optinal data
        if len(DEALER_PART)!=0:
            mask = self.df[["dealer_id", "part_type"]].apply(tuple, axis=1).isin(DEALER_PART)
            df_model = df_model[mask].copy()


        reg_metrics={}
        dealer_part_list = []
        for (dealer, part), g in df_model.groupby(["dealer_id", "part_type"], observed=True):
            dealer_part_list.append([dealer,part])
            series = df_model.loc[(df_model['dealer_id']==dealer) & 
                      (df_model['part_type']==part),TARGET]
            # print(series)

            #-- actural demand data --#
            demand  = series.loc[series.index >= train_period]
            demand.index.name = "date"
            demand = demand.to_frame(name="actual")
            
            #-- add noise --#
            noise = np.random.normal(loc=0, scale=self.noise, size=len(demand))
            demand["forecast"] = demand["actual"] + noise
            demand["forecast"] = demand["forecast"].clip(lower=0)
            
            # prediction metric
            mae  = mean_absolute_error(demand["actual"], demand["forecast"])
            rmse = float(np.sqrt(mean_squared_error(demand["actual"], demand["forecast"])))
            mape = mean_absolute_percentage_error(demand["actual"], demand["forecast"])
            sum_demand = np.sum(demand["actual"])
            reg_metrics[dealer+part] = {}
            reg_metrics[dealer+part]["MAE"] = mae
            reg_metrics[dealer+part]["RMSE"] = rmse
            reg_metrics[dealer+part]["MAPE"] = mape
            reg_metrics[dealer+part]["SUM_DEMAND"] = sum_demand

            # actual/forecasted demand
            self.write_actual_forecasted_demand(demand["actual"], demand["forecast"], dealer, part)

        self.write_regression_evaluation_data(reg_metrics, dealer_part_list)

        self.write_dealer_part_info(dealer_part_list)


    # function: evaluate regression accuracy for each feature
    def evaluate_regression_accuracy_each_feature(self, test):
        reg_metrics={}
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"], observed=True):
            mae  = mean_absolute_error(g['failure'], g['prediction'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction'])))
            mape = mean_absolute_percentage_error(g['failure'], g['prediction'])
            sum_demand = np.sum(g['failure'])
            reg_metrics[dealer+part] = {}
            reg_metrics[dealer+part]["MAE"] = mae
            reg_metrics[dealer+part]["RMSE"] = rmse
            reg_metrics[dealer+part]["MAPE"] = mape
            reg_metrics[dealer+part]["SUM_DEMAND"] = sum_demand
        return reg_metrics

    # method: write regression accuracy data
    def write_regression_evaluation_data(self, metric_results,  dealer_part_list):
        # creat data frame
        df_feas = pd.DataFrame(columns=["dealer_id","part_type","MAE","RMSE","SUM_DEMAND"])
        for arr in dealer_part_list:
            dealer, part = arr[0], arr[1]
            MAE = metric_results[dealer+part]["MAE"]
            RMSE = metric_results[dealer+part]["RMSE"]
            SUM_D = metric_results[dealer+part]["SUM_DEMAND"]
            df_feas.loc[len(df_feas)] = [dealer, part, MAE, RMSE, SUM_D]

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
    def write_actual_forecasted_demand(self, actual, forecast, dealer, part):
            df = pd.DataFrame(columns=["actual","forecast"])
            df["actual"] = actual.copy()
            df["forecast"] = forecast
            # print(df)

            # write data to csv file
            datapath = os.path.join(self.hold_folder, "demand_"+str(dealer)+"_"
                                    +str(part)+".csv")
            df.to_csv(datapath)
            
    

