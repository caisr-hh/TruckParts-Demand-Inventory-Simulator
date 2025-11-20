import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os, csv
from datetime import datetime, timedelta
import importlib
import XGBoost as xgb_md
import SVR as svr_md
import ARIMA as arima_md
import RandomForest as rf_md

# XGBoost class
importlib.reload(xgb_md)
XGBoost = xgb_md.XGBoost

# Random Forest
importlib.reload(rf_md)
RandomForest = rf_md.RandomForest

# SVR class
importlib.reload(svr_md)
SVR = svr_md.SupportVectorRegression

# ARIMA class
importlib.reload(arima_md)
ARIMA = arima_md.ARIMAmodel

# FEATURE DATA
FEATURE_DICT ={
    "basic": {"name": "basic", "FEATURE": ['dayofyear', 'month', 'part_type', 'dealer_id'], 
        "w":1, "n_lag":0, "online": False},
    "historical": {"name": "historical", 
        "FEATURE": ['dayofyear', 'month', "mean", "std", "median", "zero_cumulative", 'part_type', 'dealer_id'], 
        "w":90, "n_lag":7, "online": False}
}


class ForecastMaker:
    def __init__(self, model_list, feature_type, start_date, train_days):
        # learning/forecasting period
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.train_period = self.start_date + timedelta(days=int(train_days))
        
        # feature data
        self.feature_type = feature_type
        self.feature_info, self.target_info = self.get_feature_info()

        # forecasting model
        self.model_list = model_list

        # get feature series data
        self.get_feature_series()
    

    # method: get feature data
    def get_feature_info(self, FEATURE_DICT=FEATURE_DICT):
        feature_info = {}
        for feature_type in self.feature_type:
            feature_info[feature_type] = FEATURE_DICT[feature_type]
        target_info = 'failure'
        return feature_info, target_info
    

    # method: get feature series
    def get_feature_series(self):
        # project path
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        
        # "data" directory
        self.data_dir = os.path.join(project_root, "data")          # data directory
        
        # "demand" directory
        self.demand_dir = os.path.join(self.data_dir, "demand")     # demand data directory

        # feature series data
        self.feature_series = pd.read_csv(os.path.join(self.demand_dir, "demand_series.csv"), parse_dates=['date'])
        self.feature_series = self.feature_series.set_index('date')
        self.feature_series.index = pd.to_datetime(self.feature_series.index)


    # method: build model and make prediction
    def mk_forecast_model(self):
        for model in self.model_list:
            # XGBoost
            if model == "XGBoost":          
                # model
                xgb = XGBoost(self.data_dir,
                              self.feature_series.copy())

                # training/forecasting
                xgb.regression_comprehensive_feature_comparison(
                                                start_date=self.start_date, 
                                                train_period=self.train_period, 
                                                FEATURES_DICT=self.feature_info,
                                                TARGET=self.target_info,
                                                opt=False)
            # Random Forest
            if model == "RandomForest":
                # model
                rf = RandomForest(self.data_dir,
                                  self.feature_series.copy())
                
                # traning/forecasting
                rf.regression_comprehensive_feature_comparison(
                                                start_date=self.start_date,
                                                train_period=self.train_period,
                                                FEATURES_DICT=self.feature_info,
                                                TARGET=self.target_info,
                                                opt=False)
                
            # SVR
            if model == "SVR":
                # model
                svr = SVR(self.data_dir,
                          self.feature_series.copy())
                
                # training/forecasting
                svr.regression_comprehensive_feature_comparison(
                                                start_date=self.start_date,
                                                train_period=self.train_period,
                                                FEATURES_DICT=self.feature_info,
                                                TARGET=self.target_info,
                                                opt=False)
            
            # ARIMA
            if model == "ARIMA":
                # model
                arima = ARIMA(self.data_dir,
                              self.feature_series.copy())
                
                # training/forecasting
                arima.regression_comprehensive_feature_comparison(
                                                start_date=self.start_date,
                                                train_period=self.train_period,
                                                TARGET=self.target_info)
