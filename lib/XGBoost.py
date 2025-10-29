import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb

class XGBoost:
    def __init__(self, fname: str):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")

        # read data
        self.df = pd.read_csv(os.path.join(self.data_dir, fname), parse_dates=['date'])
        self.df = self.df.set_index('date')
        self.df.index = pd.to_datetime(self.df.index)
        # print(self.df)
        # print(self.df.index)
    
    # training and prediction for regression
    def train_predict_regression(self):
        # get dealer / part_type list
        dealers = self.df['dealer_id'].unique()
        parts   = self.df['part_type'].unique()

        # result list
        results = []

        # processing for each part type in each dealer
        start_date = datetime.strptime('2025-01-02', '%Y-%m-%d')
        train_period = start_date + timedelta(days=365)
        for dealer in dealers:
            i = 0
            for part in parts:
                if i < 3:
                    df_part = self.df[(self.df['dealer_id']==dealer) & (self.df['part_type']==part)].sort_values('time')

                    # train / test plot
                    train = df_part.loc[df_part.index <= train_period]
                    test = df_part.loc[df_part.index > train_period]

                    # fig, ax = plt.subplots(figsize=(15,5))
                    # train["failure"].plot(ax=ax, label="TrainData")
                    # test["failure"].plot(ax=ax, label="TestData")
                    # ax.axvline(train_period, color='black', ls='-')
                    # plt.show()

                    # create feature data
                    train = self.create_feature(train)
                    test = self.create_feature(test)
                    FEATURES = ['dayofyear', 'quarter', 'month', 'zero_failure']
                    FEATURES = ['dayofyear', 'quarter', 'month']
                    FEATURES = ['dayofyear']
                    TARGET = 'failure'

                    # train / test data
                    X_train = train[FEATURES]
                    y_train = train[TARGET]
                    X_test = test[FEATURES]
                    y_test = test[TARGET]
                    # print(y_train[:100])
                    
                    # XGBoost training
                    reg = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=100, 
                                        learning_rate=0.1)
                    reg.fit(X_train, y_train,
                            eval_set=[(X_train, y_train), (X_test, y_test)],
                            verbose=False)
                    
                    # XGBoost forecast on test
                    pred = reg.predict(X_test)
                    test['prediction'] = pred
                    
                    
                    # result1: feature importance
                    self.feature_importance(reg)

                    # result2: forecasting result
                    self.forecasting_result(df_part,test,TARGET)

                    # result3: metrics
                    mae  = mean_absolute_error(test['failure'], test['prediction'])
                    rmse = float(np.sqrt(mean_squared_error(test['failure'], test['prediction'])))
                    r2   = r2_score(test['failure'], test['prediction'])
                    results.append({
                        'dealer_id': dealer,
                        'part_type': part,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2': r2
                    })
                    print('dealer_id: ', dealer,
                        'part_type: ', part,
                        'MAE:', mae,
                        'RMSE: ', rmse,
                        'R2: ', r2)
                    i += 1

                
                
        print(results)
        self.metric(results)
    

    # training and prediction for classification
    def train_predict_classification(self):
        # get dealer / part_type list
        dealers = self.df['dealer_id'].unique()
        parts   = self.df['part_type'].unique()

        # result list
        results = []

        # processing for each part type in each dealer
        start_date = datetime.strptime('2025-01-02', '%Y-%m-%d')
        train_period = start_date + timedelta(days=365)
        for dealer in dealers:
            i = 0
            for part in parts:
                if i < 2:
                    i += 0 
                    df_part = self.df[(self.df['dealer_id']==dealer) & (self.df['part_type']==part)].sort_values('time')

                    # occurrence information
                    df_part['failure_flag'] = (df_part['failure'] > 0).astype(int)

                    # train / test plot
                    train = df_part.loc[df_part.index <= train_period]
                    test = df_part.loc[df_part.index > train_period]

                    # fig, ax = plt.subplots(figsize=(15,5))
                    # train["failure"].plot(ax=ax, label="TrainData")
                    # test["failure"].plot(ax=ax, label="TestData")
                    # ax.axvline(train_period, color='black', ls='-')
                    # plt.show()

                    # create feature data
                    train = self.create_feature(train)
                    test = self.create_feature(test)
                    # FEATURES = ['dayofyear', 'quarter', 'month', 'zero_failure']
                    FEATURES = ['dayofyear', 'quarter', 'month']
                    FEATURES = ['dayofyear']
                    TARGET = 'failure_flag'

                    # train / test data
                    X_train = train[FEATURES]
                    y_train = train[TARGET]
                    X_test = test[FEATURES]
                    y_test = test[TARGET]
                    
                    # XGBoost classifier training
                    cla = xgb.XGBClassifier(objective='binary:logistic',
                                        use_label_encoder=False,
                                        eval_metric='logloss',
                                        n_estimators=10000, 
                                        early_stopping_rounds=100, 
                                        learning_rate=0.1)
                    cla.fit(X_train, y_train,
                            eval_set=[(X_train, y_train), (X_test, y_test)],
                            verbose=False)
                    
                    # XGBoost forecast on test
                    pred = cla.predict(X_test)
                    test['prediction'] = pred
                    
                    # result1: feature importance
                    # self.feature_importance(reg)

                    # result2: forecasting result
                    self.forecasting_result(df_part,test,TARGET)

                    # result3: metrics
                    acc  = accuracy_score(test['failure_flag'], test['prediction'])
                    results.append({
                        'dealer_id': dealer,
                        'part_type': part,
                        'ACC': acc
                    })
                    print('dealer_id: ', dealer,
                        'part_type: ', part,
                        'ACC: ', acc)

                
                
        print(results)
        self.metric(results)
        
    # create feature
    def create_feature(self, df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear

        # elapsed time since last failure occurs
        df['zero_failure'] = (df['failure']==0).groupby((df['failure']!=0).cumsum()).cumcount() 
        return df

    # feature importance
    def feature_importance(self, reg):
        fi = pd.DataFrame(data=reg.feature_importances_,
                          index=reg.feature_names_in_,
                          columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()

    # forecating result
    def forecasting_result(self, df_part, test, TARGET):
        df_part = df_part.merge(test[['prediction']], how='left', left_index=True, right_index=True)
        ax = df_part[[TARGET]].plot(figsize=(15,5))
        df_part['prediction'].plot(ax=ax, style=".")
        plt.legend(['True Data', "Predicitions"])
        ax.set_title("Raw Dat and Predicition")
        plt.show()
    
    # metric
    def metric(self, results):
        res = pd.DataFrame(results)
        res = res.sort_values(by="part_type").reset_index(drop=True)
        print(res)

