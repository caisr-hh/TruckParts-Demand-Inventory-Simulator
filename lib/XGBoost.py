import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

class XGBoost:
    def __init__(self, fname: str):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")

        # XGBoost folder path
        hold_folder = os.path.join(self.data_dir, "XGBoost")
        if not os.path.exists(hold_folder):
            os.mkdir(hold_folder)
        self.inter_data_dir = os.path.join(hold_folder, "intermediate")
        if not os.path.exists(self.inter_data_dir):
            os.mkdir(self.inter_data_dir)

        # read data
        self.df = pd.read_csv(os.path.join(self.data_dir, fname), parse_dates=['date'])
        self.df = self.df.set_index('date')
        self.df.index = pd.to_datetime(self.df.index)
        # print(self.df)
        # print(self.df.index)

    # unified model: regression with time based trainning (comprehensive data)
    def regression_time_based_training_comprehensive(self, train_period):
        # categorical-variable: make unique part identifier (or learn dealer id and part type separetely)
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        self.df["dealer_id"] = self.df["dealer_id"].astype("category")
        self.df["part_type"] = self.df["part_type"].astype("category")

        # dataframe to use
        df_use = self.df.copy()

        # split train/ test data
        train = df_use.loc[df_use.index <= train_period]
        test = df_use.loc[df_use.index > train_period]

        # feature data
        train = self.create_feature(train)
        test = self.create_feature(test)
        FEATURES = ['dayofyear', 'quarter', 'month', 'dayofweek','part_type','dealer_id']
        TARGET = 'failure'

        # train / test data for model building
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        # XGBoost training
        reg = xgb.XGBRegressor(n_estimators=10000, 
                               early_stopping_rounds=100, 
                               learning_rate=0.3,
                               enable_categorical=True)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)
        
        # XGBoost forecasting
        pred = reg.predict(X_test)
        test['prediction'] = pred

        # result1: feature importance
        self.feature_importance(reg)

        # result2: forecasting result
        metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
            # demand curve: true data VS. prediction data
            df_part = self.df[(self.df['dealer_id']==dealer) & (self.df['part_type']==part)]
            df_part = df_part.merge(g[['prediction']], how='left', left_index=True, right_index=True)
            ax = df_part[[TARGET]].plot(figsize=(15,5))
            df_part['prediction'].plot(ax=ax, style=".")
            plt.title(f"demand curve: Dealer={dealer}, Part={part}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # metric: true data VS. prediction data
            mae  = mean_absolute_error(g['failure'], g['prediction'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction'])))
            r2   = r2_score(g['failure'], g['prediction'])
            # acc  = accuracy_score(g['failure'], g['prediction']) 
            acc = 0
            metrics.append({"dealer": dealer, "part": part, "MAE": mae, "RMSE": rmse, "R2":r2, "ACC":acc})
            # print(">>dealer=", dealer, ", part=", part, ", MAE=", mae, ", RMSE=", rmse, ", R2=", r2, ", ACC=",acc)
        
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.sort_values(["dealer", "part"]))


    # unified model: regression with stastical data based trainning (comprehensive data)
    def regression_statistical_based_training_comprehensive(self, start_date, train_period, w):
        # dataframe for modeling
        df_model = self.df.copy()
        
        # categorical-variable: make unique part identifier (or learn dealer id and part type separetely)
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        df_model["dealer_id"] = df_model["dealer_id"].astype("category")
        df_model["part_type"] = df_model["part_type"].astype("category")

        # time feature data
        df_model = self.create_feature(df_model)

        # stastical feature data
        def calc_statistical_feature(df_group):
            df_group = df_group.copy()
            # mean
            df_group["mean"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
            # std
            df_group["std"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            # median
            df_group["median"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            # min
            df_group["min"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).min().fillna(0)
            # max
            df_group["max"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).max().fillna(0)
            # sum
            df_group["sum"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)

            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False).apply(calc_statistical_feature)
        # print(df_model)

        # split train data / test data
        train = df_model.loc[(df_model.index > start_date) & (df_model.index <= train_period)]
        test = df_model.loc[df_model.index > train_period]
        print(test)
        # print(train)

        #-- XGBoost --#
        FEATURES = ['dayofyear', 'std', 'median', 'month', 'sum', 'part_type','dealer_id', 'zero_failure']
        TARGET = 'failure'
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        # fit #
        reg = xgb.XGBRegressor(n_estimators=10000, 
                               early_stopping_rounds=100, 
                               learning_rate=0.3,
                               enable_categorical=True,
                               tree_method="hist")
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False) 

        # off-line forecasting #
        test['prediction_off'] = reg.predict(X_test)
        # test['prediction_off'] = pred_off

        # result1: feature importance
        self.feature_importance(reg)

        # result2: forecasting result
        metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
            # demand curve: true data VS. prediction data
            df_part = df_model[(df_model['dealer_id']==dealer) & (df_model['part_type']==part)]
            df_part = df_part.merge(g[['prediction_off']], how='left', left_index=True, right_index=True)
            ax = df_part[[TARGET]].plot(figsize=(15,5))
            df_part['prediction_off'].plot(ax=ax, style=".")
            plt.title(f"demand curve: Dealer={dealer}, Part={part}")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # metric: true data VS. prediction data
            mae  = mean_absolute_error(g['failure'], g['prediction_off'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction_off'])))
            r2   = r2_score(g['failure'], g['prediction_off'])
            # acc  = accuracy_score(g['failure'], g['prediction_off'])
            acc = 0
            metrics.append({"dealer": dealer, "part": part, "MAE": mae, "RMSE": rmse, "R2":r2, "ACC":acc})
            # print(">>dealer=", dealer, ", part=", part, ", MAE=", mae, ", RMSE=", rmse, ", R2=", r2, ", ACC=", acc)
        
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.sort_values(["dealer", "part"]))


    # unified model: regression with time based trainning (optional data)
    def regression_time_based_training_optional(self, train_period, options):
        # optinal data
        mask = self.df[["dealer_id", "part_type"]].apply(tuple, axis=1).isin(options)
        df_optional = self.df.loc[mask].copy()

        # categorical-variable: make unique part identifier (or learn dealer id and part type separetely)
        # df_optional["part_id"] = (df_optional['dealer_id'] + "_" + df_optional['part_type']).astype("category")
        df_optional["dealer_id"] = df_optional["dealer_id"].astype("category")
        df_optional["part_type"] = df_optional["part_type"].astype("category")

        # dataframe to use
        df_use = df_optional.copy()

        # split train/ test data
        train = df_use.loc[df_use.index <= train_period]
        test = df_use.loc[df_use.index > train_period]

        # feature data
        train = self.create_feature(train)
        test = self.create_feature(test)
        FEATURES = ['dayofyear', 'quarter', 'month', 'dayofweek','part_type','dealer_id']
        FEATURES = ['dayofyear', 'quarter', 'month', 'part_type','dealer_id']
        TARGET = 'failure'
        # print(train)

        # train / test data for model building
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        # XGBoost training
        reg = xgb.XGBRegressor(n_estimators=10000, 
                               early_stopping_rounds=100, 
                               learning_rate=0.1,
                               enable_categorical=True)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)
        
        # XGBoost forecasting
        pred = reg.predict(X_test)
        test['prediction'] = pred

        # result1: feature importance
        self.feature_importance(reg)

        # result2: forecasting result
        metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
        #     # demand curve: true data VS. prediction data
        #     df_part = self.df[(self.df['dealer_id']==dealer) & (self.df['part_type']==part)]
        #     df_part = df_part.merge(g[['prediction']], how='left', left_index=True, right_index=True)
        #     ax = df_part[[TARGET]].plot(figsize=(15,5))
        #     df_part['prediction'].plot(ax=ax, style=".")
        #     plt.title(f"demand curve: Dealer={dealer}, Part={part}")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

            # metric: true data VS. prediction data
            mae  = mean_absolute_error(g['failure'], g['prediction'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction'])))
            r2   = r2_score(g['failure'], g['prediction'])
            # acc  = accuracy_score(g['failure'], g['prediction'])
            acc = 0
            metrics.append({"dealer": dealer, "part": part, "MAE": mae, "RMSE": rmse, "R2":r2, "ACC":acc})
        #     print(">>dealer=", dealer, ", part=", part, ", MAE=", mae, ", RMSE=", rmse, ", R2=", r2, ", ACC=", acc)
        
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.sort_values(["dealer", "part"]))


    # unified model: regression with stastical data based trainning (optional data)
    def regression_statistical_based_training_optional(self, start_date, train_period, w, options):
        # optinal data
        mask = self.df[["dealer_id", "part_type"]].apply(tuple, axis=1).isin(options)
        df_optional = self.df.loc[mask].copy()
        
        # dataframe for modeling
        df_model = df_optional.copy()
        
        # categorical-variable: make unique part identifier (or learn dealer id and part type separetely)
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        df_model["dealer_id"] = df_model["dealer_id"].astype("category")
        df_model["part_type"] = df_model["part_type"].astype("category")

        # time feature data
        df_model = self.create_feature(df_model)

        # stastical feature data
        def calc_statistical_feature(df_group):
            df_group = df_group.copy()
            # mean
            df_group["mean"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
            # std
            df_group["std"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            # median
            df_group["median"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            # min
            df_group["min"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).min().fillna(0)
            # max
            df_group["max"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).max().fillna(0)
            # sum
            df_group["sum"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)

            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False).apply(calc_statistical_feature)
        # print(df_model)

        # split train data / test data
        train = df_model.loc[(df_model.index > start_date) & (df_model.index <= train_period)]
        test = df_model.loc[df_model.index > train_period]
        # print(train)

        #-- XGBoost --#
        FEATURES = ['dayofyear', 'std', 'median', 'month', 'sum', 'part_type','dealer_id']
        TARGET = 'failure'
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        # fit #
        reg = xgb.XGBRegressor(n_estimators=10000, 
                               early_stopping_rounds=100, 
                               learning_rate=0.3,
                               enable_categorical=True,
                               tree_method="hist")
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False) 

        # off-line forecasting #
        test['prediction_off'] = reg.predict(X_test)
        # test['prediction_off'] = pred_off

        # result1: feature importance
        self.feature_importance(reg)

        # result2: forecasting result
        metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
            # demand curve: true data VS. prediction data
            # df_part = df_model[(df_model['dealer_id']==dealer) & (df_model['part_type']==part)]
            # df_part = df_part.merge(g[['prediction_off']], how='left', left_index=True, right_index=True)
            # ax = df_part[[TARGET]].plot(figsize=(15,5))
            # df_part['prediction_off'].plot(ax=ax, style=".")
            # plt.title(f"demand curve: Dealer={dealer}, Part={part}")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

            # metric: true data VS. prediction data
            mae  = mean_absolute_error(g['failure'], g['prediction_off'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction_off'])))
            r2   = r2_score(g['failure'], g['prediction_off'])
            # acc  = accuracy_score(g['failure'], g['prediction_off'])
            acc = 0
            metrics.append({"dealer": dealer, "part": part, "MAE": mae, "RMSE": rmse, "R2":r2, "ACC":acc})
            # print(">>dealer=", dealer, ", part=", part, ", MAE=", mae, ", RMSE=", rmse, ", R2=", r2, ", ACC=", acc)
        
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.sort_values(["dealer", "part"]))


    # unified model: regression model based on comprehensive data
    def regression_comprehensive(self, start_date, train_period, 
                                 FEATURES, TARGET, w=1, n_lag=0, online=False):
        # dataframe to build model
        df_model = self.df.copy()

        #-- categorical variables --#
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        df_model["dealer_id"] = df_model["dealer_id"].astype("category")
        df_model["part_type"] = df_model["part_type"].astype("category")

        #-- feature data creation --#
        # time-based data
        df_model = self.create_time_based_feature(df_model)

        # statistical data
        def calc_statistical_feature(df_group):
            df_group = df_group.copy()
            # the number of demands in the past w days #
            # mean
            df_group["mean"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
            # std
            df_group["std"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            # median
            df_group["median"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            # median (round it up)
            df_group["median_round"] = df_group["median"].round().fillna(0)
            # min
            df_group["min"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).min().fillna(0)
            # max
            df_group["max"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).max().fillna(0)
            # sum
            df_group["sum"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            
            # the ratio of change in demands #
            # difference between t-(w+1) and t-1 day
            df_group["diff"] = df_group['failure'].shift(1) - df_group['failure'].shift(w+1)
            df_group["diff"].replace([np.inf, -np.inf], 0)
            df_group["diff"].fillna(0)
            # ratio
            df_group["diff_ratio"] = (df_group['failure'].shift(1) - df_group['failure'].shift(w+1)) / (df_group['failure'].shift(w+1) + 1e-6)
            df_group["diff_ratio"].replace([np.inf, -np.inf], 0)
            df_group["diff_ratio"].fillna(0)
            # average difference
            df_group["diff_1d"]=df_group['failure'].diff(1).fillna(0)
            df_group["diff_mean"]=df_group["diff_1d"].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)

            # the consecutive days without failure before the current day #
            last_fail_date = (df_group.index[0] - timedelta(days=1))
            days_since_list = []
            for idx, row in df_group.iterrows():
                current_date = idx      # the current day
                days_since_last_failure = int((current_date - last_fail_date).days) - 1
                days_since_list.append(days_since_last_failure)

                if row["failure"] > 0:    
                    last_fail_date = current_date
            df_group["zero_cumulative"] = days_since_list
            df_group["zero_cumulative"] = df_group["zero_cumulative"].fillna(0)

            # mean_last_failure_w
            df_group["mean_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)

            # max_last_failure_w
            df_group["max_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).max().fillna(0)

            # min_last_failure_w
            df_group["min_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).min().fillna(0)

            # median_last_failure_w
            df_group["med_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            
            # demand days #
            df_group["zero"] = (df_group['failure'] == 0).astype(int)
            df_group["no_zero"] = (df_group['failure'] > 0).astype(int)
            df_group["demand_days"] = df_group['zero'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            df_group["no_demand_days"] = df_group['no_zero'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            
            # # intermediate data
            # dealer_id = df_group['dealer_id'][0]
            # part_id = df_group['part_type'][0]
            # datapath = os.path.join(self.inter_data_dir,
            #                         "feature_data_"+str(dealer_id)+"_"+str(part_id)+".csv")
            # df_group.to_csv(datapath)


            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False, observed=True).apply(calc_statistical_feature)
        # print(df_model[["days_since_last_failure", "group_zero_failure"]])
        # print(df_model["zero_failure"])

        # lag features
        def add_lag_features(df_group):
            df_group=df_group.copy()

            if n_lag>0:
                for lag in range(1, n_lag+1):
                    df_group[f'lag_{lag}'] = df_group['failure'].shift(lag).fillna(0)
            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False, observed=True).apply(add_lag_features)
        if n_lag>0:
            for lag in range(1, n_lag+1):
                FEATURES.append(f'lag_{lag}')

        #-- train data / test data splitting --#
        train = df_model.loc[(df_model.index > start_date) & (df_model.index <= train_period)]
        test = df_model.loc[df_model.index > train_period]

        #-- XGBoost --#
        # define train / test data 
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        #- training -#
        reg = xgb.XGBRegressor(n_estimators=1000, 
                               early_stopping_rounds=50, 
                               learning_rate=0.01,
                               enable_categorical=True)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False)
        
        #- offline forecasting -#
        test_off = test.copy()
        test_off['prediction'] = reg.predict(X_test)

        #-- result --#
        # result1: feature importance
        # self.feature_importance(reg)

        # result2: forecasting curve (entire period)
        # self.draw_demand_curve_entire_period(test_off, df_model, TARGET)

        # result3: evaluate regression accuracy
        # df_reg_metrics = self.evaluate_regression_accuracy(test_off)

        #- online forecasting -#
        if online:
            test_on = test.copy()
            
            # history for initial data
            history = { (dealer, part): list(train[(train['dealer_id']==dealer)&(train['part_type']==part)]
                                        .sort_index()['failure'].tail(w).values)
                                        for dealer, part in test_on.groupby(['dealer_id','part_type'], observed=True).groups.keys() }
            
            predictions = []
            for idx, row in test_on.iterrows():
                d, p = row['dealer_id'], row['part_type']
                hist = history[(d, p)]
                
                # Feature
                for feat in FEATURES:
                    row["mean"] = np.mean(hist[-w:])    # mean
                    row["std"] = np.std(hist[-w:])      # std
                    row["median"] = np.median(hist[-w:])
                    row["median_round"] = np.round(np.median(hist[-w:]))
                    row["min"] = np.min(hist[-w:])
                    row["max"] = np.max(hist[-w:])
                    row["sum"] = np.sum(hist[-w:])
                    row["diff"] = hist[-1]-hist[-w]
                    row["diff_ratio"] = (hist[-1]-hist[-w])/(hist[-w]+1e-06)
                    diff_1days=[]
                    for i in range(w, 1, -1):
                        diff_1day=hist[-i+1]-hist[-i]
                        diff_1days.append(diff_1day)
                    row["diff_mean"] = np.mean(diff_1days)
                    zero_run = 0
                    demand_days = 0
                    no_demand_days = 0
                    for x in reversed(hist[-w:]):
                        if x==0:
                            zero_run += 1
                            no_demand_days += 1
                        else:
                            demand_days += 1
                    row["zero_cumulative"] = zero_run
                    row["demand_days"] = demand_days
                    row["zero_demand_days"] = no_demand_days
                
                # Forecasting
                X_live = row[FEATURES].to_frame().T
                for feas in FEATURES:
                    if feas == "dealer_id":
                        X_live["dealer_id"] = X_live["dealer_id"].astype("category")
                    elif feas == "part_type":
                        X_live["part_type"] = X_live["part_type"].astype("category")
                    elif feas == "dayofyear":
                        X_live["dayofyear"] = X_live["dayofyear"].astype(int)
                    elif feas == "month":
                        X_live["month"] = X_live["month"].astype(int)
                    else:
                        X_live[feas] = X_live[feas].astype(float)
                pred = reg.predict(X_live)[0]
                predictions.append(pred)

                # update history
                hist.append(pred)
                # if len(hist) > w:
                #     hist.pop(0)
                history[(d, p)] = hist
            
            test_on["prediction"] = predictions
            # result2: forecasting curve (entire period)
            self.draw_demand_curve_entire_period(test_on, df_model, TARGET)

            # result3: evaluate regression accuracy
            df_reg_metrics = self.evaluate_regression_accuracy(test_on)


    # unified model: classification model based on comprehensive data
    def classification_comprehensive(self, start_date, train_period, 
                                 FEATURES, TARGET, w=1, n_lag=0, online=True):
        # dataframe to build model
        df_model = self.df.copy()

        #-- categorical variables --#
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        df_model["dealer_id"] = df_model["dealer_id"].astype("category")
        df_model["part_type"] = df_model["part_type"].astype("category")

        #-- feature data creation --#
        # failure flag
        df_model['failure_flag'] = (df_model['failure'] > 0).astype(int)

        # time-based data
        df_model = self.create_time_based_feature(df_model)

        # statistical data
        def calc_statistical_feature(df_group):
            df_group = df_group.copy()
            # the number of demands in the past w days #
            # mean
            df_group["mean"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)
            # std
            df_group["std"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
            # median
            df_group["median"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            # min
            df_group["min"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).min().fillna(0)
            # max
            df_group["max"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).max().fillna(0)
            # sum
            df_group["sum"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            
            # zero cumulative: the consecutive days without failure before the current day #
            last_fail_date = (df_group.index[0] - timedelta(days=1))
            days_since_list = []
            for idx, row in df_group.iterrows():
                current_date = idx      # the current day
                days_since_last_failure = int((current_date - last_fail_date).days) - 1
                days_since_list.append(days_since_last_failure)

                if row["failure"] > 0:    
                    last_fail_date = current_date
            df_group["zero_cumulative"] = days_since_list
            df_group["zero_cumulative"] = df_group["zero_cumulative"].fillna(0)

            # mean_last_failure_w
            df_group["mean_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).mean().fillna(0)

            # max_last_failure_w
            df_group["max_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).max().fillna(0)

            # min_last_failure_w
            df_group["min_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).min().fillna(0)

            # median_last_failure_w
            df_group["med_zero_cumulative"] = df_group["zero_cumulative"].shift(1).rolling(window=w, min_periods=1).median().fillna(0)
            
            # demand days #
            df_group["zero"] = (df_group['failure'] == 0).astype(int)
            df_group["no_zero"] = (df_group['failure'] > 0).astype(int)
            df_group["demand_days"] = df_group['zero'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            df_group["no_demand_days"] = df_group['no_zero'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)

            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False).apply(calc_statistical_feature)
        # print(df_model[["days_since_last_failure", "group_zero_failure"]])
        # print(df_model["zero_failure"])

        # lag features
        def add_lag_features(df_group):
            df_group=df_group.copy()

            if n_lag>0:
                for lag in range(1, n_lag+1):
                    df_group[f'lag_{lag}'] = df_group['failure'].shift(lag).fillna(0)
            return df_group
        df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False).apply(add_lag_features)
        if n_lag>0:
            for lag in range(1, n_lag+1):
                FEATURES.append(f'lag_{lag}')

        #-- train data / test data splitting --#
        train = df_model.loc[(df_model.index > start_date) & (df_model.index <= train_period)]
        test = df_model.loc[df_model.index > train_period]

        #-- XGBoost --#
        # define train / test data 
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        #- training -#
        reg = xgb.XGBClassifier(n_estimators=10000, 
                               learning_rate=0.01,
                               enable_categorical=True)
        reg.fit(X_train, y_train,verbose=False)
        
        #- offline forecasting -#
        test_off = test.copy()
        test_off['prediction'] = reg.predict(X_test)

        #- result -#
        # result1: feature importance
        # self.feature_importance(reg)

        # result2: forecasting curve (entire period)
        # self.draw_demand_curve_entire_period(test, df_model, TARGET)

        # result3: evaluate classification accuracy
        df_reg_metrics = self.evaluate_classification_accuracy(test_off)



    # function: creation of timebased feature data 
    def create_time_based_feature(self, df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear

        # elapsed time since last failure occurs
        df['zero_failure'] = (df['failure']==0).groupby((df['failure']!=0).cumsum()).cumcount() 
        return df
    
    # function: evaluate classification accuracy
    def evaluate_classification_accuracy(self, test):
        reg_metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
            acc = accuracy_score(g['failure'], g['prediction'])
            reg_metrics.append({"dealer": dealer, "part": part, "ACC": acc})
        df_reg_metrics = pd.DataFrame(reg_metrics)
        print(df_reg_metrics.sort_values(["dealer", "part"]))
        return df_reg_metrics

    # function: evaluate regression accuracy
    def evaluate_regression_accuracy(self, test):
        reg_metrics=[]
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"], observed=True):
            mae  = mean_absolute_error(g['failure'], g['prediction'])
            rmse = float(np.sqrt(mean_squared_error(g['failure'], g['prediction'])))
            r2   = r2_score(g['failure'], g['prediction'])
            reg_metrics.append({"dealer": dealer, "part": part, "MAE": mae, "RMSE": rmse, "R2":r2})
        df_reg_metrics = pd.DataFrame(reg_metrics)
        print(df_reg_metrics.sort_values(["dealer", "part"]))
        return df_reg_metrics

    # function: draw the demand curve
    def draw_demand_curve_entire_period(self, df_test, df_all, TARGET):
        for (dealer, part), g in df_test.groupby(['dealer_id', 'part_type']):
            df_part = df_all[(df_all['dealer_id']==dealer) & (df_all['part_type']==part)]
            df_part = df_part.merge(g[['prediction']], how='left', left_index=True, right_index=True)
            ax = df_part[[TARGET]].plot(figsize=(15,5))
            df_part['prediction'].plot(ax=ax, style=".")
            plt.title(f"demand curve: Dealer={dealer}, Part={part}")
            plt.legend()
            plt.tight_layout()
            plt.show()

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


