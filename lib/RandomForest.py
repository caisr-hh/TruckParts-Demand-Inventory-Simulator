import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class RandomForest:
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

    # unified model: regression model based on comprehensive data
    def regression_comprehensive(self, start_date, train_period, 
                                 FEATURES, TARGET, w=1, n_lag=0):
        # dataframe to build model
        df_model = self.df.copy()

        #-- categorical variables --#
        # self.df["part_id"] = (self.df['dealer_id'] + "_" + self.df['part_type']).astype("category")
        le_dealer = LabelEncoder()
        le_part   = LabelEncoder()
        df_model['dealer_id_num']   = le_dealer.fit_transform(df_model['dealer_id'])
        df_model['part_type_num']   = le_part.fit_transform(df_model['part_type'])

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
            # min
            df_group["min"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).min().fillna(0)
            # max
            df_group["max"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).max().fillna(0)
            # sum
            df_group["sum"] = df_group['failure'].shift(1).rolling(window=w, min_periods=1).sum().fillna(0)
            
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

        #-- Random Forest --#
        # define train / test data 
        X_train = train[FEATURES]
        y_train = train[TARGET]
        X_test = test[FEATURES]
        y_test = test[TARGET]
        
        # training
        rf = RandomForestRegressor(n_estimators=10000)
        rf.fit(X_train, y_train)
        
        # test forecasting
        test['prediction'] = rf.predict(X_test)

        #-- result --#
        # result1: feature importance
        self.feature_importance(rf)

        # result2: forecasting curve (entire period)
        self.draw_demand_curve_entire_period(test, df_model, TARGET)

        # result3: evaluate regression accuracy
        df_reg_metrics = self.evaluate_regression_accuracy(test)

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
        for (dealer, part), g in test.groupby(["dealer_id", "part_type"]):
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


