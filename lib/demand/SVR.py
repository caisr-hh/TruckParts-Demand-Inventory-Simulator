import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os, csv
from datetime import datetime, timedelta

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

class SupportVectorRegression:
    def __init__(self, data_dir: str, feature_series):
        # folder path
        self.data_dir = data_dir

        # SVR folder path
        self.hold_folder = os.path.join(self.data_dir, "SVR")
        if not os.path.exists(self.hold_folder):
            os.mkdir(self.hold_folder)

        # feature series data
        self.feature_series = feature_series

    
    # unified model: regression model based on comprehensive data
    def regression_comprehensive_feature_comparison(self, 
                                 start_date, train_period, 
                                 FEATURES_DICT, TARGET,  
                                 opt=False, opt_params={},
                                 lossfunc_type="RMSE", n_trials=10, 
                                 onehot=True):

        # dataframe to build model
        df_model = self.feature_series.copy()

        # Feature dictionary
        i_feas = 0
        prediction_results = []
        metric_results = {}
        feas_name_list = []
        for key in FEATURES_DICT:
            # distinct parameters
            feature_name = FEATURES_DICT[key]["name"]
            FEATURES = FEATURES_DICT[key]["FEATURE"]
            w = FEATURES_DICT[key]["w"] 
            n_lag = FEATURES_DICT[key]["n_lag"]
            online = FEATURES_DICT[key]["online"]

            # directory 
            feature_dir = os.path.join(self.hold_folder, feature_name)
            if not os.path.exists(feature_dir):
                os.mkdir(feature_dir)

            #-- categorical variables --#
            # one-hot encoding
            if onehot and (('dealer_id' in FEATURES) or ('part_type' in FEATURES)):
                df_model["part_id"] = (df_model['dealer_id'] + "_" + df_model['part_type']).astype("category")
                dummies = pd.get_dummies(df_model,
                                        columns=['part_id'],
                                        prefix=['ID'],
                                        drop_first=False,
                                        dtype=int)
                new_cols = [c for c in dummies.columns if c not in df_model.columns]
                df_model = pd.concat([df_model, dummies[new_cols]], axis=1)
                
                # feature list
                if 'dealer_id' in FEATURES:
                    FEATURES.remove('dealer_id')
                if 'part_type' in FEATURES:
                    FEATURES.remove('part_type')
                oh_cols = [c for c in dummies.columns if c.startswith('ID')]
                FEATURES = oh_cols + FEATURES 

            # integer encoding
            if not onehot and (('dealer_id' in FEATURES) or ('part_type' in FEATURES)): 
                le_dealer = LabelEncoder()
                le_part   = LabelEncoder()
                df_model['dealer_id_num'] = le_dealer.fit_transform(df_model['dealer_id'])
                df_model['part_type_num'] = le_part.fit_transform(df_model['part_type'])

                # feature list
                if 'dealer_id' in FEATURES:
                    FEATURES.remove('dealer_id')
                    FEATURES.append('dealer_id_num')
                if 'part_type' in FEATURES:
                    FEATURES.remove('part_type')
                    FEATURES.append('part_type_num')

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

                return df_group
            df_model = df_model.groupby(['dealer_id', 'part_type'], group_keys=False, observed=True).apply(calc_statistical_feature)

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

            #-- RandomForest --#
            # define train / test data 
            X_train = train[FEATURES]
            y_train = train[TARGET]
            X_test = test[FEATURES]
            y_test = test[TARGET]

            # scaling
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled  = scaler_X.transform(X_test)
        
            # training
            if not opt:
                svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='auto')
            else:
                def objective(trial):
                    kernel = "rbf"
                    C = trial.suggest_loguniform("C", 1, 1e2)
                    epsilon = trial.suggest_loguniform("epsilon", 1e-3, 1.0)
                    gamma = trial.suggest_loguniform("gamma", 1e-4, 10.0)
                    
                    svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

                    # Cross Validation
                    f_val = 0
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    # RSME
                    if lossfunc_type == "RMSE":
                        scores = cross_val_score(svr, X_train_scaled, y_train, cv=cv,
                                                scoring="neg_mean_squared_error",
                                                error_score="raise")
                        mse = -scores.mean()
                        f_val = mse ** 0.5
                    # MAE
                    if lossfunc_type == "MAE":
                        scores = cross_val_score(svr, X_train_scaled, y_train, cv=cv,
                                                scoring="neg_mean_absolute_error",
                                                error_score="raise")
                        f_val = -scores.mean()
                    return f_val
                
                # hyperparameter optimization
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                opt = optuna.create_study(direction="minimize")
                opt.optimize(objective, n_trials=n_trials)
                self.opt_params = opt.best_params
                self.save_tuned_paramters(opt.best_params, lossfunc_type, feature_dir)
                print("The best paramters = ",opt.best_params)
                print("The best loss function value = ",opt.best_value)

                svr = SVR(kernel='rbf',
                           C=opt.best_params['C'], 
                           epsilon=opt.best_params['epsilon'], 
                           gamma=opt.best_params['gamma'])
            
            #- training -#
            svr.fit(X_train_scaled, y_train)
            
            #- offline forecasting -#
            test_off = test.copy()
            test_off['prediction'] = svr.predict(X_test_scaled)

            #- total prediction results -#
            if i_feas == 0:
                i_feas += 1
                prediction_results = test.copy()
                prediction_results[feature_name] = test_off['prediction'].copy()
                feas_name_list.append(feature_name)
                metric_results[feature_name] = self.evaluate_regression_accuracy_each_feature(test_off)
            else:
                i_feas += 1
                prediction_results[feature_name] = test_off['prediction'].copy()
                feas_name_list.append(feature_name)
                metric_results[feature_name] = self.evaluate_regression_accuracy_each_feature(test_off)

            #-- result --#
            # result1: feature importance
            # self.feature_importance(reg)

            # result2: forecasting curve (entire period)
            # self.draw_demand_curve_entire_period(test_off, df_model, TARGET)

            # result3: evaluate regression accuracy
            # df_reg_metrics = self.evaluate_regression_accuracy(test_off)

        self.write_regression_evaluation_data(metric_results, feas_name_list, prediction_results)

        self.write_dealer_part_info(feas_name_list, prediction_results)

        self.write_actual_forecasted_demand(prediction_results, feas_name_list, TARGET)

        self.feas_name_list = feas_name_list

        dealer_part = []
        for (dealer, part), g in prediction_results.groupby(['dealer_id', 'part_type']):
            dealer_part.append([dealer, part])
        self.dealer_part = dealer_part

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
    
    # method: save tuned parameters
    def save_tuned_paramters(self, param_dict: dict, 
                             lossfunc: str, 
                             feature_dir: str):
        param_datapath = os.path.join(feature_dir, "tuned_parameter.csv")
        with open(param_datapath, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            for key, val in param_dict.items():
                writer.writerow([key, val])

    # feature importance
    def feature_importance(self, reg):
        fi = pd.DataFrame(data=reg.feature_importances_,
                          index=reg.feature_names_in_,
                          columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()

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
    
    # function: evaluate regression accuracy (comparison)
    def evaluate_regression_accuracy_comparison(self, metric_results, 
                                                feas_list, pred_results):
        metric_lists=["MAE", "RMSE", "MAPE", "R2", "IAE"]
        for metric in metric_lists:
            labels = []
            values = []
            cols = []
            col_list = ["blue","orange", "green", "purple", "yellow"]
            fig, ax = plt.subplots(figsize=(8,4))
            h = 0
            i = 0
            for i_feas in feas_list:
                if h == 0:
                    f_label = "B"
                else:
                    f_label = "S"
                h+=1
                for (dealer, part), g in pred_results.groupby(['dealer_id', 'part_type']):
                    labels.append(f"{part}\n{dealer}\n({f_label})")
                    values.append(metric_results[i_feas][dealer+part][metric])
                    cols.append(col_list[i])
                i+=1
            bars = ax.bar(labels, values, color=cols)
            ax.bar_label(bars, fmt='%.2f', padding=3)
            ax.set_xlabel("Feature")
            ax.set_ylabel(metric)
            ax.set_title(str(metric))
            plt.show()

    # method: write regression accuracy data
    def write_regression_evaluation_data(self, metric_results, feas_list, pred_results):
        for i_feas in feas_list:
            # creat data frame
            df_feas = pd.DataFrame(columns=["dealer_id","part_type","MAE","RMSE","R2","IAE","SUM_DEMAND"])
            for (dealer, part), g in pred_results.groupby(['dealer_id', 'part_type']):
                MAE = metric_results[i_feas][dealer+part]["MAE"]
                RMSE = metric_results[i_feas][dealer+part]["RMSE"]
                R2 = metric_results[i_feas][dealer+part]["R2"]
                IAE = metric_results[i_feas][dealer+part]["IAE"]
                SUM_D = metric_results[i_feas][dealer+part]["SUM_DEMAND"]
                df_feas.loc[len(df_feas)] = [dealer, part, MAE, RMSE, R2, IAE, SUM_D]

            # write data to csv file
            feature_dir = os.path.join(self.hold_folder, i_feas)
            datapath = os.path.join(feature_dir, "evaluation_metric.csv")
            df_feas.to_csv(datapath, index=False)

    # method: record dealer and part information
    def write_dealer_part_info(self, feas_list, pred_results):
        for i_feas in feas_list:
            # creat data frame
            df_feas = pd.DataFrame(columns=["dealer_id","part_type"])
            for (dealer, part), g in pred_results.groupby(['dealer_id', 'part_type']):
                df_feas.loc[len(df_feas)] = [dealer, part]

            # write data to csv file
            feature_dir = os.path.join(self.hold_folder, i_feas)
            datapath = os.path.join(feature_dir, "dealer_part_info.csv")
            df_feas.to_csv(datapath, index=False)
    
    # method: record actual and forecasted demand
    def write_actual_forecasted_demand(self, pred_results, feas_list, TARGET):
        for i_feas in feas_list:
            for (dealer, part), g in pred_results.groupby(['dealer_id', 'part_type']):
                df = pd.DataFrame(columns=["actual","forecast"])
                df["actual"] = g[[TARGET]].copy()
                df["forecast"] = g[[i_feas]].copy()

                # write data to csv file
                feature_dir = os.path.join(self.hold_folder, i_feas)
                datapath = os.path.join(feature_dir, "demand_"+str(dealer)+"_"
                                        +str(part)+".csv")
                df.to_csv(datapath)
            
            
    # function: draw the demand curve 
    def draw_demand_curve_prediction_period_comparison(self, pred_results, feas_list, TARGET):
        for (dealer, part), g in pred_results.groupby(['dealer_id', 'part_type']):
            ax = g[[TARGET]].plot(figsize=(20,5))
            for i_feas in feas_list:
                g[i_feas].plot(ax=ax, style="-", linewidth=2.5)
            # plt.title(f"demand curve: Dealer={dealer}, Part={part}")
            plt.xlabel("Month", fontsize=14)
            plt.ylabel("Number of demand", fontsize=14)
            plt.ylim(0,10)
            plt.legend()
            plt.tight_layout()
            plt.show()
    

