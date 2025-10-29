import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os, sys, re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import datetime

class StatisticAnalysis:
    def __init__(self, fname: str, sim_failure):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")

        # read data
        self.df = pd.read_csv(os.path.join(self.data_dir, fname), parse_dates=['date'])

        # failure model for each part in each dealer
        self.sim_failure = sim_failure

        # get information
        self.get_info() 

        # total up
        self.total_up()

    # get information
    def get_info(self):
        # part list in each dealer
        self.dealer_parts = (self.df.groupby('dealer_id')['part_type'].unique().reset_index()
                        .rename(columns={'part_type':'part_list'}))

    # total up
    def total_up(self):
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        self.df['failed_flag'] = (self.df['failure'] > 0).astype(int)

        # dealer - part type - failure
        self.agg = self.df.groupby(['dealer_id', 'part_type', 'year', 'month']).agg(
            total_records = ('failure', 'count'),
            total_failures = ('failed_flag', 'sum'),
            sum_failure_counts = ('failure', 'sum')).reset_index()
        self.agg['failure_rate'] = self.agg['total_failures'] / self.agg['total_records']


    def visualization(self):
        chunk_size=5
        for idx, row in self.dealer_parts.iterrows():
            dealer = row['dealer_id']
            parts = row['part_list'].tolist()
            # 部品名を番号順にソート
            parts_sorted = sorted(parts, key=lambda x: int(re.sub(r'\D+', '', x)))
            
            # 10個ずつチャンクに分けて描画
            for i in range(0, len(parts_sorted), chunk_size):
                parts_chunk = parts_sorted[i:i+chunk_size]
                sel = self.agg[(self.agg['dealer_id']==dealer) &
                            (self.agg['part_type'].isin(parts_chunk))]
                
                plt.figure(figsize=(10,6))
                ax = sns.lineplot(data=sel, x='month', y='sum_failure_counts',
                            hue='part_type', hue_order=parts_chunk, marker='o')
                plt.title(f'Monthly Failure Counts at Dealer {dealer} for Part Types {parts_chunk[0]}-{parts_chunk[-1]}')
                plt.xlabel('Month')
                plt.ylabel('Failure Count')
                plt.xticks(range(1,13))


                handles, labels = ax.get_legend_handles_labels()
                new_labels = []
                for lbl in labels:
                    if lbl in self.sim_failure[dealer]:
                        new_labels.append(f"{lbl} ({self.sim_failure[dealer][lbl]['failure_model']}, {self.sim_failure[dealer][lbl]['season_type']} , {self.sim_failure[dealer][lbl]['location']})")
                    else:
                        new_labels.append(lbl)
                ax.legend(handles=handles, labels=new_labels,
                      title='Part Type (Model)',
                      loc='center left', bbox_to_anchor=(1.0, 0.5),
                      frameon=True)
                plt.tight_layout(rect=[0,0,0.85,1])
                plt.show()


# class XGBoost:
#     def __init__(self, fname: str):
#         # folder path
#         current_file = Path(__file__)       # current filepath
#         project_root = current_file.parent.parent
#         self.data_dir = os.path.join(project_root, "data")

#         # read data
#         self.df = pd.read_csv(os.path.join(self.data_dir, fname), parse_dates=['date'])

#     def run1(self, df):
#         # 特徴量設計
#         df['month'] = df['date'].dt.month
#         df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
#         df['dealer_id_enc'] = df['dealer_id'].astype('category').cat.codes
#         df['part_type_enc'] = df['part_type'].astype('category').cat.codes

#         # ラグ特徴量（前日分）
#         df = df.sort_values(['dealer_id','part_type','time'])
#         df['failure_prev1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(1).fillna(0)
#         df['failure_prev2'] = df.groupby(['dealer_id','part_type'])['failure'].shift(2).fillna(0)

#         # ターゲット：翌日の故障件数
#         df['failure_next1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(-1)
#         df = df.dropna(subset=['failure_next1'])

#         # 学習用特徴量とターゲット
#         feature_cols = ['time','month','day_of_week','dealer_id_enc','part_type_enc',
#                         'failure_prev1','failure_prev2']
#         X = df[feature_cols]
#         y = df['failure_next1']

#         # 訓練／テスト分割（時系列考慮）
#         train_size = int(len(df)*0.8)
#         X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#         y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

#         # XGBoost 回帰モデルの構築
#         model = xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=6,
#             subsample=0.8,
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # 予測＆評価
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

#         # 特徴量重要度の表示
#         import matplotlib.pyplot as plt
#         xgb.plot_importance(model, max_num_features=10)
#         plt.show()

#     def run2(self, df):
#         # 前処理
#         df['dealer_id_enc'] = df['dealer_id'].astype('category').cat.codes
#         df['part_type_enc'] = df['part_type'].astype('category').cat.codes

#         # ラグ特徴量（拠点×部品別）
#         df = df.sort_values(['dealer_id','part_type','time'])
#         df['failure_prev1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(1).fillna(0)
#         df['failure_prev2'] = df.groupby(['dealer_id','part_type'])['failure'].shift(2).fillna(0)

#         # ターゲット：翌日の故障件数（time+1）
#         df['failure_next1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(-1)
#         df = df.dropna(subset=['failure_next1'])

#         # 特徴量とターゲットの設定（「月」「曜日」を使わない）
#         feature_cols = [
#             'time',
#             'dealer_id_enc',
#             'part_type_enc',
#             'failure_prev1',
#             'failure_prev2'
#         ]
#         X = df[feature_cols]
#         y = df['failure_next1']

#         # 学習／テストの分割（時系列を考慮）
#         train_size = int(len(df) * 0.8)
#         X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#         y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

#         # XGBoost回帰モデルの構築
#         model = xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # 予測と評価
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"MAE: {mae:.3f}")
#         print(f"RMSE: {rmse:.3f}")
#         print(f"R²: {r2:.3f}")

#         # 特徴量重要度のプロット
#         import matplotlib.pyplot as plt
#         xgb.plot_importance(model, max_num_features=10)
#         plt.show()


#     def run3(self, df):
#         # 2. 前処理・特徴量設計
#         # 経過日数 time をそのまま使う
#         # カテゴリ変数を数値化
#         df['dealer_id_enc'] = df['dealer_id'].astype('category').cat.codes
#         df['part_type_enc'] = df['part_type'].astype('category').cat.codes

#         # 並び替え
#         df = df.sort_values(['dealer_id','part_type','time'])

#         # ラグ特徴量（直前１日・直前２日分の故障数）
#         df['failure_prev1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(1).fillna(0)
#         df['failure_prev2'] = df.groupby(['dealer_id','part_type'])['failure'].shift(2).fillna(0)

#         # ターゲット：翌日の故障件数
#         df['failure_next1'] = df.groupby(['dealer_id','part_type'])['failure'].shift(-1)
#         df = df.dropna(subset=['failure_next1'])

#         # 3. 特徴量・ターゲット設定
#         feature_cols = [
#             'time',
#             'dealer_id_enc',
#             'part_type_enc',
#             'failure_prev1',
#             'failure_prev2'
#         ]
#         X = df[feature_cols]
#         y = df['failure_next1']

#         # 4. 学習データ／テストデータ分割（時系列を考慮して）
#         train_size = int(len(df) * 0.8)
#         X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#         y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

#         # 5. モデル構築：回帰モデル（XGBoost）
#         model = xgb.XGBRegressor(
#             objective='reg:squarederror',  # 回帰用目的関数
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         model.fit(X_train, y_train)

#         # 6. 予測＆評価
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         print(f"MAE: {mae:.3f}")
#         print(f"RMSE: {rmse:.3f}")
#         print(f"R²: {r2:.3f}")

#         # 7. 拠点・部品毎の精度を見る場合
#         df_test = df.iloc[train_size:].copy()
#         df_test['y_true'] = y_test
#         df_test['y_pred'] = y_pred
#         grouped = df_test.groupby(['dealer_id','part_type']).agg(
#             mae_dealer_part = ('y_true','sub').apply(lambda x: mean_absolute_error(x, df_test.loc[x.index,'y_pred'])),
#             count = ('y_true','count')
#         ).reset_index()
#         print(grouped.sort_values('mae_dealer_part', ascending=False).head())

#         # 8. 特徴量重要度の可視化
#         import matplotlib.pyplot as plt
#         xgb.plot_importance(model, max_num_features=10)
#         plt.show()


# class ARIMAmodel:
#     def __init__(self, fname: str):
#         # folder path
#         current_file = Path(__file__)       # current filepath
#         project_root = current_file.parent.parent
#         self.data_dir = os.path.join(project_root, "data")

#         # read data
#         self.df = pd.read_csv(os.path.join(self.data_dir, fname), parse_dates=['date'])
    
#     # run ARIMA model
#     def run(self, df):
#         # 2. 例として「拠点 D00・部品 type0」を使った時系列を抽出
#         dealer = 'D00'
#         part   = 'type0'
#         sub = df[(df['dealer_id']==dealer) & (df['part_type']==part)].sort_values('time')

#         # 3. 純粋な時系列データ化
#         #    index を time にして、応答 y を failure に
#         y_series = sub.set_index('time')['failure']

#         # 4. 学習／テスト分割（例：80% を学習、残り20% をテスト）
#         train_size = int(len(y_series) * 0.8)
#         y_train   = y_series.iloc[:train_size]
#         y_test    = y_series.iloc[train_size:]

#         # 5. ARIMA モデル構築（p,d,q を仮に (1,0,1) とする）
#         model = ARIMA(y_train, order=(1,0,1))
#         model_fit = model.fit()

#         # 6. 予測（テスト期間分）
#         steps = len(y_test)
#         forecast = model_fit.forecast(steps=steps)

#         # 7. 評価
#         mae  = mean_absolute_error(y_test, forecast)
#         rmse = np.sqrt(mean_squared_error(y_test, forecast))
#         # R² は時系列単独の予測では少し解釈に注意が必要ですが、計算できます
#         r2   = r2_score(y_test, forecast)

#         print(f"ARIMA (Dealer={dealer}, Part={part}) → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

#         # 8. 可視化
#         plt.figure(figsize=(10,6))
#         plt.plot(y_train.index, y_train.values, label='Train (actual)')
#         plt.plot(y_test.index,  y_test.values,  label='Test (actual)',   color='grey')
#         plt.plot(y_test.index,  forecast.values, label='Forecast',           color='red', linestyle='--')
#         plt.title(f'ARIMA Forecast for Dealer {dealer}, Part {part}')
#         plt.xlabel('Time (days)')
#         plt.ylabel('Failure count')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
    
#     # forecating demand within [366, 730]
#     def run2(self, df):
#         # 拠点 × 部品別ループ用リスト取得
#         dealers = df['dealer_id'].unique()
#         parts   = df['part_type'].unique()

#         # 結果格納用リスト
#         results = []

#         # 各拠点×部品ごとに処理
#         for dealer in dealers:
#             for part in parts:
#                 sub = df[(df['dealer_id']==dealer) & (df['part_type']==part)].sort_values('time')
#                 if len(sub) < 365*2:
#                     # データが2年分(≒365×2日)未満ならスキップ
#                     continue
                
#                 # 時系列データ化（インデックス：time、値：failure）
#                 series = sub.set_index('time')['failure']
                
#                 # 学習用／予測用に分割
#                 train_days = 365
#                 horizon    = 365
                
#                 y_train  = series.iloc[:train_days]
#                 y_true   = series.iloc[train_days:train_days + horizon]
#                 if len(y_true) < horizon:
#                     # 予測対象期間が不足しているならスキップ
#                     continue
                
#                 # ARIMA モデル構築（p,d,q を簡易に仮設定：例 (1,0,1)）
#                 try:
#                     model = ARIMA(y_train, order=(1,0,1))
#                     res   = model.fit()
#                 except Exception as e:
#                     print(f"ERROR for {dealer}-{part}: {e}")
#                     continue
                
#                 # 予測：次 365 日分
#                 y_pred = res.forecast(steps=horizon)
                
#                 # 評価指標算出
#                 mae  = mean_absolute_error(y_true.values,    y_pred.values)
#                 rmse = np.sqrt(mean_squared_error(y_true.values, y_pred.values))
#                 r2   = r2_score(y_true.values,               y_pred.values)
                
#                 results.append({
#                     'dealer_id': dealer,
#                     'part_type': part,
#                     'MAE': mae,
#                     'RMSE': rmse,
#                     'R2': r2
#                 })
                
#                 # 可視化（代表的な1系列分）
#                 plt.figure(figsize=(10,5))
#                 plt.plot(y_train.index,        y_train.values,        label='Train (past 365 days)')
#                 plt.plot(y_true.index,         y_true.values,         label='True next 365 days', color='grey')
#                 plt.plot(y_true.index,         y_pred.values,         label='ARIMA Forecast 365 days', color='red', linestyle='--')
#                 plt.title(f"Forecast for Dealer={dealer}, Part={part}")
#                 plt.xlabel('Time (days)')
#                 plt.ylabel('Failure count')
#                 plt.legend()
#                 plt.tight_layout()
#                 plt.show()

#         # 結果を DataFrame にまとめて表示
#         df_results = pd.DataFrame(results)
#         print(df_results.sort_values('RMSE').head(10))