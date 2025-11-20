from dataclasses import is_dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os, sys, re

class DemandDataArrange:
    def __init__(self, model:str):
        # ML model
        self.MLmodel = model

        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent.parent
        self.data_dir = os.path.join(project_root, "data")          # data directory
        self.MLmodel_dir = os.path.join(self.data_dir, self.MLmodel) # ML model directory

    # method: load single demand series data
    def load_single_demand_series(self, feas_name: str, dealer: str, part: str):
        # directory, datapath
        feas_dir = os.path.join(self.MLmodel_dir, feas_name)
        datapath = os.path.join(feas_dir, "demand_"+str(dealer)+"_"+str(part)+".csv")

        # load data
        df = pd.read_csv(datapath)

        # start time
        start_time = df["date"][0]
        
        # actual and forecasted demand
        actual_demand = df["actual"].copy().tolist()
        forecasted_demand = df["forecast"].copy().tolist()
        
        return start_time, actual_demand, forecasted_demand
    
    # method: load single demand series data for TSA
    def load_single_demand_series_for_TSA(self, dealer: str, part: str):
        # directory, datapath
        datapath = os.path.join(self.MLmodel_dir, "demand_"+str(dealer)+"_"+str(part)+".csv")

        # load data
        df = pd.read_csv(datapath)

        # start time
        start_time = df['date'][0]
        
        # actual and forecasted demand
        actual_demand = df["actual"].copy().tolist()
        forecasted_demand = df["forecast"].copy().tolist()
        
        return start_time, actual_demand, forecasted_demand

    # method: all dealer and part information 
    def load_all_part_dealer_information(self, feas_name: str):
        # directory, datapath
        self.feas_name = feas_name
        feas_dir = os.path.join(self.MLmodel_dir, feas_name)
        datapath = os.path.join(feas_dir, "dealer_part_info.csv")

        # load data
        df = pd.read_csv(datapath)

        # dealer_part_list
        dealer = df['dealer_id'].copy().tolist()
        part = df['part_type'].copy().tolist()
        self.dealer_part_list = [dealer, part]
        self.n_parts = len(part)
    
    # method: all dealer and part information for time series analysis
    def load_all_part_dealer_information_for_TSA(self):
        # directory, datapath
        datapath = os.path.join(self.MLmodel_dir, "dealer_part_info.csv")

        # load data
        df = pd.read_csv(datapath)

        # dealer_part_list
        dealer = df['dealer_id'].copy().tolist()
        part = df['part_type'].copy().tolist()
        self.dealer_part_list = [dealer, part]
        self.n_parts = len(part)
    
    # method: write all kpis
    def write_kpis_results(self, kpi_results):
        # directory, datapath
        feas_dir = os.path.join(self.MLmodel_dir, self.feas_name)
        datapath = os.path.join(feas_dir, "kpi_results.csv")

        # write results
        kpi_results.to_csv(datapath)
    
    # method: write all kpis for TSA
    def write_kpis_results_for_TSA(self, kpi_results):
        # directory, datapath
        datapath = os.path.join(self.MLmodel_dir, "kpi_results.csv")

        # write results
        kpi_results.to_csv(datapath)
    
    def write_cost_detail_for_TSA(self, detailed_cost_results):
        # directory, datapath
        datapath = os.path.join(self.MLmodel_dir, "detailed_cost_results.csv")

        # write results
        detailed_cost_results.to_csv(datapath)

    # method: summarize results
    def summrize_results(self):
        # directory, datapath
        feas_dir = os.path.join(self.MLmodel_dir, self.feas_name)
        metric_datapath = os.path.join(feas_dir, "evaluation_metric.csv")
        kpis_datapath = os.path.join(feas_dir, "kpi_results.csv")

        #- demand prediction accuracy -#
        metric = pd.read_csv(metric_datapath)
        MAE = metric['MAE'].copy().tolist()
        RMSE = metric['RMSE'].copy().tolist()
        R2 = metric['R2'].copy().tolist()
        IAE = metric['IAE'].copy().tolist()
        SUM_DEMAND = metric['SUM_DEMAND'].copy().tolist()

        # weighted mean of MAE (for all part among all dealers)
        MAE_weight = sum([SUM_DEMAND[i]*MAE[i] for i in range(len(SUM_DEMAND))])/sum(SUM_DEMAND)
        
        # std of MAE
        MAE_std = (sum([SUM_DEMAND[i]*((MAE[i]-MAE_weight)**2) for i in range(len(MAE))])
                   /sum(SUM_DEMAND))**(1/2)

        # mean of RMSE (for all part among all dealers)
        RMSE_mean = sum(RMSE)/len(RMSE)

        # std of RMSE
        RMSE_sigma = (sum([(RMSEi-RMSE_mean)**2 for RMSEi in RMSE])/len(RMSE))**(1/2)

        # mean of R2 (for all part among all dealers)
        R2_mean = sum(R2)/len(R2)

        # std of R2
        R2_sigma = (sum([(R2i-R2_mean)**2 for R2i in R2])/len(R2))**(1/2)

        # mean of IAE (for all part among all dealers)
        IAE_mean = sum(IAE)/len(IAE)

        # std of IAE
        IAE_sigma = (sum([(IAEi-IAE_mean)**2 for IAEi in IAE])/len(IAE))**(1/2)

        #- kpis -#
        kpis = pd.read_csv(kpis_datapath)
        total_costs = kpis['total_costs']
        ISL = kpis['ISL']
        total_stockouts = kpis['total_stockouts']
        total_demand = kpis['total_demand']
        immediate_fulfilled = kpis['immediate_fulfilled']
        backorder_fulfileed = kpis['backorder_fulfileed']

        # Total cost
        TotalCost_all = sum(total_costs)
        # mean of Total Cost
        TotalCost_mean = sum(total_costs)/len(total_costs)
        # std of Total Cost
        TotalCost_std = (sum([(total_costi-TotalCost_mean)**2 for total_costi in total_costs])
                         /len(total_costs))**(1/2)

        # ISL for all parts
        ISL_all = sum(immediate_fulfilled)/sum(total_demand)
        # mean of ISL
        ISL_mean = sum([immediate_fulfilled[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of ISL
        ISL_std = (sum([(immediate_fulfilled[i]/total_demand[i]-ISL_mean)**2 for i in range(len(total_demand))])
                   /len(total_demand))**(1/2)

        # Stockouts rate for all parts
        StockoutsRate_all = sum(total_stockouts)/sum(total_demand)
        # mean of stockout rate
        StockoutsRate_mean = sum([total_stockouts[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of stockouts rate
        StockoutsRate_std = (sum([(total_stockouts[i]/total_demand[i]-StockoutsRate_mean)**2 for i in range(len(total_demand))])
                             /len(total_demand))**(1/2)

        # Cost per demand
        CostPerDemand = sum(total_costs)/sum(total_demand)
        # mean of Cost per demand
        CostPerDemand_mean = sum([total_costs[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of Cost per demand
        CostPerDemand_std = (sum([(total_costs[i]/total_demand[i]-CostPerDemand_mean)**2 for i in range(len(total_demand))])
                              /len(total_demand))**(1/2)

        # record
        summary_df = pd.DataFrame(columns=["MAE_weight",
                                           "sigma(MAE)",
                                           "RMSE_mean",
                                           "sigma(RMSE)",
                                           "R2_mean",
                                           "sigma(R2)",
                                           "IAE_mean",
                                           "sigma(IAE)",
                                           "TotalCost_all",
                                           "TotalCost_mean",
                                           "sigma(TotalCost)",
                                           "ISL_all",
                                           "ISL_mean",
                                           "sigma(ISL)",
                                           "StockoutsRate_all",
                                           "StockoutsRate_mean",
                                           "sigma(StockoutsRate)",
                                           "CostPerDemand",
                                           "CostPerDemand_mean",
                                           "sigma(CostPerDemand)"])
        summary_df.loc[self.MLmodel] = [MAE_weight,
                                        MAE_std,
                                        RMSE_mean,
                                        RMSE_sigma,
                                        R2_mean,
                                        R2_sigma,
                                        IAE_mean,
                                        IAE_sigma,
                                        TotalCost_all,
                                        TotalCost_mean,
                                        TotalCost_std,
                                        ISL_all,
                                        ISL_mean,
                                        ISL_std,
                                        StockoutsRate_all,
                                        StockoutsRate_mean,
                                        StockoutsRate_std,
                                        CostPerDemand,
                                        CostPerDemand_mean,
                                        CostPerDemand_std]
        summary_datapath = os.path.join(feas_dir, "summary_results.csv")
        summary_df.to_csv(summary_datapath)

    # method: summarize results
    def summrize_results_for_TSA(self):
        # directory, datapath
        metric_datapath = os.path.join(self.MLmodel_dir, "evaluation_metric.csv")
        kpis_datapath = os.path.join(self.MLmodel_dir, "kpi_results.csv")

        #- demand prediction accuracy -#
        metric = pd.read_csv(metric_datapath)
        MAE = metric['MAE'].copy().tolist()
        RMSE = metric['RMSE'].copy().tolist()
        R2 = metric['R2'].copy().tolist()
        IAE = metric['IAE'].copy().tolist()
        SUM_DEMAND = metric['SUM_DEMAND'].copy().tolist()

        # weighted mean of MAE (for all part among all dealers)
        MAE_weight = sum([SUM_DEMAND[i]*MAE[i] for i in range(len(SUM_DEMAND))])/sum(SUM_DEMAND)
        
        # std of MAE
        MAE_std = (sum([SUM_DEMAND[i]*((MAE[i]-MAE_weight)**2) for i in range(len(MAE))])
                   /sum(SUM_DEMAND))**(1/2)

        # mean of RMSE (for all part among all dealers)
        RMSE_mean = sum(RMSE)/len(RMSE)

        # std of RMSE
        RMSE_sigma = (sum([(RMSEi-RMSE_mean)**2 for RMSEi in RMSE])/len(RMSE))**(1/2)

        # mean of R2 (for all part among all dealers)
        R2_mean = sum(R2)/len(R2)

        # std of R2
        R2_sigma = (sum([(R2i-R2_mean)**2 for R2i in R2])/len(R2))**(1/2)

        # mean of IAE (for all part among all dealers)
        IAE_mean = sum(IAE)/len(IAE)

        # std of IAE
        IAE_sigma = (sum([(IAEi-IAE_mean)**2 for IAEi in IAE])/len(IAE))**(1/2)

        #- kpis -#
        kpis = pd.read_csv(kpis_datapath)
        total_costs = kpis['total_costs']
        ISL = kpis['ISL']
        total_stockouts = kpis['total_stockouts']
        total_demand = kpis['total_demand']
        immediate_fulfilled = kpis['immediate_fulfilled']
        backorder_fulfileed = kpis['backorder_fulfileed']

        # Total cost
        TotalCost_all = sum(total_costs)
        # mean of Total Cost
        TotalCost_mean = sum(total_costs)/len(total_costs)
        # std of Total Cost
        TotalCost_std = (sum([(total_costi-TotalCost_mean)**2 for total_costi in total_costs])
                         /len(total_costs))**(1/2)

        # ISL for all parts
        ISL_all = sum(immediate_fulfilled)/sum(total_demand)
        # mean of ISL
        ISL_mean = sum([immediate_fulfilled[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of ISL
        ISL_std = (sum([(immediate_fulfilled[i]/total_demand[i]-ISL_mean)**2 for i in range(len(total_demand))])
                   /len(total_demand))**(1/2)

        # Stockouts rate for all parts
        StockoutsRate_all = sum(total_stockouts)/sum(total_demand)
        # mean of stockout rate
        StockoutsRate_mean = sum([total_stockouts[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of stockouts rate
        StockoutsRate_std = (sum([(total_stockouts[i]/total_demand[i]-StockoutsRate_mean)**2 for i in range(len(total_demand))])
                             /len(total_demand))**(1/2)

        # Cost per demand
        CostPerDemand = sum(total_costs)/sum(total_demand)
        # mean of Cost per demand
        CostPerDemand_mean = sum([total_costs[i]/total_demand[i] for i in range(len(total_demand))])/len(total_demand)
        # std of Cost per demand
        CostPerDemand_std = (sum([(total_costs[i]/total_demand[i]-CostPerDemand_mean)**2 for i in range(len(total_demand))])
                              /len(total_demand))**(1/2)

        # record
        summary_df = pd.DataFrame(columns=["MAE_weight",
                                           "sigma(MAE)",
                                           "RMSE_mean",
                                           "sigma(RMSE)",
                                           "R2_mean",
                                           "sigma(R2)",
                                           "IAE_mean",
                                           "sigma(IAE)",
                                           "TotalCost_all",
                                           "TotalCost_mean",
                                           "sigma(TotalCost)",
                                           "ISL_all",
                                           "ISL_mean",
                                           "sigma(ISL)",
                                           "StockoutsRate_all",
                                           "StockoutsRate_mean",
                                           "sigma(StockoutsRate)",
                                           "CostPerDemand",
                                           "CostPerDemand_mean",
                                           "sigma(CostPerDemand)"])
        summary_df.loc[self.MLmodel] = [MAE_weight,
                                        MAE_std,
                                        RMSE_mean,
                                        RMSE_sigma,
                                        R2_mean,
                                        R2_sigma,
                                        IAE_mean,
                                        IAE_sigma,
                                        TotalCost_all,
                                        TotalCost_mean,
                                        TotalCost_std,
                                        ISL_all,
                                        ISL_mean,
                                        ISL_std,
                                        StockoutsRate_all,
                                        StockoutsRate_mean,
                                        StockoutsRate_std,
                                        CostPerDemand,
                                        CostPerDemand_mean,
                                        CostPerDemand_std]
        summary_datapath = os.path.join(self.MLmodel_dir, "summary_results.csv")
        summary_df.to_csv(summary_datapath)


    # method: summarize results
    def summrize_detail_cost_results_for_TSA(self):
        # directory, datapath
        datapath = os.path.join(self.MLmodel_dir, "detailed_cost_results.csv")

        #- detailed cost -#
        detail_cost = pd.read_csv(datapath)
        holding_cost = detail_cost['holding_cost'].copy()
        order_cost = detail_cost['order_cost'].copy()
        transport_cost = detail_cost['transport_cost'].copy()
        return_cost = detail_cost['return_cost'].copy()
        badwill_cost = detail_cost['badwill_cost'].copy()

        # sum of holding cost
        sum_holding = holding_cost.sum()
        print(sum_holding)

        # sum of order cost
        sum_order = order_cost.sum()

        # sum of transport cost
        sum_transport = transport_cost.sum()

        # sum of return cost
        sum_return = return_cost.sum()

        # sum of badwill cost
        sum_badwill = badwill_cost.sum()

        # total cost
        total_cost = sum_holding + sum_order + sum_transport + sum_return + sum_badwill

        

        # record
        summary_df = pd.DataFrame(columns=["hold_sum",
                                           "order_sum",
                                           "transport_sum",
                                           "return_sum",
                                           "badwill_sum",
                                           "total_cost"])
        summary_df.loc[self.MLmodel] = [sum_holding,
                                        sum_order,
                                        sum_transport,
                                        sum_return,
                                        sum_badwill,
                                        total_cost]
        summary_datapath = os.path.join(self.MLmodel_dir, "summary_detailed_cost_results.csv")
        summary_df.to_csv(summary_datapath)

    # method: compute correlation coefficient
    def corrcoef_results(self):
        # directory, datapath
        feas_dir = os.path.join(self.MLmodel_dir, self.feas_name)
        metric_datapath = os.path.join(feas_dir, "evaluation_metric.csv")
        kpis_datapath = os.path.join(feas_dir, "kpi_results.csv")

        #- demand prediction accuracy -#
        metric = pd.read_csv(metric_datapath)
        MAE = metric['MAE'].copy().tolist()
        RMSE = metric['RMSE'].copy().tolist()
        R2 = metric['R2'].copy().tolist()
        IAE = metric['IAE'].copy().tolist()

        #- kpis -#
        kpis = pd.read_csv(kpis_datapath)
        total_costs = kpis['total_costs']
        ISL = kpis['ISL']
        total_stockouts = kpis['total_stockouts']
        total_demand = kpis['total_demand']
        immediate_fulfilled = kpis['immediate_fulfilled']
        backorder_fulfilled = kpis['backorder_fulfileed']

        #- correlation coefficient  -#
        df_corr = pd.DataFrame({
            'MAE'               : MAE,
            'RMSE'              : RMSE,
            'R2'               : R2,
            'IAE'              : IAE,
            'TotalCost'         : total_costs,
            'ISL'               : ISL,
            'TotalStockouts'    : total_stockouts,
            'ImmediateFulfilled': immediate_fulfilled,
            'BackorderFulfilled' : backorder_fulfilled
        })

        corr_matrix = df_corr.corr()
        corr_matrix.to_csv(os.path.join(feas_dir,"CorrCoef.csv"))
        print(corr_matrix)

    # method: compute correlation coefficient
    def corrcoef_results_for_TSA(self):
        # directory, datapath
        metric_datapath = os.path.join(self.MLmodel_dir, "evaluation_metric.csv")
        kpis_datapath = os.path.join(self.MLmodel_dir, "kpi_results.csv")

        #- demand prediction accuracy -#
        metric = pd.read_csv(metric_datapath)
        MAE = metric['MAE'].copy().tolist()
        RMSE = metric['RMSE'].copy().tolist()
        R2 = metric['R2'].copy().tolist()
        IAE = metric['IAE'].copy().tolist()

        #- kpis -#
        kpis = pd.read_csv(kpis_datapath)
        total_costs = kpis['total_costs']
        ISL = kpis['ISL']
        total_stockouts = kpis['total_stockouts']
        total_demand = kpis['total_demand']
        immediate_fulfilled = kpis['immediate_fulfilled']
        backorder_fulfilled = kpis['backorder_fulfileed']

        #- correlation coefficient  -#
        df_corr = pd.DataFrame({
            'MAE'               : MAE,
            'RMSE'              : RMSE,
            'R2'               : R2,
            'IAE'              : IAE,
            'TotalCost'         : total_costs,
            'ISL'               : ISL,
            'TotalStockouts'    : total_stockouts,
            'ImmediateFulfilled': immediate_fulfilled,
            'BackorderFulfilled' : backorder_fulfilled
        })

        corr_matrix = df_corr.corr()
        corr_matrix.to_csv(os.path.join(self.MLmodel_dir,"CorrCoef.csv"))
        print(corr_matrix)

