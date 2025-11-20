::contentReference[oaicite:0]{index=0}
# PartDemand_Simulator
## 🔍 Overview
This repository provides the implementation and supplementary materials used in the paper “[Paper Title]”.
The system consists of three major components:
1. **Demand Generator** — generates synthetic demand time-series data for truck parts under a dealer–truck–part hierarchy.  
2. **Forecasting Model** — builds and evaluates various forecasting techniques, including machine learning and time-series analysis, to predict future parts demand.  
3. **Cost Simulator** — simulates inventory management using demand and forecast data, applies inventory policies, computes costs and KPIs, and supports comparative evaluation of forecasting models.  
4. **Outputs** — the system produces result datasets and visualisations including (but not limited to):  
   - Forecasting accuracy metrics: MAE, RMSE, R2 and IAE, **aggregated across all parts for each forecasting model**.
   - Dealer-part-level KPIs: total costs, immediate service level (ISL), stockouts, total demand, fulfilled/backorder counts, **aggregated across all parts for each forecasting model**.  
   - Cost comparison charts: total cost by model (aggregated across all parts)

## 📁 Repository Structure  
│── src/
│ ├── demand_generator.py # Demand generation module
│ ├── forecasting_model.py # Forecasting module
│ └── … # Additional core modules
│── lib/
│ └── cost/
│   ├── Preprocessor.py
│   ├── simulationLogic.py
│   ├── inventoryPolices.py
│   ├── costTracker.py
│   ├── eventManagement.py
│   ├── orderManagement.py
│   ├── stateManagement.py
│   ├── timeManagement.py
│   ├── timeStamp.py
│   ├── plotMetrics.py
│   └── DemandDataManagement.py
│ └── demand/
│   ├── Environment.py
│   ├── EVENT.py
│   ├── dealer.py
│   ├── truck.py
│   ├── part.py
│   ├── FailureModel.py
│   ├── forecast.py
│   ├── IntermittentAlignmentError.py
│   ├── Noise_model.py
│   ├── Parameter.py
│   ├── RandomForest.py
│   ├── SVR.py
│   └── ARIMA.py
│── notebooks/
│ └── main.ipynb # Main workflow notebook
│── data/
│ ├── demand/ # Generated demand datasets
│ ├── XGBoost/ # Forecasted demand data and simulated results
│ ├── RandomForest/
│ ├── SVR/
│ └── ARIMA/
│── main.ipynb
│── requirements.txt # Python dependencies
└── README.md

## 🚀 Getting Started  
### 1. Clone the repository  
'''bash
git clone https://github.com/yourname/yourrepo.git
'''

### 2. Install dependencies  
'''
pip install -r requirements.txt
'''
(Recommended Python version: Python 3.11)


## 🚀 Running the Workflow  
_Open and execute `notebooks/main.ipynb`. The workflow is structured into three phases:_

### Phase 1: Demand Generator  
'''python
from datetime import datetime

start_time = datetime(2024, 12, 31)
end_time   = datetime(2027, 12, 31)
delta_time = 1
seed       = 3

n_dealers     = 2
n_truck_range = [5, 10]
n_part_range  = [5, 7]

cfg = SimulationConfig(
    start_time = start_time,
    end_time   = end_time,
    delta_time = delta_time
)
sim = Simulator(
    config       = cfg,
    seed         = seed,
    n_dealers    = n_dealers,
    n_truck_range= n_truck_range,
    n_part_range = n_part_range
)
events = sim.run()
'''

### Phase 2: Forecasting
'''python
import forecast as forecast_md
import importlib
importlib.reload(forecast_md)
ForecastMK = forecast_md.ForecastMaker

start_date          = '2025-01-01'
train_days          = 365 * 2
ML_model            = ["XGBoost", "SVR", "RandomForest"]
TSA_model           = ["ARIMA"]
forecast_model_list = ML_model + TSA_model
feature_type_list   = ["basic", "historical"]

ForecastMK = ForecastMK(
    forecast_model_list,
    feature_type_list,
    start_date,
    train_days
)
ForecastMK.mk_forecast_model()
'''

### Phase 3: Cost Simulation & Inventory Policy
''' python
import os, sys
sys.path.append(os.path.abspath('./lib/cost'))

from Preprocessor import Preprocessor
from simulationLogic import SimulationConfig, IntegratedSimulator
from inventoryPolices import StandardInventoryPolicy, InventoryPolicyParams, PredictiveIntervalPolicy, PredictiveInventoryPolicyParams
from plotMetrics import MetricsPlotter
from DemandDataArrangement import DemandDataArrange

lead_time     = 14     # days between placing order and arrival
service_level = 0.95   # desired fill rate
initial_stock = 80     # initial stock per part
review_period = 1      # review frequency (days)

policy_params = InventoryPolicyParams(
    lead_time     = lead_time,
    service_level = service_level,
    review_period = review_period
)
policy = StandardInventoryPolicy(policy_params)

for model in forecast_model_list:
    dda = DemandDataArrange(model=model)
    for feature_type in feature_type_list:
        if model in ML_model:
            dda.load_all_part_dealer_information(feature_type)
        else:
            dda.load_all_part_dealer_information_for_TSA()

        kpi_results = pd.DataFrame(columns=[
            "dealer_id", "part_type", "total_costs", "ISL",
            "total_stockouts", "total_demand",
            "immediate_fulfilled", "backorder_fulfilled"
        ])

        for i in range(dda.n_parts):
            dealer = dda.dealer_part_list[0][i]
            part   = dda.dealer_part_list[1][i]
            if model in ML_model:
                start_time, actual_demand, forecasted_demand = \
                    dda.load_single_demand_series(feature_type, dealer, part)
            else:
                start_time, actual_demand, forecasted_demand = \
                    dda.load_single_demand_series_for_TSA(dealer, part)
            start_time = datetime.strptime(start_time, "%Y-%m-%d")

            forecast_config = SimulationConfig(
                start_time       = start_time,
                forecast_demand  = forecasted_demand,
                actual_demand    = actual_demand,
                inventory_policy = policy,
                initial_stock    = initial_stock
            )
            forecast_simulator = IntegratedSimulator(forecast_config)
            res_forecast = forecast_simulator.run()
            kpi_results.loc[len(kpi_results)] = [
                dealer,
                part,
                res_forecast['kpis']['total_costs'],
                res_forecast['kpis']['immediate_service_level'],
                res_forecast['kpis']['total_stockouts'],
                res_forecast['kpis']['total_demand'],
                res_forecast['kpis']['immediate_fulfilled'],
                res_forecast['kpis']['backorder_fulfilled']
            ]

            config = SimulationConfig(
                start_time       = start_time,
                forecast_demand  = actual_demand,
                actual_demand    = actual_demand,
                inventory_policy = policy,
                initial_stock    = initial_stock
            )
            simulator = IntegratedSimulator(config)
            res_actual = simulator.run()

        if model in ML_model:
            dda.write_kpis_results(kpi_results)
            print(kpi_results)
            dda.summrize_results()
            dda.corrcoef_results()
        else:
            dda.write_kpis_results_for_TSA(kpi_results)
            print(kpi_results)
            dda.summrize_results_for_TSA()
            dda.corrcoef_results_for_TSA()
'''

### Output & Comparison
''' python
import importlib
import ResultComparison as comp_mod
importlib.reload(comp_mod)
ResultComparison = comp_mod.ResultComparison

rscmp = ResultComparison()
noise_list = []
rscmp.visual_multiple_feature_results(feature_type_list, ML_model, TSA_model, noise_list)
'''
