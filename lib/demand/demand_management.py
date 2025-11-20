from dataclasses import is_dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os, sys, re
import numpy as np
import matplotlib.pyplot as plt

class DataManager:
    def __init__(self, events, start_time: datetime | None = None):
        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent.parent
        self.data_dir = os.path.join(project_root, "data")          # data directory
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.demand_dir = os.path.join(self.data_dir, "demand")     # demand data directory
        if not os.path.exists(self.demand_dir):
            os.mkdir(self.demand_dir)

        # make dataframe
        self.event_data = self.events_to_dataframe(events=events, start_time=start_time)
        
        # make damand series data
        self.mk_demand_series()

    # method: convert events into pandas dataframe
    def events_to_dataframe(self, events, start_time: datetime | None = None):
        rows = []
        for ev in events:
            row = asdict(ev) if is_dataclass(ev) else dict(ev)
            rows.append(row)
        events_df = pd.DataFrame(rows)
        events_df["date"] = (start_time + pd.to_timedelta(events_df["time"], unit="D")).dt.floor("D")
        return events_df
    
    # method: make demand series data
    def mk_demand_series(self):
        # datapath
        self.series_datapath = os.path.join(self.demand_dir, "demand_series.csv")

        # demand of each part type for each dealer id
        self.part_type_dealer = self.event_data.groupby(["time", "date", "dealer_id", "part_type"])["failure"].sum().reset_index()
        self.part_type_dealer_sort = self.part_type_dealer.sort_values(["part_type", "time", "dealer_id"])
        self.part_type_dealer.to_csv(self.series_datapath, index=False)

    # method: save dealer information
    def save_dealer_info(self, dealer_list):
        # strage
        df_dealer_info = pd.DataFrame(columns=["dealer",
                                               "part_type", 
                                               "failure_model",
                                               "location",
                                               "season_type"])
        # save
        for dealer in dealer_list:
            dealer_id = dealer.dealer_id
            part_info = dealer.PARTS_DICT
            for part_type in part_info.keys():
                info = [dealer_id,
                        part_type,
                        part_info[part_type]["failure_model"],
                        part_info[part_type]["location"],
                        part_info[part_type]["season_type"]]
                df_dealer_info.loc[len(df_dealer_info)] = info 
