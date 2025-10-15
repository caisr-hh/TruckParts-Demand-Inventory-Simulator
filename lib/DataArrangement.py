from dataclasses import is_dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os, sys, re

class DataArrange:
    def __init__(self, events, start_time: datetime | None = None):
        # all demands data
        self.event_data = self.events_to_dataframe(events=events, start_time=start_time)

        # folder path
        current_file = Path(__file__)       # current filepath
        project_root = current_file.parent.parent
        self.data_dir = os.path.join(project_root, "data")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        # organize data
        self.organize1_data()

    # convert events into pandas dataframe
    def events_to_dataframe(self, events, start_time: datetime | None = None):
        rows = []
        for ev in events:
            row = asdict(ev) if is_dataclass(ev) else dict(ev)
            rows.append(row)
        events_df = pd.DataFrame(rows)
        events_df["date"] = (start_time + pd.to_timedelta(events_df["time"], unit="D")).dt.floor("D")
        return events_df
    
    # organize data
    def organize1_data(self):
        # demand of each part type for each dealer id
        self.part_type_dealer = self.event_data.groupby(["time", "date", "dealer_id", "part_type"])["failure"].sum().reset_index()
        self.part_type_dealer_sort = self.part_type_dealer.sort_values(["part_type", "time", "dealer_id"])
        self.part_type_dealer.to_csv(os.path.join(self.data_dir, "MLlearning_by_parttype_dealer.csv"), index=False)




# class DataArrange:
#     def __init__(self, events, start_time: datetime | None = None):
#         # all demands data
#         self.event_data = self.events_to_dataframe(events=events, start_time=start_time)
#         # print(self.event_data)
        
#         # demand data on each date for each part
#         self.data_part = self.failure_by_part_daily()
#         # self.data_part = self.failure_by_part_time()

#         # demand data on each date for each part of each truck
#         self.data_partid = self.failure_by_partid_daily()
#         # self.data_partid = self.failure_by_partid_time()

#         # folder path
#         current_file = Path(__file__)       # current filepath
#         project_root = current_file.parent.parent
#         self.data_dir = os.path.join(project_root, "data")
#         if not os.path.exists(self.data_dir):
#             os.mkdir(self.data_dir)
        
#         # learning data for ML
#         self.mk_learning_data_by_part()
#         self.mk_learning_data_by_partid()


#     # convert events into pandas dataframe
#     def events_to_dataframe(self, events, start_time: datetime | None = None):
#         rows = []
#         for ev in events:
#             row = asdict(ev) if is_dataclass(ev) else dict(ev)
#             rows.append(row)
#         events_df = pd.DataFrame(rows)
#         events_df["date"] = (start_time + pd.to_timedelta(events_df["time"], unit="D")).dt.floor("D")
#         return events_df
    
#     # demand data for each part (date)
#     def failure_by_part_daily(self):
#         # daily_part = self.event_data.groupby(["date", "part_type"]).size().unstack(fill_value=0).sort_index()
#         date_part = self.event_data.groupby(["date", "part_type"])['failure'].sum().unstack(fill_value=0)
#         date_part = date_part.sort_index(axis=1,key=lambda s: s.str.extract(r'(\d+)').astype(int)[0])
#         return date_part
    
#     # demand data for each part (time)
#     def failure_by_part_time(self):
#         # daily_part = self.event_data.groupby(["date", "part_type"]).size().unstack(fill_value=0).sort_index()
#         time_part = self.event_data.groupby(["time", "part_type"])['failure'].sum().unstack(fill_value=0)
#         return time_part
    
#     # failure data for each part of each truck (date)
#     def failure_by_partid_daily(self):
#         date_partid = self.event_data.groupby(["date", "part_id"])['failure'].sum().unstack(fill_value=0)
#         sorted_cols = sorted(date_partid.columns,key=lambda x: int(re.search(r"type(\d+)", x).group(1)) if re.search(r"type(\d+)", x) else float("inf"))
#         date_partid = date_partid[sorted_cols]
#         return date_partid
    
#     # failure data for each part of each truck (time)
#     def failure_by_partid_time(self):
#         time_partid = self.event_data.groupby(["time", "part_id"])['failure'].sum().unstack(fill_value=0)
#         return time_partid
    
#     # Learning data
#     def mk_learning_data_by_part(self):
#         MLdata_by_part = self.event_data[["date", "time", "part_type", "failure"]].sort_values(by=["part_type"]).reset_index(drop=True).copy()
#         MLdata_by_part = (MLdata_by_part.groupby(["date", "time", "part_type"], as_index=False).agg(failure=('failure','sum'))).sort_values(by=["part_type", "time"])
#         MLdata_by_part.to_csv(os.path.join(self.data_dir, "MLlearning_by_part.csv"), index=False)
        
    
#     # Learning data
#     def mk_learning_data_by_partid(self):
#         MLdata_by_partid = self.event_data[["date", "time", "part_age", "part_id", "failure"]].sort_values(by=["part_id","time"]).reset_index(drop=True).copy()
#         MLdata_by_partid.to_csv(os.path.join(self.data_dir, "MLlearning_by_partid.csv"), index=False)
