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

