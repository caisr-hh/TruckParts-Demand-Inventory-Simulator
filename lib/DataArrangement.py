from dataclasses import is_dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd

class DataArrange:
    def __init__(self, events, start_time: datetime | None = None):
        # all demands data
        self.event_data = self.events_to_dataframe(events=events, start_time=start_time)
        
        # demand data for each part
        self.part_data = self.daily_by_part()

    # convert events into pandas dataframe
    def events_to_dataframe(self, events, start_time: datetime | None = None):
        rows = []
        for ev in events:
            row = asdict(ev) if is_dataclass(ev) else dict(ev)
            rows.append(row)
        events_df = pd.DataFrame(rows)
        events_df["date"] = (start_time + pd.to_timedelta(events_df["time"], unit="D")).dt.floor("D")
        return events_df
    
    # daily demand data for each part
    def daily_by_part(self):
        daily_part = self.event_data.groupby(["date", "part_type"]).size().unstack(fill_value=0).sort_index()
        return daily_part











    