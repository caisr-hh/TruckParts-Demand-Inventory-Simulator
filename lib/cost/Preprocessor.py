import pandas as pd
import numpy as np
from calendar import monthrange

class Preprocessor:
    def __init__(self, target_part: str, target_location: str, shift: int = 1, 
                 lags: list = None, rolling_windows: list = None, 
                 rolling_features: list = None, smoothing_window: int = 7,
                 smoothing_method: str = "rolling"):
        """
        Initialize Preprocessor with options for time series data processing.
        
        Args:
            target_part: ID of the target spare part
            target_location: ID of the target location
            shift: Forecast horizon shift
            lags: List of lag periods to use as features
            rolling_windows: List of rolling window sizes
            rolling_features: List of rolling statistics to calculate
            smoothing_window: Window size for smoothing
            smoothing_method: Only "rolling" is supported
        """
        self.target_part = target_part
        self.target_location = target_location
        self.time_col = 'time_bucket'
        self.target_col = 'total_demand_quantity'
        self.lags = lags if lags is not None else []
        self.rolling_windows = rolling_windows if rolling_windows is not None else []
        self.rolling_features = rolling_features if rolling_features is not None else ['mean']
        self.smoothing_window = smoothing_window 
        self.smoothing_method = "rolling"  # Always use rolling regardless of input

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data for target part/location combination"""
        return df[(df['spare_part_id'] == self.target_part) & 
                (df['location_id'] == self.target_location)]
    
    def _convert_time_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time_bucket to datetime and add date column"""
        df_copy = df.copy()
        
        # Convert time_bucket to datetime
        df_copy['date'] = pd.to_datetime(
            df_copy[self.time_col].astype(str),
            format='%Y%m',
            errors='coerce'
        )
        
        # Drop rows with invalid dates
        df_copy = df_copy.dropna(subset=['date'])
        
        # Rename total_demand_quantity to demand
        df_copy['demand'] = df_copy[self.target_col]
        
        return df_copy
    
    def smooth_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling smoothing to daily demand data, particularly at month boundaries.
        
        Args:
            df: DataFrame containing daily data with 'demand' column
        
        Returns:
            DataFrame with smoothed demand values
        """
        if self.smoothing_window <= 1:
            return df  # No smoothing needed
            
        # Make a copy to avoid modifying the original
        smoothed_df = df.copy()
        
        # Calculate original total demand
        original_total_demand = smoothed_df['demand'].sum()
        
        # Apply rolling average
        window = self.smoothing_window
        smoothed_df['demand'] = smoothed_df['demand'].rolling(
            window=window, center=False, min_periods=1
        ).mean()
        
        # Normalize to preserve total demand
        current_total = smoothed_df['demand'].sum()
        if current_total > 0:  # Avoid division by zero
            smoothed_df['demand'] = smoothed_df['demand'] * (original_total_demand / current_total)
        
        return smoothed_df
    
    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to daily frequency with random daily demands"""
        import numpy as np
        daily_data = []
        
        # Process each month's data
        for _, row in df.iterrows():
            date = row['date']
            demand = row['demand']
            
            # Get the number of days in the month
            _, days_in_month = monthrange(date.year, date.month)
            
            # Generate random factors for each day and scale them to match monthly demand
            random_factors = np.random.rand(days_in_month)
            daily_demands = (random_factors / random_factors.sum()) * demand
            
            # Create a row for each day in the month
            for day in range(1, days_in_month + 1):
                daily_data.append({
                    'date': pd.Timestamp(date.year, date.month, day),
                    'month': date.month,
                    'day': day,
                    'demand': daily_demands[day - 1]
                })
        
        # Create a dataframe from the daily data
        result_df = pd.DataFrame(daily_data)
        
        # Sort by date
        result_df = result_df.sort_values('date')
        
        # Apply smoothing if window size > 1
        if self.smoothing_window > 1:
            result_df = self.smooth_daily_data(result_df)
        
        return result_df
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract required features from the dataframe"""
        return df[['month', 'day', 'demand']].reset_index(drop=True)
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features based on the lags list"""
        if not self.lags:
            return df
            
        df_copy = df.copy()
        
        # Create lag features
        for lag in self.lags:
            df_copy[f'demand_lag_{lag}'] = df_copy['demand'].shift(lag)
            
        # Drop rows with NaN values created by the lag
        if len(self.lags) > 0:
            max_lag = max(self.lags)
            df_copy = df_copy.iloc[max_lag:].reset_index(drop=True)
            
        return df_copy
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and median features based on the rolling windows list"""
        if not self.rolling_windows:
            return df
            
        df_copy = df.copy()
        
        # Create rolling features
        for window in self.rolling_windows:
            rolling_obj = df_copy['demand'].rolling(window=window)
            
            if 'mean' in self.rolling_features:
                df_copy[f'demand_roll_mean_{window}'] = rolling_obj.mean()
                
            if 'median' in self.rolling_features:
                df_copy[f'demand_roll_median_{window}'] = rolling_obj.median()
                
            if 'std' in self.rolling_features:
                df_copy[f'demand_roll_std_{window}'] = rolling_obj.std()
                
            if 'min' in self.rolling_features:
                df_copy[f'demand_roll_min_{window}'] = rolling_obj.min()
                
            if 'max' in self.rolling_features:
                df_copy[f'demand_roll_max_{window}'] = rolling_obj.max()
        
        # Drop rows with NaN values created by the rolling window
        if len(self.rolling_windows) > 0:
            max_window = max(self.rolling_windows)
            df_copy = df_copy.iloc[max_window-1:].reset_index(drop=True)
            
        return df_copy
    
    def fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps between start date and end date in the dataframe.
        Focus on demand and date columns, with other features set to 0.
        
        Args:
            df (pd.DataFrame): Input dataframe with date and demand columns
            
        Returns:
            pd.DataFrame: Dataframe with filled gaps
        """
        if df.empty:
            return df
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Get unique part and location combinations
        part_loc_combinations = df[['spare_part_id', 'location_id']].drop_duplicates()
        
        # Initialize an empty dataframe to store results
        result_df = pd.DataFrame()
        
        # Process each part-location combination separately
        for _, row in part_loc_combinations.iterrows():
            part_id = row['spare_part_id']
            loc_id = row['location_id']
            
            # Filter data for this part-location combination
            subset = df[(df['spare_part_id'] == part_id) & (df['location_id'] == loc_id)]
            
            if subset.empty:
                continue
                
            # Get min and max dates for this part-location
            min_date = subset['date'].min()
            max_date = subset['date'].max()
            
            # Create a complete date range at monthly frequency
            # Extract the first day of each month to match the format in the data
            date_range = pd.date_range(
                start=min_date.replace(day=1),
                end=max_date.replace(day=1),
                freq='MS'  # Month Start frequency
            )
            
            # Create a template dataframe with all dates
            template_df = pd.DataFrame({'date': date_range})
            
            # Merge with actual data
            merged = pd.merge(template_df, subset, on='date', how='left')
            
            # Fill NaN values
            # For demand, fill with mean of existing values
            if subset['demand'].notna().any():
                mean_demand = subset['demand'].mean()
                merged['demand'] = merged['demand'].fillna(mean_demand)
            else:
                merged['demand'] = merged['demand'].fillna(0)  # Fallback if no valid demand values
            
            # For other columns, fill with appropriate values
            # First, fill spare_part_id and location_id
            merged['spare_part_id'] = part_id
            merged['location_id'] = loc_id
            
            # For time_bucket, create from date (YYYYMM format)
            merged['time_bucket'] = merged['date'].dt.strftime('%Y%m').astype(int)
            
            # For time_bucket_type, use 'MO' (monthly)
            merged['time_bucket_type'] = 'MO'
            
            # For all other numeric columns, fill with 0
            numeric_cols = [
                'total_demand_lines', 'total_demand_quantity',
                'repetitive_demand_lines', 'repetitive_demand_quantity',
                'exceptional_demand_lines', 'exceptional_demand_quantity',
                'urgent_demand_lines', 'urgent_demand_quantity',
                'non_urgent_demand_lines', 'non_urgent_demand_quantity'
            ]
            
            for col in numeric_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)
            
            # Append to result
            result_df = pd.concat([result_df, merged], ignore_index=True)
            
        return result_df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process full dataset for forecasting"""
        # Step 1: Filter data for target part/location
        filtered_df = self._filter_data(df)
        
        # Step 2: Convert time to datetime
        datetime_df = self._convert_time_to_datetime(filtered_df)
        
        filled_df = self.fill_gaps(datetime_df)
        
        # Step 3: Resample to daily frequency with smoothing
        daily_df = self._resample_to_daily(filled_df)
        
        # Step 4: Extract required features
        features_df = self._extract_features(daily_df)
        
        # Step 5: Create lag features if specified
        lag_df = self._create_lag_features(features_df)
        
        # Step 6: Create rolling features if specified
        result_df = self._create_rolling_features(lag_df)
        
        return result_df

   