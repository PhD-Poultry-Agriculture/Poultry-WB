import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

class RossGroups(Enum):
    CONTROL = 'Control'
    WIDE_BREAST = 'WB'

class DataManager:
    
    def __init__(self, raw_data):
        self._CHICKENS_PER_GROUP = 8
        self.data = raw_data
        self.data_groups = {
            RossGroups.CONTROL: {},
            RossGroups.WIDE_BREAST : {}
        }
        self._preprocess()

    def _preprocess(self):
        if self.data is not None:
            print("Preprocessing data...")
            
            df_z_scaled = self.data.copy()            
            df_z_scaled = self._normalize_columns(df_z_scaled, 'C')
            df_z_scaled = self._normalize_columns(df_z_scaled, 'B')

            df_z_scaled = df_z_scaled.drop(['Tags', 'Formula'], axis=1)
            df_T = df_z_scaled.T
            df_T.columns = df_T.iloc[0]
            df_T.index.name = 'Timestamp'

            df_T = df_T.drop(df_T.index[0])
           
            self._process_group(df_T, RossGroups.CONTROL, 'C')
            self._process_group(df_T, RossGroups.WIDE_BREAST, 'WB')
            
            # print(self.data_groups[RossGroups.CONTROL]['C2'].tail(6))
            # print(self.data_groups[RossGroups.WIDE_BREAST]['WB2'].tail(6))
        else:
            print("No data to preprocess.")

    def plot_averages(self):
        """Plot all averages for 'C' group in blue and 'WB' group in red."""
        plt.figure(figsize=(10, 6))
        key = 'T1C1'
        # Plot averages for the 'C' (Control) group in blue
        for index in range(1, self._CHICKENS_PER_GROUP):
            control_index = 'C' + str(index)
            if control_index in self.data_groups[RossGroups.CONTROL]:
                avg_row = self.data_groups[RossGroups.CONTROL][control_index].loc[key]
                first_column_avg = avg_row.iloc[0]
                plt.scatter(index, first_column_avg, label=f'Control {control_index}', color='blue')

        # Plot averages for the 'WB' (Wide Breast) group in red
        for index in range(1, self._CHICKENS_PER_GROUP):
            wb_index = 'WB' + str(index)
            if wb_index in self.data_groups[RossGroups.WIDE_BREAST]:
                avg_row = self.data_groups[RossGroups.WIDE_BREAST][wb_index].loc[key]
                first_column_avg = avg_row.iloc[0]
                plt.scatter(index, first_column_avg, label=f'Wide Breast {wb_index}', color='red')


        plt.title('Averages for Control (C) and Wide Breast (WB) Groups')
        plt.xlabel('Chemical Compounds')
        plt.ylabel('Average Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _normalize_columns(self, df, chicken_ref):
        """Helper method to normalize columns that contains the chicken's reference."""
        columns = [col for col in df.columns if chicken_ref in col]
        if columns:
            df[columns] = df[columns].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        return df

    def _process_group(self, df_T, group_name, prefix):
        """Helper method to process and append average and median rows for a group."""
        for index in range(1, self._CHICKENS_PER_GROUP):
            control_index = prefix + str(index)
            df_T_filtered = df_T[df_T.index.str.contains(control_index, na=False)]
            
            mean_row = df_T_filtered.mean().to_frame().T
            median_row = df_T_filtered.median().to_frame().T

            mean_row.index = ['Average']
            median_row.index = ['Median']

            df_with_stats = pd.concat([df_T_filtered, mean_row, median_row])
            self.data_groups[group_name][control_index] = df_with_stats


#%% End
