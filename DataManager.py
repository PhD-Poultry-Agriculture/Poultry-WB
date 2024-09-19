import pandas as pd
import numpy as np
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
        self.data_groups[RossGroups.CONTROL]['C-avg'] = pd.DataFrame()
        self.data_groups[RossGroups.WIDE_BREAST]['WB-avg'] = pd.DataFrame()
        self.data_groups[RossGroups.CONTROL]['C-median'] = pd.DataFrame()
        self.data_groups[RossGroups.WIDE_BREAST]['WB-median'] = pd.DataFrame()
        self._preprocess()

    def _preprocess(self):
        if self.data is not None:
            print("Preprocessing data...")
            
            df_z_scaled = self.data.copy()            
            # df_z_scaled = self._normalize_columns(df_z_scaled, 'C')
            # df_z_scaled = self._normalize_columns(df_z_scaled, 'B')

            df_z_scaled = df_z_scaled.drop(['Tags', 'Formula'], axis=1)
            df_T = df_z_scaled.T
            df_T.columns = df_T.iloc[0]
            df_T.index.name = 'Timestamp'

            df_T = df_T.drop(df_T.index[0])
           
            self._process_group(df_T, RossGroups.CONTROL, 'C')
            self._process_group(df_T, RossGroups.WIDE_BREAST, 'WB')
            
            # print(self._process_median_group(df_T, RossGroups.CONTROL, 'C').head(2))
            self._process_median_group(df_T, RossGroups.CONTROL, 'C')
            self._process_median_group(df_T, RossGroups.WIDE_BREAST, 'WB')
            self.plot_average_tables(column_name='5-Keto-D-gluconic acid',key='median')
        else:
            print("No data to preprocess.")
    
    def plot_average_tables(self, column_name=None, key='median'):
        # Extract average tables
        avg_control = self.data_groups[RossGroups.CONTROL]['C-'+key]
        avg_wide_breast = self.data_groups[RossGroups.WIDE_BREAST]['WB-'+key]

        # Ensure both tables have the same index (T1, T2, T3, T4)
        avg_control = avg_control.loc[['T1', 'T2', 'T3', 'T4']]
        avg_wide_breast = avg_wide_breast.loc[['T1', 'T2', 'T3', 'T4']]

        if column_name == None:
            for column_name in avg_control.columns[1:]:  # Skip index column
                plt.figure(figsize=(10, 6))
                plt.plot(avg_control.index, avg_control[column_name], label='Control')
                plt.plot(avg_wide_breast.index, avg_wide_breast[column_name], label='Wide Breast')

                plt.xlabel('Timestamp')
                plt.ylabel(column_name)
                plt.title(f'{key} {column_name} Comparison')
                plt.legend()
                plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(avg_control.index, avg_control[column_name], label='Control')
            plt.plot(avg_wide_breast.index, avg_wide_breast[column_name], label='Wide Breast')

            plt.xlabel('Timestamp')
            plt.ylabel(column_name)
            plt.title(f'{key} {column_name} Comparison')
            plt.legend()
            plt.show()

    def _normalize_columns(self, df, chicken_ref):
        """Helper method to normalize columns that contains the chicken's reference."""
        columns = [col for col in df.columns if chicken_ref in col]
        if columns:
            df[columns] = df[columns].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        return df

    def _process_group(self, df_T, group_name, prefix):
        avg_table = pd.DataFrame()

        for index in range(1, self._CHICKENS_PER_GROUP):
            control_index = prefix + str(index)
            df_T_filtered = df_T[df_T.index.str.contains(control_index, na=False)]
            
            df_T_filtered.index = df_T_filtered.index.str[:2]

            mean_row = df_T_filtered.mean().to_frame().T

            mean_row.index = ['Average']

            df_with_stats = pd.concat([df_T_filtered, mean_row])
            self.data_groups[group_name][control_index] = df_with_stats

            if avg_table.empty:
                avg_table = df_T_filtered
            else:
                avg_table += df_T_filtered

        avg_table /= self._CHICKENS_PER_GROUP
        avg_table.index = df_T_filtered.index

        avg_mean_row = avg_table.mean().to_frame().T
        avg_mean_row.index = ['Overall Average']

        avg_table_with_stats = pd.concat([avg_table, avg_mean_row])

        self.data_groups[group_name][prefix + '-avg'] = avg_table_with_stats
      
    def _process_median_group(self, df_T, group_name, prefix):
        tables = []
        
        for index in range(1, self._CHICKENS_PER_GROUP):
            current_key = prefix + str(index)
            # print(self.data_groups[group_name])
            tables.append(self.data_groups[group_name][current_key])

        if not tables:
            raise ValueError("No tables found for the specified group.")

        median_values = np.median([table.values[:, 1:] for table in tables], axis=0)
        median_table = tables[0].copy()
        median_table.iloc[:, 1:] = median_values

        self.data_groups[group_name][prefix + '-median'] = median_table
        # print(self.data_groups[group_name][prefix + '-median'].head())

        # return median_table


#%% End

# %%
