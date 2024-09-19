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
        self.data_groups[RossGroups.CONTROL]['C-avg'] = pd.DataFrame()
        self.data_groups[RossGroups.WIDE_BREAST]['WB-avg'] = pd.DataFrame()
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
            
            # print(self.data_groups[RossGroups.CONTROL]['C-avg'].head())
            # print(self.data_groups[RossGroups.WIDE_BREAST]['WB-avg'].head())
            # print(self.data_groups[RossGroups.WIDE_BREAST]['WB2'].tail(6))
            self.plot_average_tables()
        else:
            print("No data to preprocess.")
    
    def plot_average_tables(self):
        # Extract average tables
        avg_control = self.data_groups[RossGroups.CONTROL]['C-avg']
        avg_wide_breast = self.data_groups[RossGroups.WIDE_BREAST]['WB-avg']

        # Ensure both tables have the same index (T1, T2, T3, T4)
        avg_control = avg_control.loc[['T1', 'T2', 'T3', 'T4']]
        avg_wide_breast = avg_wide_breast.loc[['T1', 'T2', 'T3', 'T4']]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(avg_control.index, avg_control.iloc[:, 1:], label='Control')
        plt.plot(avg_wide_breast.index, avg_wide_breast.iloc[:, 1:], label='Wide Breast')

        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Average Tables Comparison')
        plt.legend()
        plt.show()
        
    def plot_averages(self):
        """Plot all averages for 'C' and 'WB' groups."""
        
        # Define plot settings
        plot_title = 'Averages for Control (C) and Wide Breast (WB) Groups'
        x_label = 'Chicken Group Index'
        y_label = 'Average Values'
        figsize = (10, 6)
        key = 'T4'

        # Define groups and their corresponding colors
        groups = {
            RossGroups.CONTROL: {'prefix': 'C', 'color': 'blue', 'label': 'Control Group (C)'},
            RossGroups.WIDE_BREAST: {'prefix': 'WB', 'color': 'red', 'label': 'Wide Breast Group (WB)'}
        }

        # Extract data for plotting
        x_values = []
        y_values = {group: [] for group in groups}

        for index in range(1, self._CHICKENS_PER_GROUP):
            for group, settings in groups.items():
                group_index = settings['prefix'] + str(index)
                if group_index in self.data_groups[group]:
                    avg_row = self.data_groups[group][group_index].loc[key]
                    y_values[group].append(avg_row.iloc[0])
            x_values.append(index)

        # Create the plot
        plt.figure(figsize=figsize)
        for group, settings in groups.items():
            plt.plot(x_values, y_values[group], label=settings['label'], color=settings['color'])

        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title='Groups')
        plt.grid(True)

        # Show the plot
        # plt.show()

    def _normalize_columns(self, df, chicken_ref):
        """Helper method to normalize columns that contains the chicken's reference."""
        columns = [col for col in df.columns if chicken_ref in col]
        if columns:
            df[columns] = df[columns].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        return df

    def _process_group(self, df_T, group_name, prefix):
        """Helper method to process and append average and median rows for a group."""
        avg_table = pd.DataFrame()  # Initialize average table

        for index in range(1, self._CHICKENS_PER_GROUP):
            control_index = prefix + str(index)
            df_T_filtered = df_T[df_T.index.str.contains(control_index, na=False)]
            
            df_T_filtered.index = df_T_filtered.index.str[:2]

            mean_row = df_T_filtered.mean().to_frame().T
            median_row = df_T_filtered.median().to_frame().T

            mean_row.index = ['Average']
            median_row.index = ['Median']

            df_with_stats = pd.concat([df_T_filtered, mean_row, median_row])
            self.data_groups[group_name][control_index] = df_with_stats

            # Accumulate sum for average table
            if avg_table.empty:
                avg_table = df_T_filtered
            else:
                avg_table += df_T_filtered

        # Calculate average table
        avg_table /= self._CHICKENS_PER_GROUP
        avg_table.index = df_T_filtered.index  # Preserve original index

        # Append average and median rows to average table (optional)
        avg_mean_row = avg_table.mean().to_frame().T
        avg_median_row = avg_table.median().to_frame().T

        avg_mean_row.index = ['Overall Average']
        avg_median_row.index = ['Overall Median']

        avg_table_with_stats = pd.concat([avg_table, avg_mean_row, avg_median_row])

        # Store average table
        self.data_groups[group_name][prefix + '-avg'] = avg_table_with_stats
      
#%% End
