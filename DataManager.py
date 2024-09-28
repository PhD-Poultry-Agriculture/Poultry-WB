import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations

class RossGroups(Enum):
    CONTROL = 'Control'
    WIDE_BREAST = 'WB'

class DataManager:
    
    def __init__(self, raw_data):
        self.CHICKENS_PER_GROUP = 8
        self.TIMESTAMPS = 4
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
            
            self._process_median_group(df_T, RossGroups.CONTROL, 'C')
            self._process_median_group(df_T, RossGroups.WIDE_BREAST, 'WB')
            # self.plot_average_tables(column_name='GLUTAMATE',key='median')
        else:
            print("No data to preprocess.")
    
    def plot_average_tables(self, column_name=None, key='median'):
        avg_control = self.data_groups[RossGroups.CONTROL]['C-'+key]
        avg_wide_breast = self.data_groups[RossGroups.WIDE_BREAST]['WB-'+key]

        avg_control = avg_control.loc[['T1', 'T2', 'T3', 'T4']]
        avg_wide_breast = avg_wide_breast.loc[['T1', 'T2', 'T3', 'T4']]

        if column_name == None:
            for column_name in avg_control.columns[1:]:
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

        for index in range(1, self.CHICKENS_PER_GROUP+1):
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

        avg_table /= self.CHICKENS_PER_GROUP
        avg_table.index = df_T_filtered.index

        avg_mean_row = avg_table.mean().to_frame().T
        avg_mean_row.index = ['Overall Average']

        avg_table_with_stats = pd.concat([avg_table, avg_mean_row])

        self.data_groups[group_name][prefix + '-avg'] = avg_table_with_stats
      
    def _process_median_group(self, df_T, group_name, prefix):
        tables = []
        
        for index in range(1, self.CHICKENS_PER_GROUP+1):
            current_key = prefix + str(index)
            # print(self.data_groups[group_name])
            tables.append(self.data_groups[group_name][current_key])

        if not tables:
            raise ValueError("No tables found for the specified group.")

        median_values = np.median([table.values[:, 1:] for table in tables], axis=0)
        median_table = tables[0].copy()
        median_table.iloc[:, 1:] = median_values

        self.data_groups[group_name][prefix + '-median'] = median_table

    def get_medians_dataset(self):
        C_median = self.data_groups[RossGroups.CONTROL]['C-median']
        BW_median = self.data_groups[RossGroups.WIDE_BREAST]['WB-median']
        return [C_median, BW_median]

    def _distance_between_two_groups(self, df_group1, df_group2):
        features_count = len(df_group1.columns)
        distances_per_feature = []
        for feature in range(1, features_count):
            total_distance_per_feature = 0
            for index_Ti in range(1, self.TIMESTAMPS):
                total_distance_per_feature += abs(df_group1.iloc[index_Ti, feature] - df_group2.iloc[index_Ti, feature])
            distances_per_feature.append(total_distance_per_feature)
        return distances_per_feature

    def _process_group_median(self, group):
        median_values = np.median([chicken.values[:, 1:] for chicken in group], axis=0)
        median_table = group[0].copy()
        median_table.iloc[:, 1:] = median_values

        return median_table

    
    def generate_combinations(self, n):
        all_combinations = list(combinations(range(1, n+1), n//2))
        
        valid_combinations = []
        
        for comb in all_combinations:
            group1 = list(comb)
            group2 = [x for x in range(1, n+1) if x not in group1]            
            valid_combinations.append((group1, group2))
        
        return valid_combinations

    def _create_permutations_distances(self):
        groud_truth_distances = self._distance_between_two_groups(self.data_groups[RossGroups.CONTROL]['C-median'], self.data_groups[RossGroups.WIDE_BREAST]['WB-median'])
        features_count = len(self.data_groups[RossGroups.CONTROL]['C-median'].columns)-1
        count_agreements_GT = [0]*features_count
        chicks_by_index = {}
       
        for index in range(1, self.CHICKENS_PER_GROUP+1):  # Gather original two groups into dictionary
            current_key_C = 'C' + str(index)
            current_key_WB = 'WB' + str(index)
            
            if current_key_C in self.data_groups[RossGroups.CONTROL]:
                indexon = str(index)
                chicks_by_index[indexon] = self.data_groups[RossGroups.CONTROL][current_key_C]

            if current_key_WB in self.data_groups[RossGroups.WIDE_BREAST]:
                chicks_by_index[str(index+self.CHICKENS_PER_GROUP)] = self.data_groups[RossGroups.WIDE_BREAST][current_key_WB]
      
        n = self.CHICKENS_PER_GROUP * 2  
        combinations_list = self.generate_combinations(n)
        all_features_distances = []
        
        print('Total combinations:', len(combinations_list))
        for idx, combination in enumerate(combinations_list):
            print(f"Processing combination {idx+1}/{len(combinations_list)}: Group A: {combination[0]}, Group B: {combination[1]}")  # Print current combination

            tuple_group_A = combination[0]
            tuple_group_B = combination[1]

            group_A = []
            group_B = []
      
            for value in tuple_group_A:
                df = chicks_by_index[str(value)]
                group_A.append(df)

            for value in tuple_group_B:
                df = chicks_by_index[str(value)]
                group_B.append(df)

            all_features_distances.append(
                self._distance_between_two_groups(
                    self._process_group_median(group_A),
                    self._process_group_median(group_B)
                )
            )

        all_features_distances = all_features_distances[1:]

        for distance_vector in all_features_distances:
            feature_index = 0
            for distance_value in distance_vector:                
                if distance_value > groud_truth_distances[feature_index]:
                    count_agreements_GT[feature_index] += 1
                feature_index += 1
        
        # 3. Compute p-values regarding source-of-truth, 
        p_value_per_feature = {}
        columns = self.data_groups[RossGroups.CONTROL]['C-median'].columns.tolist()
        p_values = [agreements / features_count for agreements in count_agreements_GT]
        for index in range(0, features_count):
            p_value_per_feature[columns[index]] = p_values[index]
        return p_value_per_feature
#%% End