import pandas as pd
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

class RossGroups(Enum):
    CONTROL = 'C'
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
        self.FEATURE_LST = []
        self._preprocess()

    def _preprocess(self):
        if self.data is not None:
            print("Preprocessing data...")
            
            df_z_scaled = self.data.copy()            
            # df_z_scaled = self._normalize(df_z_scaled, 'C')
            # df_z_scaled = self._normalize(df_z_scaled, 'B')

            df_z_scaled = df_z_scaled.drop(['Tags', 'Formula'], axis=1)
            df_T = df_z_scaled.T
            df_T.columns = df_T.iloc[0]
            df_T.index.name = 'Timestamp'
            df_T = df_T.drop(df_T.index[0])
            self._FEATURE_CNT = len(df_T.columns)-1
            self.FEATURE_LST = df_T.columns.tolist()
            self._process_group(df_T, RossGroups.CONTROL)
            self._process_group(df_T, RossGroups.WIDE_BREAST)
            
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

    def _normalize(self, df, chicken_ref):
        """Helper method to normalize columns that contains the chicken's reference."""
        columns = [col for col in df.columns if chicken_ref in col]
        if columns:
            df[columns] = df[columns].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        return df

    def _process_group(self, df_T, group_name):
        for index in range(1, self.CHICKENS_PER_GROUP + 1):
            control_index = group_name.value + str(index)
            
            df_T_filtered = df_T[df_T.index.str.contains(control_index, na=False)]            
            df_T_filtered.index = df_T_filtered.index.str[:2]

            print(f"\nProcessing control index: {control_index}")
            print(f"Filtered DataFrame for {control_index}:\n{df_T_filtered}")
            
            self.data_groups[group_name][control_index] = df_T_filtered


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
        data_stack = np.dstack([chicken.values for chicken in group])
        median_values = np.median(data_stack, axis=2)
        median_table = pd.DataFrame(median_values, index=group[0].index, columns=group[0].columns)
        median_table.iloc[:, 0] = group[0].iloc[:, 0]

        return median_table
    
    def generate_combinations(self, n):
        all_combinations = list(combinations(range(1, n + 1), n // 2))

        valid_combinations = []
        
        for comb in all_combinations:
            group1 = list(comb)
            group2 = [x for x in range(1, n + 1) if x not in group1]
            if group1 < group2:
                valid_combinations.append((group1, group2))
        
        return valid_combinations

    def _create_permutations_distances(self):
        count_agreements_GT = [0] * self._FEATURE_CNT
        chicks_by_index = {}
    
        # Populate chicks_by_index dictionary
        for index in range(1, self.CHICKENS_PER_GROUP + 1):
            current_key_C = RossGroups.CONTROL.value + str(index)
            current_key_WB = RossGroups.WIDE_BREAST.value + str(index)
            
            if current_key_C in self.data_groups[RossGroups.CONTROL]:
                indexon = str(index)
                chicks_by_index[indexon] = self.data_groups[RossGroups.CONTROL][current_key_C]

            if current_key_WB in self.data_groups[RossGroups.WIDE_BREAST]:
                chicks_by_index[str(index + self.CHICKENS_PER_GROUP)] = self.data_groups[RossGroups.WIDE_BREAST][current_key_WB]

        # Debug: Print the chicks_by_index dictionary
        print("Chicks by index:", chicks_by_index)
        
        n = self.CHICKENS_PER_GROUP * 2  
        combinations_list = self.generate_combinations(n)
        all_features_distances = []
        
        print('Total combinations:', len(combinations_list))
        
        for idx, combination in enumerate(combinations_list):
            print(f"Processing combination {idx+1}/{len(combinations_list)}: Group A: {combination[0]}, Group B: {combination[1]}")

            tuple_group_A = combination[0]
            tuple_group_B = combination[1]

            group_A = []
            group_B = []
            
            for value in tuple_group_A:
                try:
                    df = chicks_by_index[str(value)]
                except KeyError:
                    print(f"KeyError: {value} not found in chicks_by_index")
                    raise  # Re-raise the exception after printing

                group_A.append(df)

            for value in tuple_group_B:
                try:
                    df = chicks_by_index[str(value)]
                except KeyError:
                    print(f"KeyError: {value} not found in chicks_by_index")
                    raise  # Re-raise the exception after printing

                group_B.append(df)

            # Compute distance between the two groups
            all_features_distances.append(
                self._distance_between_two_groups(
                    self._process_group_median(group_A),
                    self._process_group_median(group_B)
                )
            )

        # Ensure that the list is not empty before proceeding
        if len(all_features_distances) == 0:
            raise ValueError("No distances were calculated.")

        ground_truth_distances = all_features_distances[0]
        all_features_distances = all_features_distances[1:]

        # Initialize count_agreements_GT
        count_agreements_GT = [0] * self._FEATURE_CNT

        # Process each distance vector and compare with ground truth
        for distance_vector in all_features_distances:
            for feature_index, distance_value in enumerate(distance_vector):
                if distance_value > ground_truth_distances[feature_index]:
                    count_agreements_GT[feature_index] += 1

        # Compute p-values as the proportion of agreements over the total number of permutations
        p_value_per_feature = {}
        for feature_index in range(self._FEATURE_CNT):
            p_value_per_feature[self.FEATURE_LST[feature_index]] = count_agreements_GT[feature_index] / len(all_features_distances)

        return p_value_per_feature



#%% End