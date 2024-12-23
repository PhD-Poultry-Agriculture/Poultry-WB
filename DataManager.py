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
        self._TOTAL_TIMESTAMPS = ['T1', 'T2', 'T3', 'T4']
        self._TIMESTAMPS = ['T3', 'T4']
        self.CHICKENS_PER_GROUP = 8
        self.data = raw_data
        self.data_groups = {
            RossGroups.CONTROL: {},
            RossGroups.WIDE_BREAST : {}
        }
        self.FEATURE_LST = []
        
        # T1, T2, T3, T4
        self.selected_features = ['ADENINE', 'ADENOSINE', 'ASPARTATE', 'Leu-Ala', 'Leu-Gly', 'Pseudouridine', 'GLUTAMATE', 'L-Methionine sulfoxide', 'TREHALOSE', 'PROLINE', 'ALLANTOIN', '4-Hydroxyphenylacetic acid', 'CYSTATHIONINE', 'Dehydroascorbic acid', 'Leucine', 'PYROGLUTAMATE', 'ASPARAGINE', 'CREATINE', 'DIACETYL', 'D-MANNOSAMINE', '2-Isopropylmalic acid', '2-OXOADIPATE', '3-Methyl-L-histidine', 'GLYCEROL', 'SERINE', 'Homocitrulline', 'Saccharic acid 1,4-lactone', 'NORLEUCINE']
        # T1, T2, T3
        self.selected_features = ['Leu-Ala', 'Leu-Gly', 'ADENOSINE', 'Pseudouridine', 'ADENINE', 'ASPARTATE']
        # T2, T3, T4
        self.selected_features = ['2-OXOADIPATE', 'ADENINE', 'GLUTAMATE', 'N-(2-Methylbutyryl)glycine', 'RHAMNOSE', 'Saccharopine', 'Uric acid', 'ASPARTATE', 'ADENOSINE', 'TREHALOSE', 'DIACETYL', 'TRANS-ACONITATE', 'Choline glycerophosphate', 'Leu-Ala', 'GLUTARATE', 'Pseudouridine', 'Pyrogallol', 'GLYCEROL', 'L-Methionine sulfoxide', '5-Keto-D-gluconic acid', 'Ac-Val-OH', '4-Hydroxyphenylacetic acid', 'Ureidosuccinic acid', '3-Methyl-L-histidine', 'ALLANTOIN', 'Homocitrulline', 'QUINATE', 'BETAINE', 'Dehydroascorbic acid', 'Gamma-L-Glutamyl-L-alanine', 'SERINE', 'TAURINE', 'Dehydroacetic acid', 'PYROGLUTAMATE', 'DEOXYCYTIDINE', 'CYSTATHIONINE', 'CREATINE', 'GLYCERATE', 'N-Acetyl-L-methionine', '2-Hydroxyoctanoic acid', '2-Isopropylmalic acid', 'Citraconic acid', 'ERYTHRITOL', 'GLYCINE', 'Leucine', 'Leu-Gly', 'OROTATE', 'PROLINE', 'Saccharic acid 1,4-lactone', 'Sorbose', 'Nalpha-Acetyl-L-ornithine', '5-Methyluridine', 'Gulonic acid gamma-lactone', 'TRIGONELLINE', '4-ACETAMIDOBUTANOATE', 'Alpha-Ketoglutaric acid', 'N-ACETYLGLUTAMATE', 'SUCCINATE', 'GUANOSINE', 'Myo-Inositol', 'Fructose', 'Xylose', 'TRANS-4-HYDROXY-L-PROLINE', 'Mannose', 'N-Acetyl-L-proline', '2-Methyl-4-pentenoic acid', 'TYROSINE', 'N-ACETYLASPARAGINE', '2,5-DIHYDROXYBENZOATE', 'D-MANNOSAMINE', 'GLUTAMINE', 'N-Acetyl-L-tyrosine', 'Nepsilon-Trimethyllysine', 'Alpha-Hydroxyglutaric acid', 'PHENYLALANINE', 'N-FORMYL-L-METHIONINE', 'HYPOXANTHINE', '4-HYDROXYBENZALDEHYDE', 'NORLEUCINE', 'CITRULLINE', 'INDOLE-3-ACETATE', 'N-ACETYLALANINE', 'N-ACETYLPHENYLALANINE', 'p-Coumaric acid']
        # T1, T2
        self.selected_features = ['Leu-Ala', 'Leu-Gly', 'ADENOSINE', 'Pseudouridine', 'ADENINE']
        # T2, T3
        self.selected_features = ['ADENINE', 'ADENOSINE', 'GLUTAMATE', 'RHAMNOSE', 'TREHALOSE', 'Leu-Ala', 'Leu-Gly', 'Saccharopine', 'GLYCEROL', 'N-Acetyl-L-methionine', 'Ac-Val-OH', 'ASPARTATE', 'O-ACETYLSERINE', 'Pseudouridine', 'N-(2-Methylbutyryl)glycine', 'Pyrogallol', 'SUCCINATE', '2-OXOADIPATE', 'DIACETYL', 'GUANOSINE', 'ALLANTOIN', 'Ureidosuccinic acid', 'L-Methionine sulfoxide', 'GLUTARATE', 'SERINE', 'BETAINE', 'CYSTATHIONINE', 'Homocitrulline']
        # T3, T4
        self.selected_features = ['2-OXOADIPATE', '3-Methyl-L-histidine', 'ADENINE', 'ASPARTATE', 'GLUTAMATE', 'GLUTARATE', 'QUINATE', 'RHAMNOSE', 'Pseudouridine', 'ADENOSINE', 'TRANS-ACONITATE', 'TREHALOSE', 'Uric acid', '5-Keto-D-gluconic acid', 'Choline glycerophosphate', 'DIACETYL', 'L-Methionine sulfoxide', 'N-ACETYLPHENYLALANINE', 'Pyrogallol', 'CREATINE', 'PHENYLALANINE', 'N-ACETYLASPARAGINE', '2-Isopropylmalic acid', '4-Hydroxyphenylacetic acid', 'Dehydroacetic acid', 'DIHYDROURACIL', 'Saccharopine', 'Gulonic acid gamma-lactone', 'CYSTATHIONINE', 'Homocitrulline']
        
        self._preprocess()


    def _preprocess(self):
        if self.data is not None:
            df_z_scaled = self.data.drop(['Tags', 'Formula'], axis=1)
            reduced_timestamps = [ts for ts in self._TOTAL_TIMESTAMPS if ts not in self._TIMESTAMPS]
            drop_columns = [col for col in df_z_scaled.columns if any(ts in col for ts in reduced_timestamps)]
            df_z_scaled = df_z_scaled.drop(columns=drop_columns)
            df_T = df_z_scaled.T
            df_T.columns, df_T.index.name = df_T.iloc[0], 'Timestamp'
            df_T = df_T.drop(df_T.index[0])
            print(df_T.T.columns)
            self._FEATURE_CNT, self.FEATURE_LST = len(df_T.columns), df_T.columns.tolist()
            self._process_group(df_T, RossGroups.CONTROL)
            self._process_group(df_T, RossGroups.WIDE_BREAST)
        else:
            print("No data to preprocess.")



    
    def _process_4_random_forest(self):
        results = []
        for index in range(1, self.CHICKENS_PER_GROUP + 1):
            control_index = RossGroups.CONTROL.value + str(index)        
            control_chick = self.data_groups[RossGroups.CONTROL][control_index]
            df_filtered = control_chick[control_chick.columns.intersection(self.selected_features)]
            single_row_chick = df_filtered.values.flatten()
            result_row_with_zero = np.append(single_row_chick, 0)
            results.append(result_row_with_zero)
            print(f"CONTROL chick processed: {control_chick}")  # Debug print

        for index in range(1, self.CHICKENS_PER_GROUP + 1):
            control_index = RossGroups.WIDE_BREAST.value + str(index)        
            control_chick = self.data_groups[RossGroups.WIDE_BREAST][control_index]
            df_filtered = control_chick[control_chick.columns.intersection(self.selected_features)]
            single_row_chick = df_filtered.values.flatten()
            result_row_with_one = np.append(single_row_chick, 1)
            results.append(result_row_with_one)
            print(f"WIDE_BREAST chick processed: {control_chick}")  # Debug print

        final_df = pd.DataFrame(results)
        # print(f"Final DataFrame shape: {final_df.shape}")  # Debug print
        # print(f"Final DataFrame:\n{final_df.head()}")  # Debug print
        # print(f"Final DataFrame:\n{final_df.tail()}")  # Debug print

        return final_df

    def plot_average_tables(self, column_name=None, key='median'):
        avg_control = self.data_groups[RossGroups.CONTROL]['C-'+key]
        avg_wide_breast = self.data_groups[RossGroups.WIDE_BREAST]['WB-'+key]

        avg_control = avg_control.loc[self._TIMESTAMPS]
        avg_wide_breast = avg_wide_breast.loc[self._TIMESTAMPS]

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
   
    def generate_combinations(self, n):
        all_combinations = list(combinations(range(1, n + 1), n // 2))

        valid_combinations = []
        
        for comb in all_combinations:
            group1 = list(comb)
            group2 = [x for x in range(1, n + 1) if x not in group1]
            if group1 < group2:
                valid_combinations.append((group1, group2))
        
        return valid_combinations
    
    def _process_group(self, df_T, group_name):
        for index in range(1, self.CHICKENS_PER_GROUP + 1):
            control_index = group_name.value + str(index)

            df_T_filtered = df_T[df_T.index.str.contains(control_index, na=False)]
            df_T_filtered.index = df_T_filtered.index.str[:2]

            # print(f"\nProcessing control index: {control_index}")
            # print(f"Filtered DataFrame for {control_index}:\n{df_T_filtered}")

            self.data_groups[group_name][control_index] = df_T_filtered

    def _distance_between_two_groups(self, df_group1, df_group2):
        # Calculate the differences between the two dataframes
        differences = df_group1 - df_group2
        
        # Calculate the sum of absolute differences for each feature
        distances_per_feature = abs(differences.sum(axis=0))
        # print('my test')
        # print(df_group1.head())
        # print(df_group2.head())        

        return distances_per_feature.tolist()


    def _process_group_median(self, group):
        # Calculate the median for the group
        data_stack = np.dstack([chicken.values for chicken in group])
        median_values = np.median(data_stack, axis=2)
        median_table = pd.DataFrame(median_values, index=group[0].index, columns=group[0].columns)
        # median_table.iloc[:, 0] = group[0].iloc[:, 0]
        # print(median_table.head())
        return median_table


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
        # print("Chicks by index:", chicks_by_index)

        n = self.CHICKENS_PER_GROUP * 2
        combinations_list = self.generate_combinations(n)
        all_features_distances = []
        print('Total combinations:', len(combinations_list))
        # combinations_list = combinations_list[0:1000]
        for idx, combination in enumerate(combinations_list):
            print(f"Processing combination {idx + 1}/{len(combinations_list)}: Group A: {combination[0]}, Group B: {combination[1]}")

            tuple_group_A = combination[0]
            tuple_group_B = combination[1]            
            group_A = [chicks_by_index[str(value)] for value in tuple_group_A]
            group_B = [chicks_by_index[str(value)] for value in tuple_group_B]
            
            # Compute distance between the two groups using median differences
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
        # all_features_distances = all_features_distances[1:]

        # Initialize count_agreements_GT
        count_agreements_GT = [0] * self._FEATURE_CNT
        # Process each distance vector and compare with ground truth
        for distance_vector in all_features_distances:
            for feature_index, distance_value in enumerate(distance_vector):
                # print('list size: ', count_agreements_GT, 'then', feature_index)
                if distance_value > ground_truth_distances[feature_index]:
                    # print('size: ', len(count_agreements_GT))
                    count_agreements_GT[feature_index] += 1

        # Compute p-values as the proportion of agreements over the total number of permutations
        print('combinations_list')
        print(len(combinations_list))
        
        p_values = []
        for feature_index in range(self._FEATURE_CNT):
            p_values.append(count_agreements_GT[feature_index] / len(combinations_list))
        print('p-values')
        print(len(p_values))
        return p_values



#%% End