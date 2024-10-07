import pandas as pd
from scipy import stats
import numpy as np
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import csv
from DataManager import *


class MLManager:
    
    def __init__(self, dataManager):
        self.dataManager = dataManager
        self.model = None
        
    def train_model(self):
        """
        Train the machine learning model on the preprocessed data.
        """
        if self.data is not None:
            print("Training the model...")
            # Add training logic here
        else:
            print("No data available for training.")

    def evaluate_fdr(self):
        feature_p_values = self.dataManager._create_permutations_distances()
        
        print('feature_p_values')
        print(feature_p_values)
        # methods = [
        #     'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 
        #     'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
        # ]
        methods = ['fdr_bh']
        
        for method in methods:
            fdr_results = smm.multipletests(feature_p_values, alpha=0.1, method=method, is_sorted=False, returnsorted=False)
            
            mrks = [f"{self.dataManager.FEATURE_LST[index]}|{index}|Passed: {passed}" for index, passed in enumerate(fdr_results[0])]
            print("Final Markers with Status:")
            print("\n".join(mrks))

            sorted_results = sorted(
                [(self.dataManager.FEATURE_LST[index], feature_p_values[index], fdr_results[1][index], fdr_results[0][index]) 
                for index in range(len(feature_p_values))],
                key=lambda x: x[2]  # Sort by corrected p-value
            )

            with open(f'Data/fdr_results_{method}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Feature', 'Original P-value', 'Corrected P-value', 'Significant'])
                
                for feature_name, original_p_value, corrected_p_value, is_significant in sorted_results:
                    writer.writerow([feature_name, original_p_value, corrected_p_value, 'Yes' if is_significant else 'No'])

            print(f"Results saved to 'fdr_results_{method}.csv'")

            significant_features = [feature_name for feature_name, original_p_value, corrected_p_value, is_significant 
                                    in sorted_results if is_significant]

            with open(f'Data/fdr_results_significant_features_{method}.csv', mode='w', newline='') as final_file:
                writer = csv.writer(final_file)
                writer.writerow(['Feature'])
                for feature in significant_features:
                    writer.writerow([feature])

            print(f"Significant features for {method}:", significant_features)
            print(f"Results saved to 'fdr_results_significant_features_{method}.csv'")

        return significant_features

    def evaluate_model(self):
        chickens_list = self.dataManager.get_medians_dataset()
        group_C = chickens_list[0]
        group_WB = chickens_list[1]
        max_distances = {}
        columns_count = len(group_C.columns)-1
        
        for cell_index in range(1, columns_count):
            for timestamp in range(1, 4):
                distance_Ti = 0
                distance_Ti += abs(group_C.iloc[timestamp, cell_index] - group_WB.iloc[timestamp, cell_index])
            current_column_key = group_C.columns[cell_index]
            max_distances[current_column_key] = distance_Ti/4
        
        for key in sorted(max_distances, key=max_distances.get, reverse=True):
            print(str(key) + ' : ' + str(max_distances[key]))

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        
        :param file_path: The path to save the model.
        """
        if self.model is not None:
            print(f"Saving model to {file_path}...")
            # Add logic to save the model
        else:
            print("No model to save.")

    def load_model(self, file_path):
        """
        Load a pre-trained model from a file.
        
        :param file_path: The path to the model file.
        """
        print(f"Loading model from {file_path}...")
        # Add logic to load the model
        self.model = "Loaded Model"  # Placeholder for actual model loading
