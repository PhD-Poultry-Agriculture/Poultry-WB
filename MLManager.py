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

        sorted_feature_p_values = sorted(feature_p_values.items(), key=lambda x: x[1])
        p_values = [x[1] for x in sorted_feature_p_values]

        methods = [
            'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 
            'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
        ]
        
        # Dictionary to track if a feature is significant across all methods
        feature_agreement = {feature[0]: True for feature in sorted_feature_p_values}

        for method in methods:
            reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.1, method=method)

            with open(f'Data/fdr_results_{method}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Feature', 'Original P-value', 'Corrected P-value', 'Significant'])

                for i in range(len(sorted_feature_p_values)):
                    feature_name = sorted_feature_p_values[i][0]
                    original_p_value = sorted_feature_p_values[i][1]
                    corrected_p_value = pvals_corrected[i]
                    is_significant = reject[i]
                    
                    writer.writerow([feature_name, original_p_value, corrected_p_value, 'Yes' if is_significant else 'No'])

                    # If any method rejects a feature, it won't be significant for all methods
                    if not is_significant:
                        feature_agreement[feature_name] = False

            print(f"Results saved to 'fdr_results_{method}.csv'")

        # Collect features that are significant in all methods
        agreed_significant_features = [feature for feature, agreed in feature_agreement.items() if agreed]

        # Save the features agreed upon by all methods in a final result file
        with open('Data/fdr_results_agreed_features.csv', mode='w', newline='') as final_file:
            writer = csv.writer(final_file)
            writer.writerow(['Feature'])

            for feature in agreed_significant_features:
                writer.writerow([feature])

        print("Features agreed upon by all methods:", agreed_significant_features)
        print("Results saved to 'fdr_results_agreed_features.csv'")

        return agreed_significant_features


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
