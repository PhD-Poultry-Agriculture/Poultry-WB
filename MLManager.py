import pandas as pd
from scipy import stats
import numpy as np
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind

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
        chickens_list = self.dataManager.get_medians_dataset()
        group_C = chickens_list[0]
        group_WB = chickens_list[1]
        
        columns_count = len(group_C.columns)
        feature_p_values = []  # To store both feature names and their p-values

        for feature_index in range(1, columns_count):
            feature_name = group_C.columns[feature_index]  # Get the feature name
            control_distances = []
            wb_distances = []

            for timestamp in range(1, 4):
                control_distances.append(abs(group_C.iloc[timestamp, feature_index] - group_C.iloc[timestamp + 1, feature_index]))
                wb_distances.append(abs(group_WB.iloc[timestamp, feature_index] - group_WB.iloc[timestamp + 1, feature_index]))
            
            # Perform t-test only if there are enough data points
            if len(control_distances) > 1 and len(wb_distances) > 1:
                t_stat, p_value = stats.ttest_ind(control_distances, wb_distances)
                feature_p_values.append((feature_name, p_value))  # Store feature name and its p-value
            else:
                feature_p_values.append((feature_name, np.nan))  # Append NaN if not enough data points

        # Sort the list by p-value (p-value is the second item in the tuple)
        feature_p_values.sort(key=lambda x: x[1])

        # Extract the sorted p-values for FDR correction
        p_values = [x[1] for x in feature_p_values]
        reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.3, method='fdr_bh')

        # Print the results, preserving the feature names
        print("Feature-wise P-values before correction:", feature_p_values)
        print("Reject the null hypothesis (FDR corrected):", reject)
        print("Corrected p-values:", pvals_corrected)

        # Optionally, you can print or return the features that were significant after correction
        significant_features = [feature_p_values[i][0] for i in range(len(reject)) if not reject[i]]
        print("Significant features after FDR correction:", significant_features)



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
