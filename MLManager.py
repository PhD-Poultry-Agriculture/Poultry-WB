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
        p_values = []

        for feature_index in range(1, columns_count):
            # Collect all distances for each group across timestamps
            control_distances = []
            wb_distances = []

            for timestamp in range(1, 4):
                control_distances.append(abs(group_C.iloc[timestamp, feature_index] - group_C.iloc[timestamp + 1, feature_index]))
                wb_distances.append(abs(group_WB.iloc[timestamp, feature_index] - group_WB.iloc[timestamp + 1, feature_index]))
            
            # Check if the distances have more than one value
            if len(control_distances) > 1 and len(wb_distances) > 1:
                t_stat, p_value = stats.ttest_ind(control_distances, wb_distances)
                p_values.append(p_value)
            else:
                p_values.append(np.nan)  # Append NaN if not enough data points
        p_values.sort()
        reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.1, method='fdr_bh')

        print("P-values before correction:", p_values)
        print("Reject the null hypothesis (FDR corrected):", reject)
        print("Corrected p-values:", pvals_corrected)


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
