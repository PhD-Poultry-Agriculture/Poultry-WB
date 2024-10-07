import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import csv
from DataManager import *
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics
import numpy.random
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.model_selection import KFold
import math
from sklearn.model_selection import LeaveOneOut

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    def evaluate_random_forest_model(self):
        processed_df = self.dataManager._process_4_random_forest()
        rand_state = 2
        X = processed_df.iloc[:, :-1].values
        Y = processed_df.iloc[:, -1].values

        cvout = LeaveOneOut()

        y_true, y_pred = list(), list()
        for train_ix, test_ix in cvout.split(X):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = Y[train_ix], Y[test_ix]

            classifier = RandomForestClassifier(random_state=rand_state)
            classifier.fit(X_train, y_train)
            yhat = classifier.predict(X_test)

            y_true.append(y_test[0])
            y_pred.append(yhat[0])

        acc = accuracy_score(y_true, y_pred)
        print('Accuracy: %.3f' % acc)
        print('True labels:', y_true)
        print('Predictions:', y_pred)

        cnfmtrx = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:\n', cnfmtrx)

        cm = confusion_matrix(y_true, y_pred, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
        disp.plot()
        plt.show()

        permlabels = np.random.permutation(Y)
        print('Permuted labels:', permlabels)

        X_perm = np.hstack((X, permlabels.reshape(-1, 1)))

        y1_true, y1_pred = list(), list()
        for train_ix, test_ix in cvout.split(X_perm):
            X1_train, X1_test = X_perm[train_ix, :], X_perm[test_ix, :]
            y1_train, y1_test = permlabels[train_ix], permlabels[test_ix]

            classifier = RandomForestClassifier(random_state=rand_state)
            classifier.fit(X1_train, y1_train)
            y1hat = classifier.predict(X1_test)

            y1_true.append(y1_test[0])
            y1_pred.append(y1hat[0])

        acc1 = accuracy_score(y1_true, y1_pred)
        print('Accuracy with permuted labels: %.3f' % acc1)
        print('True labels (permuted):', y1_true)
        print('Predictions (permuted):', y1_pred)

        cnfmtrx_perm = confusion_matrix(y1_true, y1_pred)
        print('Confusion Matrix (Permuted):\n', cnfmtrx_perm)

        cm_perm = confusion_matrix(y1_true, y1_pred, labels=classifier.classes_)
        disp_perm = ConfusionMatrixDisplay(confusion_matrix=cm_perm, display_labels=classifier.classes_)
        disp_perm.plot()
        plt.show()





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
