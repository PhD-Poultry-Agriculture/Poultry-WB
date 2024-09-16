import pandas as pd


class MLManager:
    
    def __init__(self, data, model=None):
        self.data = data
        self.model = model

    def preprocess(self):
        if self.data is not None:
            print("Preprocessing data...")
            # Normalization
            df_z_scaled = self.data.copy()
            subframe = df_z_scaled.iloc[:, 3:].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
            
            # view normalized data
            # print(subframe.head())
            print("subframe.head()")

            # Reference Markers
        else:
            print("No data to preprocess.")





    def train_model(self):
        """
        Train the machine learning model on the preprocessed data.
        """
        if self.data is not None:
            print("Training the model...")
            # Add training logic here
        else:
            print("No data available for training.")

    def evaluate_model(self):
        """
        Evaluate the machine learning model on the test data.
        """
        if self.model is not None:
            print("Evaluating the model...")
            # Add model evaluation logic here
        else:
            print("No model to evaluate.")

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
