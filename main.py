#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from MLManager import MLManager

# Save XLSX to CSV
def save_xlsx_to_csv():
    df = pd.read_excel('polar data.xlsx')
    df.to_csv('Experiment-PolarData.csv', index=False)
# save_xlsx_to_csv()

def reset_state():
    # Reset global variables or any stateful elements here
    global ml_manager
    ml_manager = None  # Resetting global variable

def main():
    print("Start Simulation!")
    data = pd.read_csv('Data/Experiment-PolarData.csv')   
    ml_manager = MLManager(data)
    ml_manager.preprocess()
    ml_manager = None
    # manager.train_model()
    # manager.evaluate_model()
    # manager.save_model("path/to/save/model.pkl")

if __name__ == "__main__":
    main()

# %%
# Tags are the final stage of the chest.