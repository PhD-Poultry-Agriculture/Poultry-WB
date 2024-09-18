#%% Imports
import pandas as pd
import time
from MLManager import MLManager
from DataManager import DataManager

# Save XLSX to CSV
def save_xlsx_to_csv():
    df = pd.read_excel('polar data.xlsx')
    df.to_csv('Experiment-PolarData.csv', index=False)
# save_xlsx_to_csv()

def main():
    print("Start Simulation!")
    raw_data = pd.read_csv('Data/Experiment-PolarData.csv')
    data_manager = DataManager(raw_data)
    ml_manager = MLManager(data_manager)
    
if __name__ == "__main__":
    main()

# %%
# Tags are "how good is the classification of the molecule - A:Best, D:Worse".
# -> Those tags may be used as a scalar per read.