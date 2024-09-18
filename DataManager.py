import pandas as pd
from enum import Enum

class RossGroups(Enum):
    CONTROL = 'Control'
    WIDE_BREAST = 'WB'

class DataManager:
    
    def __init__(self, raw_data):
        self.data = raw_data
        self.data_groups = {
            RossGroups.CONTROL: None,
            RossGroups.WIDE_BREAST : None
        }
        self._preprocess()
        # self._organize_features()

    def _preprocess(self):
        if self.data is not None:
            print("Preprocessing data...")
            # Normalization
            df_z_scaled = self.data.copy()
            df_z_scaled.iloc[:, 3:] = df_z_scaled.iloc[:, 3:].apply(lambda row: (row - row.mean()) / row.std(), axis=1)
            df_z_scaled = df_z_scaled.drop(['Tags', 'Formula'], axis=1)
            
            # Filter by groups
            df_T = df_z_scaled.T
            df_T.columns = df_T.iloc[0]
            df_T.index.name = 'Timestamp'

            df_T = df_T.drop(df_T.index[0])
            df_T_filtered = df_T[df_T.index.str.contains('C2', na=False)]
            
    
            print(df_T_filtered.head())
            # print(df_T.iloc[:,0:2].head())
            # print(df_T.head())
            # df_T.to_csv('output.csv', index=True)
        else:
            print("No data to preprocess.")

    def _organize_features(self):
        features = self.data['Name'].T
        print(features.head())

#%% End
