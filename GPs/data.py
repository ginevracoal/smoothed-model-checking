import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def squash_df(df):
    new_df = df.copy()
    new_df = new_df.value_counts(subset=['beta','gamma','Result'], sort=False).to_frame(name='Conteggio').reset_index()
    
    pos = new_df[new_df['Result'] == True]
    neg = new_df[new_df['Result'] == False]
    
    X = pd.merge(neg, pos, how='outer', on=['beta','gamma'])
    
    X.rename(columns = {'Conteggio_x':'Count_FALSE', 'Conteggio_y':'Count_TRUE'}, inplace = True)
    
    X= X.drop(columns=['Result_x', 'Result_y'])
    
    X['Count_FALSE'] = X['Count_FALSE'].fillna(0)
    X['Count_TRUE'] = X['Count_TRUE'].fillna(0)

    X['Count_TRUE'] = X['Count_TRUE'].astype(float)
    
    X['Count_ALL'] = X['Count_FALSE'] + X['Count_TRUE']
    
    return X

def squashed_DF(DF_standard):
    DF_squashed = squash_df(DF_standard)
    df_train_squashed, df_test_squashed = split_train_test(DF_squashed)
    return DF_squashed

class MyDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        # Create a dictionary using which wewill remap the values
        D = {False : 0, True : 1}

        # Remap the values of the dataframe
        self.dataframe['Result'] == self.dataframe['Result'].map(D)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        a = torch.tensor(row[['beta','gamma']].to_numpy(dtype=np.float32))
        b = row[['Result']].to_numpy(dtype=np.float32)
        
        return a,b
    
    
def split_train_test(df, frac=.8, random_state=100):
    df_train=df.sample(frac= frac,random_state=random_state) 
    df_test=df.drop(df_train.index) 
    return df_train, df_test


def get_DataLoader(df):
    dataset = MyDataset(df, transform=transforms.ToTensor())
    return DataLoader(dataset, batch_size=10, shuffle=True)