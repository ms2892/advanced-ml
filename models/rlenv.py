import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class UCIMushroomDataset(nn.Module):
    def __init__(self):
        self.path = '../data/UCI_Mushroom/agaricus-lepiota.data'
        self.dataframe = pd.read_csv(self.path,sep=',',header=None)
        self.dataframe = self.dataframe.apply(LabelEncoder().fit_transform)
        # print(self.dataframe )
        self.data = self.dataframe.iloc[:,:].values
        self.X = self.data[:,1:]
        self.Y = self.data[:,0]
        self.Y = np.reshape(self.Y,(self.Y.shape[0],1))
        self.n = self.data.shape[0]
        # print(self.X.shape,self.Y.shape)

    def __getitem__(self,index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.n

class UCIMushroomEnvironment:
    def __init__(self):
        pass
        
    def get_mushroom(self):
        pass
    
    def get_reward(self):
        pass
    
if __name__=='__main__':
    db = UCIMushroomDataset()
    
