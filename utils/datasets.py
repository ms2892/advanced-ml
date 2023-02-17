import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging
import urllib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

class Regression(Dataset):
    def __init__(self,x,y,transform=None):
        super(Regression,self).__init__()
        self.x = x
        self.y = y
        self.n = len(x)
        self.transform=transform
        
    def __getitem__(self, index):
        sample = self.x[index],self.y[index]
        if self.transform:
            x,y = self.transform(sample)
        return x,y
    
    def __len__(self):
        return self.n

class ToTensor:
    # Convert Sample to Tensor
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class Datasets():
    def __init__(self):
        pass
    
    def get_MNIST(self):
        data = torchvision.datasets.MNIST('../data/MNIST',download=True)
        return data
    
    def regression_function(self,x,sigma=0.02):
        y = x + 0.3 * np.sin(2*np.pi*(x+np.random.normal(0,sigma,x.shape))) + 0.3 * np.sin(4*np.pi*(x + np.random.normal(0,sigma,x.shape))) + np.random.normal(0,sigma,x.shape)
        return y    
    
    def get_regression(self,f_range=(-0.2,1.3),train_range=(0,0.5),points=250,sigma=0.02):
        train_pth = 'train.csv'
        test_pth = 'test.csv'
        logging.info('get_regression method called')
        if os.path.isfile(train_pth) and os.path.isfile(test_pth):
            logging.info('Found the Files')
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            train_df = train_df.iloc[:].values
            test_df = test_df.iloc[:].values
            
            train_x = train_df[:,0]
            train_y = train_df[:,1]
            
            test_x = test_df[:,0]
            test_y = test_df[:,1]
        else:
            logging.critical('Datasets not found in the current folder or corrupted files')
            np.random.seed(911)
            test_range_l = (f_range[0],train_range[0])
            test_range_r = (train_range[1],f_range[1])
            train_x = train_range[0] + (train_range[1]-train_range[0])*np.random.rand(points)
            test_x = np.concatenate((test_range_l[0] + (test_range_l[1]-test_range_l[0])*np.random.rand(points), test_range_r[0] + (test_range_r[1]-test_range_r[0])*np.random.rand(points)))
            
            train_y = self.regression_function(train_x)
            test_y = self.regression_function(test_x)
            train_df = {'x':train_x,'y':train_y}
            test_df = {'x':test_x,'y':test_y}
            
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)
            
            train_df.to_csv('train.csv',index=False)
            test_df.to_csv('test.csv',index=False)
            logging.info('Saved train.csv and test.csv files')
        train_x = np.reshape(train_x,(train_x.shape[0],1))
        train_y = np.reshape(train_y,(train_y.shape[0],1))
        test_x = np.reshape(test_x,(test_x.shape[0],1))
        test_y = np.reshape(test_y,(test_y.shape[0],1))
        compose = transforms.Compose([ToTensor()])
        return Regression(train_x,train_y,transform=compose),Regression(test_x,test_y,transform=compose)
            
            
    def download_UCI(self):
        if not os.path.exists('../data/UCI_Mushroom'):
            logging.info('UCI Mushroom Dataset Not Found')
            logging.info('Downloading UCI Mushroom Dataset and saving it in location ../data/UCI_Mushroom')
            os.mkdir('../data/UCI_Mushroom')
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", "../data/UCI_Mushroom/agaricus-lepiota.data")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names", "../data/UCI_Mushroom/agaricus-lepiota.names")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/expanded.Z", "../data/UCI_Mushroom/expanded.Z")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/README", "../data/UCI_Mushroom/README")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/Index", "../data/UCI_Mushroom/Index")
            logging.info('Download Complete')
        else:
            logging.info('Folder for UCI Mushroom found in the data folder. If the files are corrupted please delete the folder at location ../data/UCI_Mushroom')
        
if __name__=='__main__':
    dataset = Datasets()
    mnist = dataset.download_UCI()