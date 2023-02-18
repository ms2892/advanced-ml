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

# Imports
#==================================#

# Logging Configuration - Defines the format and severity of the logs 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("datasets_debug.log"),
        logging.StreamHandler()
    ]
)

class Regression(Dataset):
    """
        Class:
            This class helps in creating a Dataset object for the user.
            The user can eventually use this class to forward to a DataLoader for training
            
        Attributes:
            self.x          (np.array): This contains the input value of the regression data
            self.y          (np.array): This contains the output value of the regression data
            self.n          (int)     : This tracks the length of the dataset
            self.transform  (Object)  : This object contains all the transforms that need to be applied
                                        to the data points
    """
    def __init__(self,x,y,transform=None):
        super(Regression,self).__init__()
        
        # Input Values
        self.x = x
        
        # Output Values
        self.y = y
        
        # Length of the Dataset
        self.n = len(x)
        
        # Transforms on the dataset
        self.transform=transform
        
    def __getitem__(self, index):
        """
            Method:
                This method corresponds to overriding the functionality of the class when it's index
                Meaning return value when the class is called in such a manner -> class_name[index]
                
            Args:
                index (int): The index corresponding to the datapoint needed
            
            Output:
                (tuple):    Returns tensor objects in a tuple that corresponds to (x,y) 
        """    
        
        # Extract the samples
        sample = self.x[index],self.y[index]
        
        # Apply the transforms
        if self.transform:
            x,y = self.transform(sample)
            
        # Return the transforms
        return x,y
    
    def __len__(self):
        """
            Method:
                This method overrides the functionality when len() function is applied on the class
            Args:
                None
            Output:
                (int):  Describes the length of the dataset
        """
        
        # Return the length of the dataset
        return self.n



class ToTensor:
    # Transformation to convert sample to tensor
    def __call__(self, sample):
        
        # Divide the sample into input and targets
        inputs, targets = sample
        
        # Return tensor objects for the inputs and targets
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class Datasets():
    """
        Class:
            This class holds the collection of all the datasets that will be needed for the study
            
        Attributes:
            None
        
        Methods:
            get_MNIST() -> Dataset object:  
                This method downloads the MNIST dataset in the ../data/MNIST folder if the dataset is not present.
                Also returns the dataset in a PyTorch Dataset object which can be forwarded to a DataLoader as per the 
                study's specification
                
            regression_fucntion() -> np.array:
                This method takes in input data points and returns the output values points with a Gaussian Noise (0,0.02)
                
            get_regression() -> Train Dataset, Test Dataset:
                This method returns a train and test dataset object for conducting the regression study. These dataset objects
                can be forwarded to a DataLoader as per a person's requirement
            
            download_UCI() -> None:
                (Under Development)
                This method downloads the datasets for the reinforcement learning study (UCI_Mushroom Dataset). 
                Note: This method doesn't return any Dataset object due to lack of understanding as to how the structure of the data should be
                      With a better understanding of the study, this function can be enhanced to return a Dataset object as well.
    """
    def __init__(self):
        pass
    
    def get_MNIST(self):
        """
            Method:
                Returns 2 dataset objects (training and testing) with MNIST dataset in it.
            Args:
                None
            Output:
                (Dataset Object):   Returns the dataset object containing the training MNIST dataset
                (Dataset Object):   Returns the dataset object containing the testing MNIST dataset
        """
        
        # Generate the dataset object for MNIST
        train_data = torchvision.datasets.MNIST('data/MNIST',transform=transforms.ToTensor(),download=True,train=True)
        test_data = torchvision.datasets.MNIST('data/MNIST',train=False,transform=transforms.ToTensor())
        
        # Return the dataset object
        return train_data,test_data
    
    def regression_function(self,x,sigma=0.02):
        """
            Method:
                Returns the output values of the generating function with Gaussian Noise applied
            Args:
                x       (np.array): Array of Input Variables
                sigma   (float)   : Variance of Gaussian Noise to be applied
            Output:
                (np.array)  : Numpy Array containing output values for each input values
        """
        
        # Calculate the outputs with Gaussian Noise applied
        y = x + 0.3 * np.sin(2*np.pi*(x+np.random.normal(0,sigma,x.shape))) + 0.3 * np.sin(4*np.pi*(x + np.random.normal(0,sigma,x.shape))) + np.random.normal(0,sigma,x.shape)
        
        # Return Output Values
        return y    
    
    def get_regression(self,f_range=(-0.2,1.3),train_range=(0,0.5),points=250,sigma=0.02):
        """
            Method:
                This method will return the dataset object containing the dataset for regression
                
            Args:
                f_range     (tuple): This defines the entire range of values that need to be studied
                train_range (tuple): This defines the range that'll be used for training. The range outside of this range will be used for testing
                points      (int)  : Number of points in the training set
                sigma       (float): Variance of the Gaussian Noise to be added
                
            Output:
                (Dataset Object) : The first unpacked variable will correspond to the train dataset object
                (Dataset Object) : The second unpacked variable will correspond to the test dataset object
        """
        
        train_pth = 'utils/train.csv'
        test_pth = 'utils/test.csv'
        logging.info('get_regression method called')
        
        # Check if the dataset is in the utils path
        if os.path.isfile(train_pth) and os.path.isfile(test_pth):
            logging.info('Found the Files')
            
            # If found then load the files
            train_df = pd.read_csv(train_pth)
            test_df = pd.read_csv(test_pth)
            train_df = train_df.iloc[:].values
            test_df = test_df.iloc[:].values
            
            train_x = train_df[:,0]
            train_y = train_df[:,1]
            
            test_x = test_df[:,0]
            test_y = test_df[:,1]
        else:
            logging.critical('Datasets not found in the current folder or corrupted files')
            
            # If not found then set a seed and generate the dataset from scratch.
            np.random.seed(911)
            
            # Define the Test Range
            test_range_l = (f_range[0],train_range[0])
            test_range_r = (train_range[1],f_range[1])
            
            # Define the training input values
            train_x = train_range[0] + (train_range[1]-train_range[0])*np.random.rand(points)
            
            # Define the testing input values
            test_x = np.concatenate((test_range_l[0] + (test_range_l[1]-test_range_l[0])*np.random.rand(points), test_range_r[0] + (test_range_r[1]-test_range_r[0])*np.random.rand(points)))
            
            # Get the output values for training and testing input values
            train_y = self.regression_function(train_x)
            test_y = self.regression_function(test_x)
            
            # Convert to a Pandas DataFrame Object
            train_df = {'x':train_x,'y':train_y}
            test_df = {'x':test_x,'y':test_y}
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)
            
            # Save it in a CSV File
            train_df.to_csv(train_pth,index=False)
            test_df.to_csv(test_pth,index=False)
            logging.info('Saved train.csv and test.csv files')
        
        # Reshape to match the shape -> (Batch * Feature)
        train_x = np.reshape(train_x,(train_x.shape[0],1))
        train_y = np.reshape(train_y,(train_y.shape[0],1))
        test_x = np.reshape(test_x,(test_x.shape[0],1))
        test_y = np.reshape(test_y,(test_y.shape[0],1))

        # Transforms that need to be performed on the dataset
        compose = transforms.Compose([ToTensor()])

        # Return the Dataset objects for the training and testing dataset
        return Regression(train_x,train_y,transform=compose),Regression(test_x,test_y,transform=compose)
            
            
    def download_UCI(self):
        """
            Method:
                This method downloads the UCI Mushroom dataset in the ../data/UCI_Mushroom folder
            Args:
                None
            Output:
                None
        """
        
        # Check if the dataset has been downloaded
        if not os.path.exists('data/UCI_Mushroom'):
            logging.info('UCI Mushroom Dataset Not Found')
            logging.info('Downloading UCI Mushroom Dataset and saving it in location ../data/UCI_Mushroom')
            
            # If not found then make the directory 
            os.mkdir('data/UCI_Mushroom')
            
            # Download the dataset
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", "data/UCI_Mushroom/agaricus-lepiota.data")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names", "data/UCI_Mushroom/agaricus-lepiota.names")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/expanded.Z", "data/UCI_Mushroom/expanded.Z")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/README", "data/UCI_Mushroom/README")
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/Index", "data/UCI_Mushroom/Index")
            logging.info('Download Complete')
        else:
            
            # Found the folder.
            logging.info('Folder for UCI Mushroom found in the data folder. If the files are corrupted please delete the folder at location ../data/UCI_Mushroom and re-run this command')
 
 
# This was meant for Debugging        
if __name__=='__main__':
    dataset = Datasets()
    mnist = dataset.download_UCI()