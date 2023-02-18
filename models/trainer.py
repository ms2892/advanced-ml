import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
import math
import sys
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class TrainModel:

    def error_print(self):
        print("Please follow the given format of passing the argument")
        print("Please pass the argument as such a format")
        print('''
              args = {
                  \'model\'         : model (object),
                  \'dataset\'       : training dataset (object),
                  \'criterion\'     : loss function (object),
                  \'batch_size\'    : batch size (int),
                  \'optimizer\'     : optimizer (object),
                  \'scheduler\'     : scheduler (object) (optional),
                  \'num_epochs\'    : number of epochs (int),   
                  \'val_size\'      : number of validation data points (int)
              }
              ''')    
    
    def __init__(self,**kwargs):
        if 'model' in kwargs:
            self.model = kwargs['model']
        else:
            logging.error("Model Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Model Object not Found")
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
        else:
            logging.error("Training Dataset Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Training Dataset Object not Found")
        if 'criterion' in kwargs:
            self.criterion = kwargs['criterion']
        else:
            logging.error("Criterion / Loss Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Criterion / Loss Object not Found")
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            logging.error("Batch Size not found amongst the Arguments")
            self.error_print()
            raise Exception("Batch Size not Found")
        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']
        else:
            logging.error("Optimizer Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Optimizer Object not Found")
        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
        else:
            logging.warning("Scheduler Object not found amongst the Arguments. Ignore warning if scheduler wasn't meant to be in the loop")
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            logging.error("Number of epochs not found amongst the Arguments")
            self.error_print()
            raise Exception("number of epochs not Found")
        if 'val_size' in kwargs:
            self.val_size = kwargs['val_size']
            if self.val_size>len(self.dataset):
                raise Exception("Please give validation size less than training size")
        else:
            logging.error("Validation size not found amongst the Arguments")
            self.error_print()
            raise Exception("Validation size not Found")
        
        
        
        
if __name__=='__main__':
    args={
        'model': 123,
        'dataset':[123],
        'criterion':123,
        'batch_size':123,
        'optimizer':123,
        'num_epochs':123,
        'val_size':0
    }
    trainer = TrainModel(**args)
    
    # trainer.error_print()