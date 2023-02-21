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
from collections import defaultdict
import randomname
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

logger = logging.getLogger(__name__)

class TrainModelWrapper:

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
                  \'es_flag\'       : Boolean value to decide whether to use early stopping or not. (Default = False)
                  \'num_epochs\'    : number of epochs (int),   
                  \'val_size\'      : number of validation data points (int)
                  \'mode\': Boolean value to decide whether this training is a classification training or not. (Default = 0) 
                                      Possible Values [0,1,2] -> 
                                        0 - Regression
                                        1 - Binary Classification
                                        2 - MultiClass Classification
              }
              ''')    
    
    def __init__(self,**kwargs):
        if 'model' in kwargs:
            self.model = kwargs['model']
            self.model_name = randomname.get_name()
        else:
            logging.error("Model Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Model Object not Found")
        
        
        if 'val_size' in kwargs:
            self.val_size = kwargs['val_size']

        else:
            logging.error("Validation size not found amongst the Arguments")
            self.error_print()
            raise Exception("Validation size not Found")
        
        
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
            if self.val_size>len(self.dataset):
                raise Exception("Please give validation size less than training size")
            self.train_size = len(self.dataset)-self.val_size
            self.train_dataset,self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])
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
            self.scheduler=None
            logging.warning("Scheduler Object not found amongst the Arguments. Ignore warning if scheduler wasn't meant to be in the loop")
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            logging.error("Number of epochs not found amongst the Arguments")
            self.error_print()
            raise Exception("number of epochs not Found")
        if 'es_flag' in kwargs:
            self.es_flag = kwargs['es_flag']
        else:
            self.es_flag = False
        
        if 'mode' in kwargs:
            self.c_flag=kwargs['mode']
        else:
            self.c_flag=0
        self.writer = SummaryWriter('runs/'+self.model_name)
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True)

    def train(self):
        start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        history = defaultdict(list)
        best_loss = np.inf
        scaler = amp.GradScaler()
        n_accumulate = 4
        dataloaders = {'train':self.train_loader,'val':self.val_loader}
        dataset_sizes = {'train':len(self.train_dataset),'val':len(self.val_dataset)}
        print("The tensorboard model name corresponding to this model is", self.model_name)
        if self.es_flag:
            earlystop = self.EarlyStop(mode = self.c_flag)
        else:
            earlystop = None
        for step, epoch in enumerate(range(1,self.num_epochs+1)):
            print('Epoch {}/{}'.format(epoch,self.num_epochs))
            print('-'*10)
            
            for phase in ['train','val']:
                if (phase=='train'):
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corr = 0
                
                for inputs,label in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    label = label.to(device)
                    
                    # Forward Pass
                    
                    with torch.set_grad_enabled(phase=='train'):
                        with amp.autocast(enabled=True):
                            output = self.model(inputs)
                            loss = self.criterion(output,label)
                            loss = loss/n_accumulate
                            
                        if phase=='train':
                            scaler.scale(loss).backward()
                            
                        if phase=='train' and (step+1)%n_accumulate==0:
                            scaler.step(self.optimizer)
                            scaler.update()
                            if self.scheduler:
                                self.scheduler.step()
                            self.optimizer.zero_grad()
                    if self.c_flag ==1:
                        running_corr += self.binary_accuracy(output,label)
                    elif self.c_flag ==2:
                        running_corr += self.multi_accuracy(output,label)    
                    running_loss +=loss.item()*inputs.size(0)
                epoch_loss = running_loss/dataset_sizes[phase]
                if self.c_flag!=0:
                    epoch_acc = running_corr/dataset_sizes[phase]
                self.writer.add_scalar(phase+'_loss',epoch_loss,epoch)
                if self.c_flag!=0:
                    self.writer.add_scalar(phase+'_acc',epoch_acc,epoch)
                history[phase + ' loss'].append(epoch_loss)
                print('{} Loss: {:.4f}'.format(phase,epoch_loss))
                if self.c_flag!=0:
                    history[phase+' acc'].append(epoch_acc)
                    print('{} Acc: {:.4f}'.format(phase,epoch_acc))
                if self.es_flag and phase=='val':
                    if self.c_flag==0:
                        if earlystop.early_stop(epoch_loss):
                            break
                    else:
                        if earlystop.early_stop(epoch_acc):
                            break
            print("")
                
        end = time.time()
        time_elapsed = end - start
        print('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed //3600, (time_elapsed%3600)//60,(time_elapsed%3600)%60))
        # print("Best Loss", best_loss)
        return self.model, history
    
    def binary_accuracy(self,pred,label):
        y_pred = pred.round()
        return y_pred.eq(label).sum()
    
    def multi_accuracy(self,pred,label):
        _,preds = torch.max(pred,1)
        correct_preds = preds.eq(label).sum()
        return correct_preds
    
    class EarlyStop:
        def __init__(self,patience=5,min_delta=0.001,mode=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_metric=np.inf
            self.mode = mode
            self.q = []
            
        def early_stop(self,validation_metric):
            if len(self.q)!=self.patience:
                self.q.append(validation_metric)
            else:
                avg_metric = sum(self.q)/self.patience
                if abs(validation_metric-avg_metric)<self.min_delta:
                    return True
            return False
                    
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
    trainer = TrainModelWrapper(**args)
    
    # trainer.error_print()