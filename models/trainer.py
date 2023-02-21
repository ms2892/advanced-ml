import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import defaultdict
import randomname
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

# Imports
#===========================#


# Logging Configuration - Defines the format and the severity of the logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trainer_debug.log",'w'),
        logging.StreamHandler()
    ]
)
class TrainModelWrapper:
    
    """
        Class:
            This class helps in defining a general trainer for regression an classification
            
        Attributes:
            This class takes in a dictionary as an attribute. The format of the dictionary is as follows
            
            args = {
                'model'         : model (object),
                'dataset'       : training dataset (object),
                'criterion'     : loss function (object),
                'batch_size'    : batch size (int),
                'optimizer'     : optimizer (object),
                'scheduler'     : scheduler (object) (optional),
                'es_flag'       : Boolean value to decide whether to use early stopping or not. (Default = False)
                'num_epochs'    : number of epochs (int),   
                'val_size'      : number of validation data points (int)
                'mode'          : Boolean value to decide whether this training is a classification training or not. (Default = 0) 
                                      Possible Values [0,1,2] -> 
                                        0 - Regression
                                        1 - Binary Classification
                                        2 - MultiClass Classification
              }
              
            To pass these attributes to the TrainModelWrapper please pass it using the following syntax
            
            trainer = TrainModelWrapper(**args)
            
        Methods:
            train() -> Model object, history (dictionary):
                This method returns the trained model and the history of the results in a dictionary. 
                Depending on the mode of training, the history will contain it's respective metrics.
                
            error_print() -> None:
                This method is to notify the user if there is a discrepancy while passing the configuration.
                
            binary_correct() -> float:
                This method returns the correct number of predictions with respect to the ground truth in a binary classification setting
                
            multi_correct() -> float:
                This method returns the correct number of predictions with respect to the ground truth in a multi class classification setting
                
        Sub Class:
            EarlyStop:
                This class helps in enabling the functionality of early stopping in training 
                    
    """

    def error_print(self):
        '''
            Method:
                This method prompts the user to follow the configuration format if there is any data missing.
                
            Args:
                None
            
            Output:
                None
        '''
        
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
                  \'val_dataset\'   : Validation Dataset to be passed (object)
                  \'mode\': Boolean value to decide whether this training is a classification training or not. (Default = 0) 
                                      Possible Values [0,1,2] -> 
                                        0 - Regression
                                        1 - Binary Classification
                                        2 - MultiClass Classification
              }
              ''')    
    
    def __init__(self,**kwargs):
        '''
            Constructor:
                This method helps in initializing the attributes of the class.
                
            Args:
                The dictionary format as specified in the error_print() method
                
            Output:
                None
        '''
        
        # Check if model is present in the configuration
        if 'model' in kwargs:
            self.model = kwargs['model']
            
            # Assign a random name for tensorboard
            self.model_name = randomname.get_name()
        else:
            
            # Raise Error if model is missing
            logging.error("Model Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Model Object not Found")
        
        # Check if validation dataset is present in the configuration
        if 'val_dataset' in kwargs:
            self.val_dataset = kwargs['val_dataset']
        else:
            # Raise Error if val_dataset is missing
            logging.error("Validation dataset not found amongst the Arguments")
            self.error_print()
            raise Exception("Validation dataset not Found")
        
        # Check if dataset is present in the configuration
        if 'dataset' in kwargs:
            self.train_dataset = kwargs['dataset']
        else:
            
            # Raise an error if Dataset is not found 
            logging.error("Training Dataset Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Training Dataset Object not Found")
        
        # check if criterion or loss function was passed in the configuration
        if 'criterion' in kwargs:
            self.criterion = kwargs['criterion']
        else:
            
            # Raise an error if criterion is missing
            logging.error("Criterion / Loss Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Criterion / Loss Object not Found")
        
        # check if batch size is present in the configuration
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            
            # Raise an error if batch size is not present 
            logging.error("Batch Size not found amongst the Arguments")
            self.error_print()
            raise Exception("Batch Size not Found")
        
        # Check if Optimizer is present in the configuration
        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']
        else:
            
            # Raise an error if optimizer is not present in configuration
            logging.error("Optimizer Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Optimizer Object not Found")
        
        # Check if scheduler is present in the configuration
        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
        else:
            
            # Notify the user that scheduler wasn't specified. Revert to not using the scheduler 
            self.scheduler=None
            logging.warning("Scheduler Object not found amongst the Arguments. Ignore warning if scheduler wasn't meant to be in the loop")
            
        # Check if number of epochs is present in the configuration
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            
            # Raise an error if number of epochs is not present in the configuration
            logging.error("Number of epochs not found amongst the Arguments")
            self.error_print()
            raise Exception("number of epochs not Found")
        
        # Check if early stopping flag is present in the configuration
        if 'es_flag' in kwargs:
            self.es_flag = kwargs['es_flag']
        else:
            
            # Set to False if flag is not present
            self.es_flag = False
        
        # Check the mode of the training
        if 'mode' in kwargs:
            self.c_flag=kwargs['mode']
        else:
            
            # Set to regression (loss-based metric) if not present
            self.c_flag=0
            
        # Initialize the run for this model
        self.writer = SummaryWriter('runs/'+self.model_name)
        
        # Initialize the dataset with the DataLoaders
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True)

    def train(self):
        # Mark the starting time
        start = time.time()
        
        # Check the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # history of the results
        history = defaultdict(list)
        
        # scaler to help mitigate precision issues between CPU and GPU
        scaler = amp.GradScaler()
        n_accumulate = 4
        
        # Define train and val datasets
        dataloaders = {'train':self.train_loader,'val':self.val_loader}
        dataset_sizes = {'train':len(self.train_dataset),'val':len(self.val_dataset)}
        
        # Notify the model name
        print("The tensorboard model name corresponding to this model is", self.model_name)
        
        # Create the early stopping checkpoint class
        if self.es_flag:
            earlystop = self.EarlyStop()
        else:
            earlystop = None
            
        # Training Loop
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
                        
                        
                        # Backward Pass
                            
                        if phase=='train':
                            scaler.scale(loss).backward()
                            
                        # Zero grad
                        if phase=='train' and (step+1)%n_accumulate==0:
                            scaler.step(self.optimizer)
                            scaler.update()
                            if self.scheduler:
                                self.scheduler.step()
                            self.optimizer.zero_grad()
                    
                    # Classification Metric Calculation 
                    if self.c_flag ==1:
                        running_corr += self.binary_correct(output,label)
                    elif self.c_flag ==2:
                        running_corr += self.multi_correct(output,label)    
                    
                    # Loss
                    running_loss +=loss.item()*inputs.size(0)
                epoch_loss = running_loss/dataset_sizes[phase]
                if self.c_flag!=0:
                    epoch_acc = running_corr/dataset_sizes[phase]
                
                # Add metric to Tensorboard
                self.writer.add_scalar(phase+'_loss',epoch_loss,epoch)
                if self.c_flag!=0:
                    self.writer.add_scalar(phase+'_acc',epoch_acc,epoch)
                history[phase + ' loss'].append(epoch_loss)
                print('{} Loss: {:.4f}'.format(phase,epoch_loss))
                if self.c_flag!=0:
                    history[phase+' acc'].append(epoch_acc)
                    print('{} Acc: {:.4f}'.format(phase,epoch_acc))
                
                # Perform Early Stopping
                if self.es_flag and phase=='val':
                    if self.c_flag==0:
                        # Regression metric
                        if earlystop.early_stop(epoch_loss):
                            
                            # If needs to stop training then finalize the training and return the model
                            end = time.time()
                            time_elapsed = end - start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed //3600, (time_elapsed%3600)//60,(time_elapsed%3600)%60))
                            return self.model, history
                    else:
                        # Classification Metric
                        if earlystop.early_stop(epoch_acc):
                            
                            # If needs to stop training then finalize the training and return the model
                            end = time.time()
                            time_elapsed = end - start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed //3600, (time_elapsed%3600)//60,(time_elapsed%3600)%60))
                            return self.model, history
            print("")
        
        # End the training and return the model
        end = time.time()
        time_elapsed = end - start
        print('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed //3600, (time_elapsed%3600)//60,(time_elapsed%3600)%60))
        return self.model, history
    
    def binary_correct(self,pred,label):
        '''
            Method:
                This method returns the number of correct predictions in a binary classification setting
                
            Args:
                pred    (tensor)  :   Predictions made by the model
                label   (tensor)  :   Ground Truth
            
            Output:
                (int) :   number of correct predictions
        '''
        y_pred = pred.round()
        return y_pred.eq(label).sum()
    
    def multi_correct(self,pred,label):
        '''
            Method:
                This method returns the number of correct predictions in a multi class classification setting
                
            Args:
                pred    (tensor)  :   Predictions made by the model
                label   (tensor)  :   Ground Truth
                
            Output:
                (int) :    number of correct predictions   
        '''
        _,preds = torch.max(pred,1)
        correct_preds = preds.eq(label).sum()
        return correct_preds
    
    class EarlyStop:
        '''
            Class:
                This class helps enable the early stopping functionality in this training loop
            
            Attributes:
                patience    :   number of epochs with metric within tolerating range
                min_delta   :   minimum difference to be tolerated
                q           :   Queue to maintain the last x amount of metrics
                
            Methods:
                early_stop() -> bool:
                    checks if early stop should occur or not

        '''
        def __init__(self,patience=10,min_delta=0.01):
            # Constructor class to take in values
            self.patience = patience
            self.min_delta = min_delta
            self.q = []
            
        def early_stop(self,validation_metric):
            '''
                Method:
                    This method checks if the training should be stopped early or not
                    
                Args:
                    validation_metric (float)   :   the metric that is being measured. (Regression - Val_Loss, Classification - Val_Acc)
                    
                Output:
                    (bool)  :   True if early stopping should occur, False if not
            '''
            
            # Check if the Queue has filled up
            if len(self.q)!=self.patience:
                self.q.append(validation_metric)
            else:
                
                # Calculate the average of the previous epochs
                avg_metric = sum(self.q)/self.patience
                
                # Check if the new metric is within tolerable range
                if abs(validation_metric-avg_metric)<self.min_delta:
                    
                    # return True to stop it early
                    return True
                else:
                    
                    # Pop the oldest metric and add the new one
                    self.q.pop(0)
                    self.q.append(validation_metric)
            
            # Return False if metric changed drastically in the past few epochs
            return False




