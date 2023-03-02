import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from datetime import datetime
import re


# Imports
#===========================#


# Logging Configuration - Defines the format and the severity of the logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trainer_debug.log", 'w'),
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
                'model'             : model (object),
                'model_name'        : model name (string)
                'train_dataset'     : training dataset (object),
                'criterion'         : loss function (object),
                'batch_size'        : batch size (int),
                'optimizer'         : optimizer (object),
                'scheduler'         : scheduler (object) (optional),
                'es_flag'           : Boolean value to decide whether to use early stopping or not. (Default = False)
                'num_epochs'        : number of epochs (int),   
                'val_dataset'       : Validation Dataset(object)
                'test_dataset'      : Test Dataset (object)
                'uniform_kl_weight' : whether weigthing type for the KL divergence is uniform.  (Default = True)
                'scale_const'       : normalizing constant to prevent exploding gradients. (Default = 1)
                'n_samples'         : number of weight samples to be used in the forward pass for BNN models (Default = 1)
                'mode'              : Boolean value to decide whether this training is a classification training or not. (Default = 0) 
                                        Possible Values [0,1,2,3,4,5] -> 
                                            0 - Regression
                                            1 - Binary Classification
                                            2 - MultiClass Classification
                                            3 - Bayesian Regression
                                            4 - Bayesian Binary Classification
                                            5 - Bayesian MultiClass Classification
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
                  \'train_dataset\' : training dataset (object),
                  \'criterion\'     : loss function (object),
                  \'batch_size\'    : batch size (int),
                  \'optimizer\'     : optimizer (object),
                  \'scheduler\'     : scheduler (object) (optional),
                  \'es_flag\'       : Boolean value to decide whether to use early stopping or not. (Default = False)
                  \'num_epochs\'    : number of epochs (int),   
                  \'val_dataset\'   : Validation Dataset to be passed (object)
                  \'test_dataset\'  : Test Dataset (object)
                  \'mode\': Boolean value to decide whether this training is a classification training or not. (Default = 0) 
                                      Possible Values [0,1,2] -> 
                                        0 - Regression
                                        1 - Binary Classification
                                        2 - MultiClass Classification
              }
              ''')

    def __init__(self, **kwargs):
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
        else:

            # Raise Error if model is missing
            logging.error("Model Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Model Object not Found")

        # Check if model name is present in the configuration
        if 'model_name' in kwargs:
            curr_dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.model_name = kwargs['model_name'] + '_' + curr_dt
        else:
            # Raise Error if model name is missing
            logging.error("Model Name Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Model Name Object not Found")

        # Check if validation dataset is present in the configuration
        if 'val_dataset' in kwargs:
            self.val_dataset = kwargs['val_dataset']
        else:
            # Raise Error if val_dataset is missing
            logging.error("Validation dataset not found amongst the Arguments")
            self.error_print()
            raise Exception("Validation dataset not Found")

        # Check if test dataset is present in the configuration
        if 'test_dataset' in kwargs:
            self.test_dataset = kwargs['test_dataset']
        else:
            # Raise Error if test_dataset is missing
            logging.error("Test dataset not found amongst the Arguments")
            self.error_print()
            raise Exception("Test dataset not Found")

        # Check if dataset is present in the configuration
        if 'train_dataset' in kwargs:
            self.train_dataset = kwargs['train_dataset']
        else:

            # Raise an error if Dataset is not found
            logging.error(
                "Training Dataset Object not found amongst the Arguments")
            self.error_print()
            raise Exception("Training Dataset Object not Found")

        # check if criterion or loss function was passed in the configuration
        if 'criterion' in kwargs:
            self.criterion = kwargs['criterion']
        else:

            # Raise an error if criterion is missing
            logging.error(
                "Criterion / Loss Object not found amongst the Arguments")
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
            self.scheduler = None
            logging.warning(
                "Scheduler Object not found amongst the Arguments. Ignore warning if scheduler wasn't meant to be in the loop")

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
        
        self.scale_const = 1 # set the default
        if "scale_const" in kwargs:
            self.scale_const = kwargs["scale_const"] # override default if needed
        
        self.uniform_kl_weight = True # set the default
        if "uniform_kl_weight" in kwargs:
            self.uniform_kl_weight = kwargs["uniform_kl_weight"] # override default if needed
        
        self.n_samples = 1 # set the default
        if "n_samples" in kwargs:
            self.n_samples = kwargs["n_samples"] # override default if needed

        # Check the mode of the training
        if 'mode' in kwargs:
            self.c_flag = kwargs['mode']
        else:

            # Set to regression (loss-based metric) if not present
            self.c_flag = 0

        # Initialize the run for this model
        self.writer = SummaryWriter('runs/'+self.model_name)

        # Initialize the dataset with the DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        
        if self.c_flag in [3,4,5]:
            return self.bnn_train()
        
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
        dataloaders = {'train': self.train_loader,
                       'val': self.val_loader, 'test': self.test_loader}
        dataset_sizes = {'train': len(self.train_dataset), 'val': len(
            self.val_dataset), 'test': len(self.test_dataset)}

        # Notify the model name
        print("The tensorboard model name corresponding to this model is",
              self.model_name)

        # Create the early stopping checkpoint class
        if self.es_flag:
            earlystop = self.EarlyStop()
        else:
            earlystop = None

        self.model = self.model.to(device)

        # Training Loop
        for step, epoch in enumerate(range(1, self.num_epochs+1)):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-'*10)

            for phase in ['train', 'val', 'test']:
                if (phase == 'train'):
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corr = 0

                for inputs, label in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    label = label.to(device)

                    # Forward Pass

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(inputs)
                        loss = self.criterion(output, label)

                        # Backward Pass

                        if phase == 'train':
                            scaler.scale(loss).backward()

                        # Zero grad
                        if phase == 'train':
                            self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()

                    # Classification Metric Calculation
                    if self.c_flag == 1:
                        running_corr += self.binary_correct(output, label)
                    elif self.c_flag == 2:
                        running_corr += self.multi_correct(output, label)

                    # Loss
                    running_loss += loss.item()*inputs.size(0)
                epoch_loss = running_loss/dataset_sizes[phase]
                if self.c_flag != 0:
                    epoch_acc = running_corr/dataset_sizes[phase]

                # Add metric to Tensorboard
                self.writer.add_scalar(phase+'_loss', epoch_loss, epoch)
                if self.c_flag != 0:
                    self.writer.add_scalar(phase+'_acc', epoch_acc, epoch)
                history[phase + ' loss'].append(epoch_loss)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if self.c_flag != 0:
                    history[phase+' acc'].append(epoch_acc)
                    print('{} Acc: {:.4f}'.format(phase, epoch_acc))

                # Perform Early Stopping
                if self.es_flag and phase == 'test':
                    if self.c_flag == 0:
                        # Regression metric
                        if earlystop.early_stop(epoch_loss):

                            # If needs to stop training then finalize the training and return the model
                            end = time.time()
                            time_elapsed = end - start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
                                time_elapsed // 3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
                            return self.model, history
                    else:
                        # Classification Metric
                        if earlystop.early_stop(epoch_acc):

                            # If needs to stop training then finalize the training and return the model
                            end = time.time()
                            time_elapsed = end - start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
                                time_elapsed // 3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
                            return self.model, history
            print("")

        # End the training and return the model
        end = time.time()
        time_elapsed = end - start
        print('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
        return self.model, history

    def binary_correct(self, pred, label):
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

    def multi_correct(self, pred, label):
        '''
            Method:
                This method returns the number of correct predictions in a multi class classification setting

            Args:
                pred    (tensor)  :   Predictions made by the model
                label   (tensor)  :   Ground Truth

            Output:
                (int) :    number of correct predictions   
        '''
        _, preds = torch.max(pred, 1)
        correct_preds = preds.eq(label).sum()
        return correct_preds

    def bnn_train(self):
        '''
            Method:
                This method trains the BNN based on the custom loss function and KL weight distribution
                
            Args:
                None
            
            Output:
                (model object)  :   returns the final instance of the model
                (dict)          :   contains information of all the performance metrics
        '''
        
        # Mark the starting time of the training
        start = time.time()
        
        # Check whether gpu is present or not
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Empty 
        history = defaultdict(list)

        dataloaders = {'train': self.train_loader,
                       'val': self.val_loader, 'test': self.test_loader}
        dataset_sizes = {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset),
            'test': len(self.test_dataset),
        }

        print("The tensorboard model name corresponding to this model is",
              self.model_name)

        if self.es_flag:
            earlystop = self.EarlyStop()
        else:
            earlystop = None

        for step, epoch in enumerate(range(1, self.num_epochs+1)):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-'*10)

            for phase in ['train', 'val', 'test']:
                if (phase == 'train'):
                    self.model.train()
                else:
                    self.model.eval()
                epoch_loss = 0.0 # ELBO scaled by 1/scale_const
                running_corr = 0.0
                epoch_elbo = 0.0 
                epoch_kl = 0.0
                epoch_nll = 0.0

                for batch_index, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs = inputs.view(inputs.shape[0], -1)
                    if inputs.type() == "torch.DoubleTensor":
                        inputs = inputs.float()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, kl_divergence = self.model(inputs, n_samples=self.n_samples)
                        kl_weight = self._get_kl_weight(
                            M=dataset_sizes[phase] // self.batch_size,
                            batch_index=batch_index, uniform_kl_weight=self.uniform_kl_weight,
                        )
                        loss, nll = self.criterion(
                            outputs, labels, kl_divergence, kl_weight,
                        )

                        if phase == 'train':
                            (loss / self.scale_const).backward()

                        # Zero Grad
                        if phase == 'train':
                            self.optimizer.step()
                            if self.scheduler:
                                self.scheduler.step()
                            self.optimizer.zero_grad()
                    if self.c_flag == 4:
                        probs = F.sigmoid(outputs)
                        probs = torch.mean(outputs, dim=1)
                        running_corr += self.binary_correct(probs, labels)

                    elif self.c_flag == 5:
                        probs = F.softmax(outputs, dim=-1)
                        probs = torch.mean(probs, dim=1)
                        
                        running_corr+= self.multi_correct(probs, labels)

                    epoch_loss += loss.item() / self.scale_const
                    epoch_elbo += loss.item()
                    weighted_kl = (loss - nll).item()
                    epoch_kl += weighted_kl
                    epoch_nll += nll.item()
                
                if self.c_flag != 0:
                    epoch_acc = running_corr/dataset_sizes[phase]

                self.writer.add_scalar(phase+'_loss', epoch_loss, epoch)
                self.writer.add_scalar(phase+'_elbo', epoch_elbo, epoch)
                self.writer.add_scalar(phase+'_kl', epoch_kl, epoch)
                self.writer.add_scalar(phase+'_nll', epoch_nll, epoch)
                if self.c_flag != 0:
                    self.writer.add_scalar(phase+'_acc', epoch_acc, epoch)

                history[phase+' loss'].append(epoch_loss)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if self.c_flag != 0:
                    history[phase+' acc'].append(epoch_acc)
                    print('{} Acc: {:.4f}'.format(phase, epoch_acc))

                if self.es_flag and phase == 'test':
                    if self.c_flag == 0:

                        if earlystop.early_stop(epoch_loss):
                            end = time.time()
                            time_elapsed = end-start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
                                time_elapsed//3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
                            return self.model, history
                    else:
                        if earlystop.early_stop(epoch_acc):
                            end = time.time()
                            time_elapsed = end-start
                            print('Early Stopping Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
                                time_elapsed//3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
                            return self.model, history
            print("")
        end = time.time()
        time_elapsed = end-start
        print('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed//3600, (time_elapsed % 3600)//60, (time_elapsed % 3600) % 60))
        return self.model, history


    def _get_kl_weight(self, M, batch_index=-1, uniform_kl_weight=True):
        '''
            M: number of batches
        '''
        
        kl_weight = 1 / M
        if not uniform_kl_weight:
            if batch_index == -1:
                raise Exception("Batch Index Not specified while getting Loss")
            
            kl_weight = 2**(M - batch_index) / (2**M - 1)
        
        return kl_weight


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

        def __init__(self, patience=10, min_delta=0.01):
            # Constructor class to take in values
            self.patience = patience
            self.min_delta = min_delta
            self.q = []

        def early_stop(self, validation_metric):
            '''
                Method:
                    This method checks if the training should be stopped early or not

                Args:
                    validation_metric (float)   :   the metric that is being measured. (Regression - Val_Loss, Classification - Val_Acc)

                Output:
                    (bool)  :   True if early stopping should occur, False if not
            '''

            # Check if the Queue has filled up
            if len(self.q) != self.patience:
                self.q.append(validation_metric)
            else:

                # Calculate the average of the previous epochs
                avg_metric = sum(self.q)/self.patience

                # Check if the new metric is within tolerable range
                if abs(validation_metric-avg_metric) < self.min_delta:

                    # return True to stop it early
                    return True
                else:

                    # Pop the oldest metric and add the new one
                    self.q.pop(0)
                    self.q.append(validation_metric)

            # Return False if metric changed drastically in the past few epochs
            return False
