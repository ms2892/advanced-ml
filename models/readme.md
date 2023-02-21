# trainer.py

This script contains the TrainModelWrapper that is responsible for running the training script for regression and classification

<pre>
models.trainer.<b style='color:pink'>TrainModelWrapper</b>(**args)
</pre>

args can be defined as a dictionary with key-value pair with the given format

<pre>
args = {
    'model'         : model (object),
    'dataset'       : training dataset (object),
    'criterion'     : loss function (object),
    'batch_size'    : batch size (int),
    'optimizer'     : optimizer (object),
    'scheduler'     : scheduler (object) (optional),
    'es_flag'       : Boolean value to decide whether to use 
                      early stopping or not. (Default = False)
    'num_epochs'    : number of epochs (int),   
    'val_size'      : number of validation data points (int)
    'mode'          : Boolean value to decide whether this training 
                      is classification training or not.(Default=0) 
                            Possible Values [0,1,2] -> 
                            0 - Regression
                            1 - Binary Classification
                            2 - MultiClass Classification
    }
</pre>

This can be used in the following way:

```
args={
    'model': model,
    'dataset': dataset,
    'criterion': criterion,
    'batch_size': 4,
    'optimizer': optimizer,
    'es_flag': True,
    'num_epochs': 100,
    'val_size': 30,
    'mode': 2
}

trainer = TrainModelWrapper(**args)

model,history = trainer.train()
```

Please refer to <i>test_trainer.ipynb</i> for examples on how to use it. The notebook can be requested from Mohd Sadiq (ms2892).


