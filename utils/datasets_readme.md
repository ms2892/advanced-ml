# How to use the dataset.py model

In your jupyter notebooks, importing certain datasets would be as simple as calling the particular method for that task present in the Datasets class.

## For Classification

The method described in this code section will return a PyTorch Dataset object that contains the MNIST dataset. This object can hence be forwarded to a dataloader of choice.

```
import utils.datasets as DB

datasets = DB.Datasets()

# For MNIST dataset
train_mnist_data, test_mnist_data = datasets.get_MNIST()
```

## For Regression

This code segment will return a PyTorch Dataset Object that will contain the regression dataset used for the study. This object can then be forwarded to a dataloader of the user's choice

```
import utils.datasets as DB

datasets = DB.Datasets()

# For regression dataset (default values)
train_regr_data, test_regr_data = datasets.get_regression() 
```

## For Reinforcement Learning - Download Only

This code only downloads the datasets in the ../data/UCI_Mushroom folder

```
import utils.datasets as DB

datasets = DB.Datasets()

# For Reinforcement Learning Dataset (download only)
datasets.download_UCI()
```