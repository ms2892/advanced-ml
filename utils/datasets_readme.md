# How to use the dataset.py classes

In your jupyter notebooks, importing certain datasets would be as simple as calling the particular method for that task present in the Datasets class.

## For Classification

The method described in this code section will return a PyTorch Dataset object that contains the MNIST dataset. This object can hence be forwarded to a dataloader of choice.

```
import utils

datasets = utils.Datasets()

# For MNIST dataset
train_mnist_data, test_mnist_data = datasets.get_MNIST()
```

## For Regression

To get the PyTorch dataset