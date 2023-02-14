# MLMI 4 Replication Project
Implementation of Paper for MLMI4 Coursework. This GitHub repository implements the paper [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf) written by [Charles Blundell](https://www.gatsby.ucl.ac.uk/~ucgtcbl/), [Julien Cornebise](https://cornebise.com/julien/), [Koray Kavukcuoglu](https://koray.kavukcuoglu.org/) and [Daan Wierstra](https://scholar.google.com/citations?user=aDbsf28AAAAJ&hl=en)


## Basic Structure (for collaborators)

Data: This folder would contain the data that will be used for this project. Please note that this will be dummy folder, the dataset won't actually be on the github repository but only used as a reference for local replication of the project.

Features: This folder would contain any kind of features extracted from the dataset

Models: This folder will contain the different model implementations that will be replicated throughout this project. The training script / class can be kept here. A basic empty file for classficiation, regression and reinforcement learning models are present. Feel free to change accordingly.

Notebooks: Keep the notebooks created (Finished) in this folder. Also for initial development, have the notebooks in the current directory. This way you can import modules like models.file_name.class_name or models.file_name.function_name

Report: The pdf files that can be created for this project can be created here

Utils: This folder keeps all the auxillary functions needed to implement the paper. Things like Augmentation, Resizing etc etc.

Visualization: The script / class for visualizing the models can be kept here. Also the images can be kept here as well 