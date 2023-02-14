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

## Github Basics (for collaborators)

Having some basic commands here for reference.

### Creating a branch

1) Upon opening the github repository you will most likely see a drop down on the top left of the file list as shown here:

![branch](visualization/readme_img/branch.jpg)

2) Click on the home_pc drop down and select _View all branches_.

3) Click on the _New Branch_ button and give an appropriate branch name to the branch. Also make sure to choose the branch source as __home_pc__.

### Cloning a Repository (from a branch)

1) Initially clone the repository from the main branch (Once)

```
git clone https://github.com/ms2892/advanced-ml
```

2) Pull the updates on this repository (everytime there is an update on the repository).

```
git pull
```

3) Use the git checkout command to access a particular branch

```
git checkout branch_name
```

### Pushing to a branch

1) 