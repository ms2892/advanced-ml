from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import re
import torch
import torch.nn as nn

from models.classification import (
    CrossEntropyELBO,
    Classification,
    Classification_Dropout,
)
from models.regression import RegressionELBO, Regression, Regression_Dropout


class AgentBandit:
    """
    Class:
        The agent in UCI mushroom contextual bandit reinforcement learning problem
    Attributes:
        This class takes in a dictionary as an attribute. The format of the dictionary is as follows
        args = {
            'model_class'   : model (class),
            'model_name'    : model name (string),
            'input_dim'     : number of input nodes to model (int),
            'output_dim'    : number of output nodes to model (int),
            'hl_type'       : type of hidden layers in model (),
            'hl_units'      : number of hidden layer nodes to model (int),
            'batch_size'    : batch size (int),
            'buffer_size'   : how many previous steps to consider reward, action etc from (int),
            'learning_rate' : model learning rate (float),
            'scheduler'     : scheduler (object) (optional),
            'data_contexts' : 2D array of integers. One row = features for one mushroom (array),
            'data_labels'   : 1D binary array of true mushroom labels (array),
            'training_steps': (int),
            'n_samples'     : number of forward passes to average over (int) (Default = 1),
            'epsilon'       : probability of exploration (float) (Default = 0),
            'learning_rate' : learning rate of normal models (float)
          }
        To pass these attributes to the AgentBandit please pass it using the following syntax
        bandit = AgentBandit(**args)
    Methods:
        train() -> Model object, history (dictionary):
            This method returns the trained model and the history of the results in a dictionary.
            Depending on the mode of training, the history will contain it's respective metrics.
    """

    def __init__(self, **kwargs):
        if "model_class" in kwargs:
            self.model_class = kwargs["model_class"]
        else:
            raise Exception("Model Class Object not Found")

        if "model_name" in kwargs:
            curr_dt = str(datetime.now())
            curr_dt = re.sub("[^0-9]", "", curr_dt)
            self.model_name = kwargs["model_name"] + "_" + curr_dt
        else:
            raise Exception("Model Name Object not Found")

        if "input_dim" in kwargs:
            self.input_dim = kwargs["input_dim"]
        else:
            raise Exception("Input Dimension not Found")

        if "output_dim" in kwargs:
            self.output_dim = kwargs["output_dim"]
        else:
            raise Exception("Output Dimension not Found")

        if "hl_type" in kwargs:
            self.hl_type = kwargs["hl_type"]
        else:
            raise Exception("Hidden Layer Type not Found")

        if "hl_units" in kwargs:
            self.hl_units = kwargs["hl_units"]
        else:
            raise Exception("Hidden Layer Units not Found")

        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        else:
            raise Exception("Batch Size not Found")

        if "buffer_size" in kwargs:
            self.buffer_size = kwargs["buffer_size"]
        else:
            raise Exception("Buffer Size not Found")

        if "scheduler" in kwargs:
            self.scheduler = kwargs["scheduler"]
        else:
            self.scheduler = None

        if "data_contexts" in kwargs:
            self.data_contexts = kwargs["data_contexts"]
        else:
            raise Exception("Data Contexts not Found")

        if "data_labels" in kwargs:
            self.data_labels = kwargs["data_labels"]
        else:
            raise Exception("Data Labels not Found")

        if "training_steps" in kwargs:
            self.training_steps = kwargs["training_steps"]
        else:
            raise Exception("training Steps not Found")

        if "n_samples" in kwargs:
            self.n_samples = kwargs["n_samples"]
        else:
            print("n_samples defaults to 1")
            self.n_samples = 1

        if "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]
        else:
            self.epsilon = 0

        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
        else:
            print("note that training rate is not set")

        self.cum_regret = []
        self.reward_buffer = np.zeros(self.buffer_size)
        self.context_buffer = np.zeros((self.buffer_size, self.data_contexts.shape[1]))
        self.label_buffer = np.zeros(self.buffer_size)
        self.action_buffer = np.zeros((self.buffer_size, 2))
        self.memory_step = 0

        self.initialize_model(self.model_class)

    def initialize_model(self, model_class):
        if self.model_class == Classification or self.model_class == Regression:
            self.model = model_class(
                self.input_dim, self.output_dim, self.hl_type, self.hl_units
            )
            self.criterion = nn.MSELoss()
        else:
            self.model = model_class()
            print("Define loss for non-linear layers")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def agent_reward(self, edible, eaten):
        if not eaten:
            return 0
        if edible:
            return 5
        if np.random.random() >= 0.5:
            return -35
        return 5

    def oracle_reward(self, edible):
        return 5 * edible

    def take_action(self, mushroom):
        """
        mushroom (int): the mushroom index to learn from now
        """
        context = self.data_contexts[mushroom, :]
        edible = self.data_labels[mushroom]

        # Network takes input as tensor of [context, one-hot-encoded action]
        context_eat = torch.Tensor(
            np.append(context, (1, 0), axis=0).reshape(1, -1)
        ).double()
        context_leave = torch.Tensor(
            np.append(context, (0, 1), axis=0).reshape(1, -1)
        ).double()

        # Predict reward for each action
        with torch.no_grad():
            self.model.eval()
            eat_reward = np.mean(
                [self.model(context_eat) for _ in range(self.n_samples)]
            )
            leave_reward = np.mean(
                [self.model(context_leave) for _ in range(self.n_samples)]
            )

        # We take the action 0 (leave) or 1 (eat) based on epsilon and expected rewards
        eaten = eat_reward > leave_reward
        # Be non-greedy (explore) w probability epsilon
        if np.random.random() < self.epsilon:
            eaten = not eaten

        agt_reward = self.agent_reward(edible, eaten)
        orcl_reward = self.oracle_reward(edible)

        # Update buffers and other parameters using wrap-around memory
        new_idx = self.memory_step % self.buffer_size
        if self.memory_step == 0:
            new_regret = orcl_reward - agt_reward
        else:
            new_regret = self.cum_regret[-1] + (orcl_reward - agt_reward)
        self.cum_regret.append(new_regret)
        self.reward_buffer[new_idx] = agt_reward
        self.context_buffer[new_idx, :] = context
        self.label_buffer[new_idx] = edible
        self.action_buffer[new_idx, :] = (1, 0) if eaten else (0, 1)
        self.memory_step += 1
        """
        We kept the last 4096 reward, context and action tuples in a buffer, 
        and trained the networks using randomly drawn minibatches of size 64 for 64 training steps 
        (64 × 64 = 4096) per interaction with the Mushroom bandit. 
        """

    def learn(self, mushroom):
        # We visit the new mushroom <mushroom>
        self.take_action(mushroom)

        # Update model based on the new buffers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pick minibatches
        # For our first few mushroom visits, the buffer will not be full,
        # and so we cannot train 64 different mini-batches.
        # We will then just train as many batches as we can
        if self.memory_step < self.batch_size:
            # We have less samples than batch-size, so we must use some samples multiple times
            max_sam = self.memory_step  # Max index to pick from
            mbatch_idx = np.random.choice(max_sam, self.batch_size, replace=True)
            # Another option here would be to return without updating the model
            # Since we have so little data to train on
        else:
            max_sam = min(self.buffer_size, self.memory_step)
            mbatch_idx = np.random.choice(max_sam, self.batch_size, replace=False)

        # Train for as many times as we have full mini-batches
        if self.model_class == Classification or self.model_class == Regression:
            last_loss = self.training_baseline(mbatch_idx, device)
        elif self.model_class == CrossEntropyELBO or self.model_class == RegressionELBO:
            last_loss = self.training_bbb(mbatch_idx, device)
        else:
            raise Exception(
                "Model class has not been given a case. Look at learn method"
            )

        return last_loss

    def training_baseline(self, mbatch_idx, device):
        n_batches = len(mbatch_idx) // self.batch_size
        for idx in range(n_batches):
            self.model.train()
            self.optimizer.zero_grad()
            train_idx = mbatch_idx[
                idx * self.batch_size : (idx + 1) * self.batch_size + 1
            ]

            contents = self.context_buffer[train_idx, :]
            actions = self.action_buffer[train_idx, :]
            context_batch = (
                torch.tensor(np.concatenate((contents, actions), axis=1))
                .double()
                .to(device)
            )
            label_batch = torch.tensor(self.label_buffer[train_idx]).to(device)

            pred_r = self.model(context_batch)
            print("pred", pred_r.shape)

            loss = self.criterion(pred_r, label_batch)
            loss.backward()
            self.optimizer.step()
        return loss

    def training_bbb(self, mbatch_idx, device, weight_type="uniform"):
        n_batches = len(mbatch_idx) // self.batch_size
        for idx in range(n_batches):
            self.model.train()
            self.optimizer.zero_grad()
            train_idx = mbatch_idx[idx * self.batch_size : idx * (1 + self.batch_size)]
            contents = self.context_buffer[train_idx]
            actions = self.action_buffer[train_idx]
            context_batch = torch.tensor(
                np.concatenate((contents, actions), axis=1)
            ).to(device)
            label_batch = torch.tensor(self.label_buffer[mbatch_idx]).to(device)

            # What is output?
            # What is kl_div?
            print("BBB steps are not correctly implemented yet")
            raise Exception("in BBB learning")
            # elbo, nll = self.model.forward(output, label=label_batch, kl_div=?, dataset_size=, batch_index=idx, weight_type=weight_type)
            # Something like this...?
            nll.backward()
            self.optimizer.step()
        return nll
        print("Training w loss is not yet implemented for BBB")


if __name__ == "__main__":
    from sklearn import preprocessing
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    import utils.datasets as DB

    print("start")

    datasets = DB.Datasets()
    datasets.download_UCI()

    col_names = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surf-above-ring",
        "stalk-surf-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-color",
        "population",
        "habitat",
    ]
    mushrooms = pd.read_csv(
        "data/UCI_Mushroom/agaricus-lepiota.data", header=None, names=col_names
    )

    labels = mushrooms.pop(mushrooms.columns[0]).to_numpy()
    context = mushrooms.copy()

    # Make labels numerical
    le = preprocessing.LabelEncoder()
    labels_numerical = le.fit_transform(labels)

    # Make each feature numerical
    for colname in context:
        le = preprocessing.LabelEncoder()
        context[colname] = le.fit_transform(context[colname])
    context_numerical = context.to_numpy()

    agent_args = {
        "model_class": Regression,
        "model_name": "test_rl",
        "input_dim": 22 + 2,
        "output_dim": 1,
        "hl_type": nn.Linear,
        "hl_units": 100,
        "batch_size": 64,
        "buffer_size": 4096,
        "scheduler": None,
        "data_contexts": context_numerical,
        "data_labels": labels_numerical,
        "training_steps": 64,
        "n_samples": 1,
        "epsilon": 0.2,
        "learning_rate": 1e-3,
    }

    agent = AgentBandit(**agent_args)
    print("start training")
    for epoch in range(5):
        print(epoch)
        mushroom = np.random.randint(0, len(labels))
        agent.learn(mushroom)

    print(agent.cum_regret)
