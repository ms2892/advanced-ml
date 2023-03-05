from datetime import datetime
from models.rlagent import AgentBandit

import numpy as np
import re
import torch
import torch.nn as nn


class Environment:
    """
    Class:
        The environment in UCI mushroom contextual bandit reinforcement learning problem
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
            'scheduler'     : list to make scheduler: [object, **args] (list),
            'data_contexts' : 2D array of integers. One row = features for one mushroom (array),
            'data_labels'   : 1D binary array of true mushroom labels (array),
            'training_steps': (int),
            'n_samples'     : number of forward passes to average over (int) (Default = 1),
            'epsilon'       : probability of exploration (float) (Default = 0),
            'learning_rate' : learning rate of normal models (float)
          }
        To pass these attributes to the AgentBandit please pass it using the following syntax
        env = Environment(**args)
    Methods:
        oracle_reward() -> reward (int)
            Returns the reward an all-knowing oracle would achieve when seing a mushroom
        new_mushroom() -> last measured loss
            This method picks a random mushroom and makes the agent react to it.
            Result is added to buffers, and then train() is called
        train() -> last measured loss
            This method picks 64 minibatches of 64 samples from the buffers, and asks the
            agent to update the model based on that; one minibatch at a time
    """

    def __init__(self, **kwargs):
        if "model_class" in kwargs:
            self.model_class = kwargs["model_class"]
        else:
            raise Exception("Model Class Object not Found")

        if "model_name" in kwargs:
            curr_dt = str(datetime.now())
            curr_dt = re.sub("[^0-9]", "", curr_dt)
            model_name = kwargs["model_name"] + "_" + curr_dt
        else:
            raise Exception("Model Name Object not Found")

        if "input_dim" in kwargs:
            input_dim = kwargs["input_dim"]
        else:
            raise Exception("Input Dimension not Found")

        if "output_dim" in kwargs:
            output_dim = kwargs["output_dim"]
        else:
            raise Exception("Output Dimension not Found")

        if "hl_type" in kwargs:
            hl_type = kwargs["hl_type"]
        else:
            raise Exception("Hidden Layer Type not Found")

        if "hl_units" in kwargs:
            hl_units = kwargs["hl_units"]
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
            scheduler = kwargs["scheduler"]
        else:
            scheduler = []
        self.scedule_holder = scheduler

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
            n_samples = kwargs["n_samples"]
        else:
            print("n_samples defaults to 1")
            n_samples = 1

        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = 0

        if "learning_rate" in kwargs:
            lr = kwargs["learning_rate"]
        else:
            lr = 1e-3
            print("note that training rate is not set")

        # Initialize Agent
        self.agent = AgentBandit(
            self.model_class,
            model_name,
            input_dim,
            output_dim,
            hl_type,
            hl_units,
            scheduler,
            epsilon,
            lr,
            n_samples,
        )

        # Initialize buffers
        self.cum_regret = []
        self.tp = []
        self.fp = []
        self.tn = []
        self.fn = []
        self.reward_buffer = np.zeros(self.buffer_size)
        self.context_buffer = np.zeros((self.buffer_size, self.data_contexts.shape[1]))
        self.label_buffer = np.zeros(self.buffer_size)
        self.action_buffer = np.zeros((self.buffer_size, 2))
        self.memory_step = 0

    def oracle_reward(self, edible):
        return 5 * edible

    def new_mushroom(self):
        """
        Do one training step
        """
        # Pick a mushroom
        mushroom = np.random.randint(0, len(self.data_labels))

        # Corresponding mushroom-data
        context = self.data_contexts[mushroom, :]
        edible = self.data_labels[mushroom]
        context_eat = torch.Tensor(
            np.append(context, (1, 0), axis=0).reshape(1, -1)
        ).double()
        context_leave = torch.Tensor(
            np.append(context, (0, 1), axis=0).reshape(1, -1)
        ).double()

        # Let Agent choose action
        eaten, agt_reward = self.agent.mushroom_update(
            context_eat, context_leave, edible
        )
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

        # Update false/true positive/negative stats
        self.tp.append(0)
        self.fp.append(0)
        self.tn.append(0)
        self.fn.append(0)
        if edible and eaten:
            self.tp[-1] = 1
        elif edible and not eaten:
            self.fn[-1] = 1
        elif not eaten and not edible:
            self.tn[-1] = 1
        else:
            self.fp[-1] = 1

        self.memory_step += 1

        # Now train the model on available data
        loss = self.train()
        return loss

    def train(self):
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
        n_batches = len(mbatch_idx) // self.batch_size
        for idx in range(n_batches):
            # We are now in one minibatch
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

            # Train the model through the agent
            loss = self.agent.model_update(context_batch, label_batch, idx)
        # If we are using a scheduler; take a step after all these iterations (1epoch)
        if len(self.scedule_holder) > 0:
            self.agent.scheduler.step()
        return loss
