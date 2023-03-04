import numpy as np
import torch
import torch.nn as nn

from models.classification import (
    Classification,
    Classification_Dropout
    )
from models.layers import VariationalLinear
from models.regression import Regression, Regression_Dropout


class AgentBandit:
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
            'learning_rate' : model learning rate (float),
            'scheduler'     : scheduler (object) (optional),
            'n_samples'     : number of forward passes to average over (int) (Default = 1),
            'epsilon'       : probability of exploration (float) (Default = 0)
          }
        To pass these attributes to the AgentBandit please pass it using the following syntax
        agent = AgentBandit(
            model_class,
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
    Methods:
        agent_reward() -> reward (int)
            Reward achieved with given action and true label
        mushroom_update() -> outcome, reward (boolean, int)
            Given context and true label, predict the best action, and
            decide what action to take.
            NOTE: may need some changes when implementing variational
        model_update() -> loss
            Update the model with a given minibatch
        model_update_baseline() -> loss
            Given that we are not using BBB; Update the model with a given minibatch
        model_update_bbb() -> loss
            Update bbb model
    """
    def __init__(
        self,
        model_class,
        model_name,
        input_dim,
        output_dim,
        hl_type,
        hl_units,
        scheduler,
        epsilon,
        lr,
        n_samples,
    ):
        self.model_class = model_class
        self.model_name = model_name
        self.epsilon = epsilon
        self.n_samples = n_samples

        # Initialize model
        if self.model_class == Classification or self.model_class == Regression:
            self.model = model_class(input_dim, output_dim, hl_type, hl_units)
            self.criterion = nn.MSELoss()
        else:
            self.model = model_class()
            print("Define loss for non-linear layers")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def agent_reward(self, edible, eaten):
        if not eaten:
            return 0
        if edible:
            return 5
        if np.random.random() >= 0.5:
            return -35
        return 5

    def mushroom_update(self, context_eat, context_leave, edible):
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

        return eaten, agt_reward

    def model_update(self, context_batch, label_batch, batch_index):
        if self.model_class == Classification or self.model_class == Regression:
            loss = self.model_update_baseline(context_batch, label_batch)
        elif self.model_class == VariationalLinear:
            loss = self.model_update_bbb(context_batch, label_batch, batch_index)
        return loss

    def model_update_baseline(self, context_batch, label_batch):
        """Do one minibatch update on the model"""
        self.model.train()
        self.optimizer.zero_grad()
        pred_r = self.model(context_batch)
        loss = self.criterion(pred_r, label_batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def model_update_bbb(self, context_batch, label_batch, batch_index):
        """Do one minibatch update on the VARIATIONAL model"""
        self.model.train()
        self.optimizer.zero_grad()
        # What is output?
        # What is kl_div?
        print("BBB steps are not correctly implemented yet")
        raise Exception("in BBB learning")
