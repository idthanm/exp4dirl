#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/3/24
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: alg.py
# =====================================

import torch
import torch.nn as nn
import datetime
import time
import os
import numpy as np
from env import GridWorld
from utils import make_one_hot, prob2greedy


class PolicyIteration(object):
    def __init__(self, values=None, logdir=None):
        self.logdir = logdir
        if values is not None:
            self.values = values
        else:
            self.values = np.zeros((16,), dtype=np.float32)
        self.action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0

    def policy_evaluation(self, ):
        # action_prob: np.array([[p0, p1, p2, p4], ..., [p0, p1, p2, p4]])
        done = False
        while not done:
            prev_values = self.values.copy()
            self.values = self.env.value_estimate(self.action_prob.copy(), 1)
            done = True if np.max(self.values - prev_values) < 0.01 else False

    def policy_improvement(self):
        done = False
        while not done:
            prev_action_prob = self.action_prob.copy()
            self.action_prob = self.env.pim(self.values.copy())
            done = True if np.max(self.action_prob - prev_action_prob) < 0.01 else False

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()

            self.policy_improvement()
            self.policy_evaluation()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


def one_hot_encoding(batch_size, batch_data):
    out = torch.zeros(batch_size, 16)
    index = torch.LongTensor(batch_data).view(-1, 1)
    out.scatter_(1, index, 1)
    return out


def binary_encoding(states):
    convert = [torch.tensor([0, 0, 0, 0], dtype=torch.float32),
               torch.tensor([0, 0, 0, 1], dtype=torch.float32),
               torch.tensor([0, 0, 1, 0], dtype=torch.float32),
               torch.tensor([0, 0, 1, 1], dtype=torch.float32),
               torch.tensor([0, 1, 0, 0], dtype=torch.float32),
               torch.tensor([0, 1, 0, 1], dtype=torch.float32),
               torch.tensor([0, 1, 1, 0], dtype=torch.float32),
               torch.tensor([0, 1, 1, 1], dtype=torch.float32),
               torch.tensor([1, 0, 0, 0], dtype=torch.float32),
               torch.tensor([1, 0, 0, 1], dtype=torch.float32),
               torch.tensor([1, 0, 1, 0], dtype=torch.float32),
               torch.tensor([1, 0, 1, 1], dtype=torch.float32),
               torch.tensor([1, 1, 0, 0], dtype=torch.float32),
               torch.tensor([1, 1, 0, 1], dtype=torch.float32),
               torch.tensor([1, 1, 1, 0], dtype=torch.float32),
               torch.tensor([1, 1, 1, 1], dtype=torch.float32),
               ]
    out = torch.stack([convert[state] for state in states], dim=0)
    return out


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.model(x)
        x = x[:, 0]
        return x


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PointwisePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.para = torch.nn.Parameter(torch.randn((16, 4), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.para)

    def forward(self,):
        return nn.Softmax(dim=1)(self.para)


class PIwithVapprfunc(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.00003)
        self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
        self.action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros((16,), dtype=np.float32)

    def train_value(self):
        done = False
        while not done:
            prev_values = self.values.copy()
            self.values = self.env.value_estimate(self.action_prob.copy(), 1)
            done = True if np.max(self.values - prev_values) < 0.01 else False
        y = torch.from_numpy(self.values.copy())

        encode_x = one_hot_encoding(16, torch.from_numpy(np.arange(16)))
        loss = 100000
        while loss > 0.001:
            value_pred = self.valuenet(encode_x)
            self.valueopt.zero_grad()
            loss = torch.square(value_pred - y).mean()
            loss.backward()
            self.valueopt.step()
            self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()

    def policy_improvement(self):
        done = False
        while not done:
            prev_action_prob = self.action_prob.copy()
            self.action_prob = self.env.pim(self.values.copy())
            done = True if np.max(self.action_prob - prev_action_prob) < 0.01 else False

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()

            self.policy_improvement()
            self.train_value()

            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


class PIwithVandP(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.003)
        self.policynet = None
        self.policyopt = None
        self.policynet = PointwisePolicyNet()
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=0.3)
        self.action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros(16, np.float32)
        self.is_debug = False
        self.states = self.env.reset()

    def train_policy(self):
        for _ in range(10):
            self.policyopt.zero_grad()
            action_probs = self.policynet()
            behav_actions = [np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0] for _ in self.states]
            logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, behav_actions)], 0)
            IS = torch.stack([action_probs[state][action]/0.25 for state, action in zip(self.states, behav_actions)], 0)
            next_states, rewards, dones = self.env.step(self.states, behav_actions, is_reset=True)
            current_values = self.values[self.states]
            next_values = self.values[next_states]
            advs = torch.tensor(IS.detach().numpy()*(rewards + self.gamma * next_values * (1-dones)) - current_values)
            advs = torch.where(torch.greater_equal(advs, 0.1), advs, torch.zeros_like(advs))
            policy_loss = -IS.detach() * logps * advs
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.policyopt.step()
            self.states = next_states.copy()
            self.action_prob = self.policynet().detach().numpy()

            if self.is_debug:
                self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_value(self):
        done = False
        while not done:
            prev_values = self.values.copy()
            one_hot_action_prob = make_one_hot(prob2greedy(self.action_prob.copy()))
            self.values = self.env.value_estimate(one_hot_action_prob, 1)
            done = True if np.max(np.abs(self.values - prev_values)) < 0.1 else False

        y = torch.from_numpy(self.values.copy())
        loss = 10000
        while loss > 0.0001:
            self.valueopt.zero_grad()
            encode_x = one_hot_encoding(16, torch.from_numpy(np.arange(16)))
            value_pred = self.valuenet(encode_x)
            loss = torch.square(value_pred - y).mean()
            loss.backward()
            self.valueopt.step()
            self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
            loss = np.square(self.values - y.numpy()).mean()
            if self.is_debug:
                self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='value training')

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        fig_name='init', logdir=self.logdir, iter=self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()
            self.train_policy()
            self.train_value()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(),
                            fig_name='one iter end', logdir=self.logdir, iter=self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


class DIRL(object):
    def __init__(self, is_true_value, is_stationary, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.003)
        self.policynet = PointwisePolicyNet()
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=0.3)
        self.action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros(16, np.float32)
        self.is_debug = False
        self.is_true_value = is_true_value
        self.is_stationary = is_stationary
        self.states = self.env.reset() if is_stationary else np.concatenate([np.arange(16) for _ in range(64)], 0)

    def train_policy(self):
        for _ in range(10):
            self.policyopt.zero_grad()
            action_probs = self.policynet()
            values = self.env.value_estimate(action_probs.detach().numpy(), 64) if self.is_true_value else self.values.copy()
            behav_actions = [np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0] for _ in self.states]
            logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, behav_actions)], 0)
            IS = torch.stack([action_probs[state][action]/0.25 for state, action in zip(self.states, behav_actions)], 0)
            next_states, rewards, dones = self.env.step(self.states.copy(), behav_actions, is_reset=True)
            current_values = values[self.states]
            next_values = values[next_states]
            advs = torch.tensor(IS.detach().numpy()*(rewards + self.gamma * next_values * (1-dones)) - current_values)
            advs = torch.where(torch.greater_equal(advs, 0.1), advs, torch.zeros_like(advs))
            policy_loss = -IS.detach() * logps * advs
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.policyopt.step()
            self.states = next_states.copy() if self.is_stationary else self.states.copy()
            self.action_prob = self.policynet().detach().numpy()
            if self.is_debug:
                self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_value(self):
        done = False
        while not done:
            prev_values = self.values.copy()
            self.values = self.env.value_estimate(self.action_prob, 64)
            done = True if np.max(np.abs(self.values - prev_values)) < 0.1 else False
        y = torch.from_numpy(self.values.copy())

        for _ in range(10):
            self.valueopt.zero_grad()
            encode_x = one_hot_encoding(16, torch.from_numpy(np.arange(16)))
            value_pred = self.valuenet(encode_x)
            loss = torch.square(value_pred - y).mean()
            loss.backward()
            self.valueopt.step()
            self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
            if self.is_debug:
                self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='value training')

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        fig_name='init', logdir=self.logdir, iter=self.iteration)





def main(alg):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{alg}/{time}'.format(alg=alg, time=time_now)
    os.makedirs(logdir)
    if alg == 'pi_tabular':
        learner = PolicyIteration(logdir=logdir)
    elif alg == 'pi_vapprfunc':
        learner = PIwithVapprfunc(logdir=logdir)
    else:
        # pi_vandp
        learner = PIwithVandP(logdir=logdir)

    learner.train()


def test_one_hot():
    a = np.random.randint(0, 16, size=(20,))
    out = one_hot_encoding(20, torch.tensor(a))
    print(a, out)


if __name__ == '__main__':
    main('pi_vandp')
