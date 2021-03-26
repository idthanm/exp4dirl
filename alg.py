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
import os
import numpy as np
from env import GridWorld
from utils import make_one_hot


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
        prev_values = self.values.copy()
        self.values = self.env.value_estimate(self.action_prob.copy(), 1)
        done = True if np.max(self.values - prev_values) < 0.01 else False
        return done

    def policy_improvement(self):
        prev_action_prob = self.action_prob.copy()
        self.action_prob = self.env.pim(self.values.copy())
        done = True if np.max(self.action_prob - prev_action_prob) < 0.01 else False
        return done

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()

            done_pim = False
            done_pev = False
            while not done_pim:
                done_pim = self.policy_improvement()
            while not done_pev:
                done_pev = self.policy_evaluation()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


def one_hot_encoding(batch_size, batch_data):
    out = torch.zeros(batch_size, 16)
    index = torch.LongTensor(batch_data).view(-1, 1)
    out.scatter_(1, index, 1)
    return out


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        x = x[:, 0]
        return x


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


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

    def train_value(self, y):
        encode_x = one_hot_encoding(16, torch.from_numpy(np.arange(16)))
        loss = 100000
        while loss > 0.001:
            value_pred = self.valuenet(encode_x)
            self.valueopt.zero_grad()
            loss = torch.square(value_pred - y).mean()
            loss.backward()
            self.valueopt.step()
            # print(loss.detach().numpy())

    def policy_evaluation(self, ):
        # action_prob: np.array([[p0, p1, p2, p4], ..., [p0, p1, p2, p4]])
        prev_values = self.values.copy()
        self.values = self.env.value_estimate(self.action_prob.copy(), 1)
        done = True if np.max(self.values - prev_values) < 0.01 else False
        return done

    def policy_improvement(self):
        prev_action_prob = self.action_prob.copy()
        self.action_prob = self.env.pim(self.values.copy())
        done = True if np.max(self.action_prob - prev_action_prob) < 0.01 else False
        return done

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()

            done_pim = False
            done_pev = False
            while not done_pim:
                done_pim = self.policy_improvement()
            while not done_pev:
                done_pev = self.policy_evaluation()
            print(self.values)
            self.train_value(torch.from_numpy(self.values.copy()))
            self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
            print(self.values)

            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


class PIwithVandP(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.policynet = PolicyNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.00003)
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=0.00003)
        self.action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        self.env = GridWorld(1024)
        self.states = self.env.states
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.random.random(16)

    def train_policy(self):
        self.policynet = PolicyNet()
        done = False
        for _ in range(100):
            prev_action_prob = self.action_prob.copy()
            action_probs = self.policynet(one_hot_encoding(16, torch.from_numpy(np.arange(16))))
            action_probs_detached = action_probs.detach().numpy()
            actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in self.states]
            logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, actions)], 0)
            next_states, rewards, dones = self.env.step(actions, is_reset=True)
            current_values = self.values[self.states]
            next_values = self.values[next_states]
            advs = torch.tensor(rewards + self.gamma * next_values - current_values)
            policy_loss = -logps * advs
            policy_loss = policy_loss.mean()
            self.policyopt.zero_grad()
            policy_loss.backward()
            print(policy_loss.detach())
            self.policyopt.step()
            self.states = next_states.copy()
            self.action_prob = self.policynet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
            # done = True if np.max(self.action_prob - prev_action_prob) < 0.000001 else False

    def train_value(self, y):
        encode_x = one_hot_encoding(16, torch.from_numpy(np.arange(16)))
        loss = 100000
        while loss > 0.1:
            value_pred = self.valuenet(encode_x)
            self.valueopt.zero_grad()
            loss = torch.square(value_pred - y).mean()
            loss.backward()
            self.valueopt.step()
            # print(loss.detach().numpy())

    def policy_evaluation(self, ):
        # action_prob: np.array([[p0, p1, p2, p4], ..., [p0, p1, p2, p4]])
        prev_values = self.values.copy()
        self.values = self.env.value_estimate(self.action_prob.copy(), 16)
        done = True if np.max(self.values - prev_values) < 0.01 else False
        return done

    def policy_improvement(self):
        prev_action_prob = self.action_prob.copy()
        self.action_prob = self.env.pim(self.values.copy())
        done = True if np.max(self.action_prob - prev_action_prob) < 0.01 else False
        return done

    def train(self):
        self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)
        done_train = False
        while not done_train:
            prev_values = self.values.copy()
            prev_action_prob = self.action_prob.copy()
            done_pev = False

            self.train_policy()
            while not done_pev:
                done_pev = self.policy_evaluation()
            print(self.values)
            self.train_value(torch.from_numpy(self.values.copy()))
            self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
            print(self.values)

            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(), self.logdir, self.iteration)

            done_train = True if np.max(self.values - prev_values) < 0.01 and \
                                 np.max(self.action_prob - prev_action_prob) < 0.01 else False


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
