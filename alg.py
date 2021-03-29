#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/3/24
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: alg.py
# =====================================

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from env import GridWorld
from utils import make_one_hot, prob2greedy

sns.set(style="darkgrid")


class PolicyIteration(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.values = np.zeros((16,), dtype=np.float32)
        self.action_prob = make_one_hot(np.zeros((16,), dtype=np.int32))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0

    def policy_evaluation(self, ):
        # action_prob: np.array([[p0, p1, p2, p4], ..., [p0, p1, p2, p4]])
        self.values = self.env.value_estimate(self.action_prob.copy(), 1)

    def policy_improvement(self):
        self.action_prob = self.env.pim(self.values.copy())

    def train(self):
        keyparams2store = []
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        logdir=self.logdir, iter=self.iteration)
        keyparams2store.append(self.values.mean())
        for _ in range(15):
            # prev_values = self.values.copy()
            # prev_action_prob = self.action_prob.copy()

            self.policy_improvement()
            self.policy_evaluation()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(),
                            logdir=self.logdir, iter=self.iteration)
            keyparams2store.append(self.values.mean())
            np.save(self.logdir + '/data.npy', keyparams2store)
            # done_train = True if np.max(self.values - prev_values) < 0.01 and \
            #                      np.max(self.action_prob - prev_action_prob) < 0.01 else False


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
        self.para = torch.nn.Parameter(torch.randn((16, 4)))
        torch.nn.init.xavier_uniform_(self.para)

    def forward(self,):
        return nn.Softmax(dim=1)(self.para)


class PIwithVapprfunc(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.00003)
        self.values = self.valuenet(one_hot_encoding(16, torch.from_numpy(np.arange(16)))).detach().numpy()
        self.action_prob = make_one_hot(np.zeros((16,), dtype=np.int32))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros((16,), dtype=np.float32)

    def train_value(self):
        self.values = self.env.value_estimate(self.action_prob.copy(), 1)
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

    def policy_improvement(self):
        self.action_prob = self.env.pim(self.values.copy())

    def train(self):
        keyparams2store = []
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        logdir=self.logdir, iter=self.iteration)
        keyparams2store.append(self.values.mean())
        for _ in range(15):
            self.policy_improvement()
            self.train_value()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(),
                            logdir=self.logdir, iter=self.iteration)
            keyparams2store.append(self.values.mean())
            np.save(self.logdir + '/data.npy', keyparams2store)


class PIwithVandP(object):
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.00003)
        self.policynet = PointwisePolicyNet()
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=0.3)
        self.action_prob = make_one_hot(np.zeros((16,), dtype=np.int32))
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros(16, np.float32)
        self.is_debug = False
        self.states = self.env.reset()

    def train_policy_off_policy(self):
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

    def train_policy(self):
        for _ in range(10):
            self.policyopt.zero_grad()
            action_probs = self.policynet()
            action_probs_detached = action_probs.detach().numpy()
            actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in self.states]
            logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, actions)], 0)
            next_states, rewards, dones = self.env.step(self.states, actions, is_reset=True)
            current_values = self.values[self.states]
            next_values = self.values[next_states]
            advs = torch.tensor((rewards + self.gamma * next_values * (1-dones)) - current_values)
            # advs = torch.where(torch.logical_or(torch.greater_equal(advs, 0.1), torch.less_equal(advs, -0.1)),
            #                    advs, torch.zeros_like(advs))
            advs = torch.where(torch.logical_or(torch.greater_equal(torch.from_numpy(next_values), 0.2),
                                                torch.greater_equal(torch.from_numpy(rewards), 0.5)),
                               advs, torch.zeros_like(advs))
            policy_loss = -logps * advs
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.policyopt.step()
            self.states = next_states.copy()
            self.action_prob = self.policynet().detach().numpy()
            if self.is_debug:
                self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_value(self):
        one_hot_action_prob = make_one_hot(prob2greedy(self.action_prob.copy()))
        self.values = self.env.value_estimate(one_hot_action_prob, 1)

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
        keyparams2store = []
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        fig_name='init', logdir=self.logdir, iter=self.iteration)
        keyparams2store.append(self.values.mean())
        for _ in range(15):
            self.train_policy()
            self.train_value()
            self.iteration += 1
            self.env.render(self.values.copy(), self.action_prob.copy(),
                            fig_name='one iter end', logdir=self.logdir, iter=self.iteration)
            keyparams2store.append(self.values.mean())
            np.save(self.logdir + '/data.npy', keyparams2store)


class DIRL(object):
    def __init__(self, is_true_value, is_stationary, logdir=None):
        self.logdir = logdir
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=0.00003)
        self.policynet = PointwisePolicyNet()
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=0.3)
        self.action_prob = self.policynet().detach().numpy()
        self.env = GridWorld(1024)
        self.gamma = 0.9
        self.iteration = 0
        self.values = np.zeros(16, np.float32)
        self.true_values = None
        self.is_debug = False
        self.is_true_value = is_true_value
        self.is_stationary = is_stationary
        self.states = self.env.reset() if is_stationary else np.concatenate([np.arange(16) for _ in range(64)], 0)
        self.writer = SummaryWriter(self.logdir)
        self.is_off_policy = False

    def train_policy_off(self):
        self.policyopt.zero_grad()
        action_probs = self.policynet()
        values = self.true_values if self.is_true_value else self.values.copy()
        behav_actions = [np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0] for _ in self.states]
        logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, behav_actions)], 0)
        IS = torch.stack([action_probs[state][action]/0.25 for state, action in zip(self.states, behav_actions)], 0)
        next_states, rewards, dones = self.env.step(self.states.copy(), behav_actions, is_reset=True)
        current_values = values[self.states]
        next_values = values[next_states]
        advs = torch.tensor(IS.detach().numpy()*(rewards + self.gamma * next_values * (1-dones)) - current_values)
        advs = torch.where(torch.logical_or(torch.greater_equal(advs, 0.1), torch.less_equal(advs, -0.1)),
                           advs, torch.zeros_like(advs))
        policy_loss = -IS.detach() * logps * advs
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policyopt.step()
        self.states = next_states.copy() if self.is_stationary else self.states.copy()
        self.action_prob = self.policynet().detach().numpy()
        if self.is_debug:
            self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_policy(self):
        self.policyopt.zero_grad()
        action_probs = self.policynet()
        action_probs_detached = action_probs.detach().numpy()
        values = self.true_values if self.is_true_value else self.values.copy()
        actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in self.states]
        logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(self.states, actions)], 0)
        next_states, rewards, dones = self.env.step(self.states.copy(), actions, is_reset=True)
        current_values = values[self.states]
        next_values = values[next_states]
        advs = torch.tensor((rewards + self.gamma * next_values * (1-dones)) - current_values)
        # advs = torch.where(torch.logical_or(torch.greater_equal(advs, 0.1), torch.less_equal(advs, -0.1)),
        #                    advs, torch.zeros_like(advs))
        policy_loss = -logps * advs
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policyopt.step()
        self.states = next_states.copy() if self.is_stationary else self.states.copy()
        self.action_prob = self.policynet().detach().numpy()
        if self.is_debug:
            self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_value(self):
        self.true_values = self.env.value_estimate(self.action_prob, 64)
        y = torch.from_numpy(self.true_values.copy())

        for _ in range(1):
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
        for _ in range(80):
            print('ite{}'.format(self.iteration))
            self.iteration += 1
            self.train_value()
            self.train_policy()
            self.writer.add_scalar('value_mean', self.true_values.mean(), global_step=self.iteration)
            self.env.render(self.true_values.copy(), self.action_prob.copy(),
                            fig_name='one iter end', logdir=self.logdir, iter=self.iteration)


def main(alg):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{alg}/{time}'.format(alg=alg, time=time_now)
    os.makedirs(logdir)
    if alg == 'pi_tabular':
        learner = PolicyIteration(logdir=logdir)
    elif alg == 'pi_vapprfunc':
        learner = PIwithVapprfunc(logdir=logdir)
    elif alg == 'pi_vandp':
        learner = PIwithVandP(logdir=logdir)
    else:
        assert alg == 'dirl'
        learner = DIRL(is_true_value=False, is_stationary=True, logdir=logdir)

    learner.train()


def test_one_hot():
    a = np.random.randint(0, 16, size=(20,))
    out = one_hot_encoding(20, torch.tensor(a))
    print(a, out)


def plot_valuemean():
    vm_tab = np.load('./results/toplot/{}/{}/data.npy'.format('pi_tabular', '2021-03-29-14-13-23'))
    vm_vapp = np.load('./results/toplot/{}/{}/data.npy'.format('pi_vapprfunc', '2021-03-29-14-13-44'))
    vm_vandp = np.load('./results/toplot/{}/{}/data.npy'.format('pi_vandp', '2021-03-29-14-14-11'))
    total_df = pd.DataFrame(dict(alg='tabular value and policy', value_mean=vm_tab, iteration=np.arange(16)))
    # total_df.append([pd.DataFrame(dict(alg='approximate value, tabular policy', value_mean=vm_vapp, iteration=np.arange(16))),
    #                  pd.DataFrame(dict(alg='approximate value and policy', value_mean=vm_vandp, iteration=np.arange(16)))],
    #                  ignore_index=True)
    total_df = pd.DataFrame(dict(alg='approximate value, tabular policy', value_mean=vm_vapp, iteration=np.arange(16)))

    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="value_mean", hue="alg", data=total_df, linewidth=2, palette="bright")
    ax1.set_ylabel('Value mean', fontsize=15)
    ax1.set_xlabel("Iteration", fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()
    print(np.square(vm_tab-vm_vapp).mean(), np.square(vm_vapp-vm_vandp).mean(), np.square(vm_tab-vm_vandp).mean(), vm_tab)


def cal_mean_and_std():
    vm_tab = np.load('./results/toplot/{}/{}/data.npy'.format('pi_tabular', '2021-03-29-14-13-23'))
    vandp_error = []
    for vandp_dir in os.listdir('./results/toplot/pi_vandp'):
        vm_vandp = np.load('./results/toplot/pi_vandp/' + vandp_dir + '/data.npy')
        vandp_error.append(np.square(vm_vandp - vm_tab))
    vandp_mean = np.array(vandp_error).mean(axis=0)
    vandp_var = np.array(vandp_error).std(axis=0)

    vappr_error = []
    for vappr_dir in os.listdir('./results/toplot/pi_vapprfunc'):
        vm_vappr = np.load('./results/toplot/pi_vapprfunc/' + vappr_dir + '/data.npy')
        vappr_error.append(np.square(vm_vappr - vm_tab))
    vappr_mean = np.array(vappr_error).mean(axis=0)
    vappr_var = np.array(vappr_error).std(axis=0)

    print('vandp_mean: ', vandp_mean, '\n', 'vandp_std*2: ', 2*vandp_var, '\n'
          'vappr_mean: ', vappr_mean, '\n', 'vappr_std*2: ', 2*vappr_var, '\n')


if __name__ == '__main__':
    # main('pi_vandp')  # pi_tabular pi_vapprfunc pi_vandp
    cal_mean_and_std()
