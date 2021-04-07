#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/3/24
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: alg.py
# =====================================

import argparse
import datetime
import json
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
    def __init__(self, is_true_value, dvf_gamma, dvf_itenum, gamma=0.9, value_iter=1, value_lr=0.00003, policy_lr=0.3,
                 batch_size=1024, true_value_estimate_sample_num=64, train_iter=80, valuenet_seed=None,
                 policynet_seed=None, is_show_fig=True, logdir=None, d0=None):
        self.logdir = logdir
        self.d0 = d0
        self.batch_size = batch_size
        self.value_iter = value_iter
        self.train_iter = train_iter
        self.is_show_fig = is_show_fig
        self.true_value_estimate_sample_num = true_value_estimate_sample_num
        if valuenet_seed is not None:
            torch.manual_seed(valuenet_seed)
        self.valuenet = ValueNet()
        self.valueopt = torch.optim.Adam(params=self.valuenet.parameters(), lr=value_lr)
        if policynet_seed is not None:
            torch.manual_seed(policynet_seed)
        self.policynet = PointwisePolicyNet()
        self.policyopt = torch.optim.Adam(params=self.policynet.parameters(), lr=policy_lr)
        self.action_prob = self.policynet().detach().numpy()
        self.env = GridWorld(batch_size)
        self.gamma = gamma
        self.iteration = 0
        self.values = np.zeros(16, np.float32)
        self.true_values = self.env.value_estimate(self.action_prob, self.true_value_estimate_sample_num)
        self.is_debug = False
        self.is_true_value = is_true_value
        self.dvf_gamma = dvf_gamma
        self.dvf_itenum = dvf_itenum
        self.writer = SummaryWriter(self.logdir)

    def train_policy(self):
        self.policyopt.zero_grad()
        action_probs = self.policynet()
        action_probs_detached = action_probs.detach().numpy()
        values = self.true_values if self.is_true_value else self.values.copy()
        states = self.sample_dvf(self.batch_size, self.dvf_gamma, self.dvf_itenum, self.d0)
        actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in states]
        logps = torch.stack([torch.log(action_probs[state][action]) for state, action in zip(states, actions)], 0)
        next_states, rewards, dones = self.env.step(states.copy(), actions, is_reset=True)
        current_values = values[states]
        next_values = values[next_states]
        advs = torch.tensor((rewards + self.gamma * next_values * (1-dones)) - current_values)
        policy_loss = -logps * advs
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policyopt.step()
        self.action_prob = self.policynet().detach().numpy()
        if self.is_debug:
            self.env.render(self.values.copy(), self.action_prob.copy(), fig_name='policy training')

    def train_value(self):
        self.true_values = self.env.value_estimate(self.action_prob, self.true_value_estimate_sample_num)
        y = torch.from_numpy(self.true_values.copy())

        for _ in range(self.value_iter):
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
        keyparams2store = {'true_value_mean': [], 'value_mean': [], 'stationary_true_value_mean': [],
                           'stationary_value_mean': [], 'dvf_true_value_mean': [], 'dvf_value_mean': [],
                           'policy_entropy': []}
        self.env.render(self.true_values.copy(), self.action_prob.copy(),
                        fig_name='true one iter end', logdir=self.logdir, iter='_true' + str(self.iteration),
                        is_show=self.is_show_fig)
        self.env.render(self.values.copy(), self.action_prob.copy(),
                        fig_name='appr one iter end', logdir=self.logdir, iter='_appr' + str(self.iteration),
                        is_show=self.is_show_fig)

        for _ in range(self.train_iter):
            print('ite{}'.format(self.iteration))
            self.iteration += 1
            self.train_value()
            self.train_policy()
            # self.writer.add_scalar('value_mean', self.true_values.mean(), global_step=self.iteration)
            self.env.render(self.true_values.copy(), self.action_prob.copy(),
                            fig_name='one iter end', logdir=self.logdir, iter='_true' + str(self.iteration),
                            is_show=self.is_show_fig)
            self.env.render(self.values.copy(), self.action_prob.copy(),
                            fig_name='one iter end', logdir=self.logdir, iter='_appr' + str(self.iteration),
                            is_show=self.is_show_fig)
            keyparams2store['true_value_mean'].append(self.true_values.mean())
            keyparams2store['value_mean'].append(self.values.mean())

            # stationary states
            stationary_states = self.sample_dvf(self.batch_size, 1., self.dvf_itenum, self.d0)
            truevalues4stationarystates = self.true_values[stationary_states]
            values4stationarystates = self.values[stationary_states]
            keyparams2store['stationary_true_value_mean'].append(truevalues4stationarystates.mean())
            keyparams2store['stationary_value_mean'].append(values4stationarystates.mean())

            # dvf states
            dvf_states = self.sample_dvf(self.batch_size, self.gamma, self.dvf_itenum, self.d0)
            truevalues4dvfstates = self.true_values[dvf_states]
            values4dvfstates = self.values[dvf_states]
            keyparams2store['dvf_true_value_mean'].append(truevalues4dvfstates.mean())
            keyparams2store['dvf_value_mean'].append(values4dvfstates.mean())

            keyparams2store['policy_entropy'].append(self.policy_entropy())
            np.save(self.logdir + '/data.npy', keyparams2store)

    def sample_dvf(self, batch_size, gamma, ite2sample, d0=None):
        action_probs = self.policynet()
        action_probs_detached = action_probs.detach().numpy()
        if gamma == 0.:
            assert batch_size % 16 == 0
            return np.random.choice(np.arange(15), size=(batch_size,)) if d0 is not None else np.random.choice(
                np.arange(16), size=(batch_size,))
            # if d0 is not None:
            #     return np.concatenate([np.array([0] + [i for i in range(15)]) for _ in range(int(batch_size/16))], 0)
            # else:
            #     return np.concatenate([np.arange(16) for _ in range(int(batch_size/16))], 0)
        elif gamma == 1.:
            assert batch_size % ite2sample == 0
            assert int(batch_size/ite2sample) % 16 == 0
            init_size = int(batch_size/ite2sample)
            states = np.random.choice(np.arange(15), size=(init_size,)) if d0 is not None else np.random.choice(
                np.arange(16), size=(init_size,))
            # states = np.concatenate([np.arange(16) for _ in range(int(init_size/16))], 0) if d0 is None else \
            #     np.concatenate([np.array([0] + [i for i in range(15)]) for _ in range(int(batch_size / 16))])
            env = GridWorld(init_size)
            all_states = [states.copy()]
            for _ in range(int(ite2sample - 1)):
                actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in states]
                states, _, _ = env.step(states.copy(), actions.copy(), is_reset=True)
                all_states.append(states.copy())
            all_states = np.concatenate(all_states, 0)
            return all_states
        else:
            assert 0. < gamma < 1.
            init_size = int(batch_size * (1 - gamma) / (1 - pow(gamma, ite2sample)))
            states = np.random.choice(np.arange(15), size=(init_size,)) if d0 is not None else\
                np.random.choice(np.arange(16), size=(init_size,))
            env = GridWorld(init_size)
            all_states = [states.copy()]
            for i in range(int(ite2sample - 1)):
                actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in states]
                states, _, _ = env.step(states.copy(), actions.copy(), is_reset=True)
                states2append = np.random.choice(states.copy(), size=(int(init_size*pow(gamma, i+1)),), replace=False)
                all_states.append(states2append.copy())
            tmp = np.concatenate(all_states, 0)
            resi = batch_size - len(tmp)
            actions = [np.random.choice([0, 1, 2, 3], 1, p=action_probs_detached[state])[0] for state in states]
            states, _, _ = env.step(states.copy(), actions.copy(), is_reset=True)
            states2append = np.random.choice(states.copy(), size=(resi,), replace=False)
            all_states.append(states2append.copy())
            all_states = np.concatenate(all_states, 0)
            return all_states

    def policy_entropy(self):
        action_probs = self.policynet()
        action_probs_detached = action_probs.detach().numpy()
        entropy_list = []
        for action_prob in action_probs_detached:
            entropy_list.append(-sum([prob*np.log(prob+1e-9) for prob in action_prob]))
        return np.mean(np.array(entropy_list))


def exp1(alg):
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{alg}/{time}'.format(alg=alg, time=time_now)
    os.makedirs(logdir)
    if alg == 'pi_tabular':
        learner = PolicyIteration(logdir=logdir)
    elif alg == 'pi_vapprfunc':
        learner = PIwithVapprfunc(logdir=logdir)
    else:
        assert alg == 'pi_vandp'
        learner = PIwithVandP(logdir=logdir)

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
    total_df.append([pd.DataFrame(dict(alg='approximate value, tabular policy', value_mean=vm_vapp, iteration=np.arange(16))),
                     pd.DataFrame(dict(alg='approximate value and policy', value_mean=vm_vandp, iteration=np.arange(16)))],
                     ignore_index=True)
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


def build_di_indi_parser(alg, case):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=int, default=0.9)
    gamma = parser.parse_args().gamma
    if alg == 'di':
        parser.add_argument('--is_true_value', type=bool, default=True)
        parser.add_argument('--dvf_gamma', type=float, default=gamma)  # used to construct batch for training
    elif alg == 'indi':
        parser.add_argument('--is_true_value', type=bool, default=False)
        parser.add_argument('--dvf_gamma', type=float, default=0.)
    elif alg == 'unify':
        parser.add_argument('--is_true_value', type=bool, default=False)
        parser.add_argument('--dvf_gamma', type=float, default=1.)
    elif alg == 'base1':
        parser.add_argument('--is_true_value', type=bool, default=True)
        parser.add_argument('--dvf_gamma', type=float, default=0.)
    elif alg == 'base2':
        parser.add_argument('--is_true_value', type=bool, default=True)
        parser.add_argument('--dvf_gamma', type=float, default=1.)
    parser.add_argument('--dvf_itenum', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--value_lr', type=float, default=0.003)
    parser.add_argument('--policy_lr', type=float, default=0.3)
    parser.add_argument('--true_value_estimate_sample_num', type=int, default=64)
    caseinfo = {'1': [None, 5], '2': [None, 30], '3': [1, 5], '4': [1, 30]}
    parser.add_argument('--d0', type=int, default=caseinfo[str(case)][0])
    parser.add_argument('--value_iter', type=int, default=caseinfo[str(case)][1])
    parser.add_argument('--train_iter', type=int, default=30)
    parser.add_argument('--valuenet_seed', type=int, default=3)
    parser.add_argument('--policynet_seed', type=int, default=4)
    parser.add_argument('--is_show_fig', type=bool, default=False)
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/dirl/case{case}/{alg}/{time}'.format(case=case, alg=alg, time=time_now)
    parser.add_argument('--logdir', type=str, default=logdir)
    return parser.parse_args()


def exp2(alg, case):
    args = build_di_indi_parser(alg, case)
    os.makedirs(args.logdir)
    with open(args.logdir + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    learner = DIRL(**vars(args))
    learner.train()


def grab_dirl_data(alg_list, case):
    df_list = []
    alg2label = {'di': 'Direct', 'indi': 'Indirect', 'unify': 'Unified',
                 'base1': 'Baseline1', 'base2': 'Baseline2'}
    for alg in alg_list:
        alg_dir = './results/toplot/dirl/case{}/{}/'.format(case, alg)
        for run_num, d in enumerate(os.listdir(alg_dir)):
            data = np.load(alg_dir + d + '/data.npy', allow_pickle=True).item()
            df1 = pd.DataFrame({'true_value_mean': data['true_value_mean'],
                                'value_mean': data['value_mean'],
                                'policy_entropy': data['policy_entropy'],
                                'iteration': np.arange(len(data['policy_entropy'])),
                                'run_num': run_num,
                                'Policy gradient': alg2label[alg],
                                'Evaluate state distribution': 'Initial'
                                })
            df2 = pd.DataFrame({'true_value_mean': data['stationary_true_value_mean'],
                                'value_mean': data['stationary_value_mean'],
                                'policy_entropy': data['policy_entropy'],
                                'iteration': np.arange(len(data['policy_entropy'])),
                                'run_num': run_num,
                                'Policy gradient': alg2label[alg],
                                'Evaluate state distribution': 'Stationary'
                                })
            # df3 = pd.DataFrame({'true_value_mean': data['dvf_true_value_mean'],
            #                     'value_mean': data['dvf_value_mean'],
            #                     'policy_entropy': data['policy_entropy'],
            #                     'iteration': np.arange(len(data['policy_entropy'])),
            #                     'run_num': run_num,
            #                     'PG': alg2label[alg],
            #                     'Evaluate state distribution': 'DVF'
            #                     })
            df_list.extend([df1, df2])
    total_df = df_list[0].append(df_list[1:]) if len(df_list) > 1 else df_list[0]
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.12, 0.12, 0.87, 0.87])
    sns.lineplot(x="iteration", y="true_value_mean", hue="Policy gradient", data=total_df,
                 style='Evaluate state distribution', linewidth=2, palette="bright", legend=False)
    ax1.set_ylabel('True value mean', fontsize=15)
    ax1.set_xlabel("Iteration", fontsize=15)
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles=handles, labels=labels, loc='best',
    #            frameon=False, fontsize=12)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f2 = plt.figure(2)
    ax2 = f2.add_axes([0.12, 0.12, 0.87, 0.87])
    sns.lineplot(x="iteration", y="value_mean", hue="Policy gradient", data=total_df,
                 style='Evaluate state distribution', linewidth=2, palette="bright", legend=False)
    ax2.set_ylabel('Approximate value mean', fontsize=15)
    ax2.set_xlabel("Iteration", fontsize=15)
    # handles, labels = ax2.get_legend_handles_labels()
    # ax2.legend(handles=handles, labels=labels, loc='best',
    #            frameon=False, fontsize=12)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f5 = plt.figure(5)
    ax5 = f5.add_axes([0.12, 0.12, 0.87, 0.87])
    sns.lineplot(x="iteration", y="policy_entropy", hue="Policy gradient", data=total_df, linewidth=2,
                 palette="bright", legend=False)
    ax5.set_ylabel('Policy entropy', fontsize=15)
    ax5.set_xlabel("Iteration", fontsize=15)
    # handles, labels = ax5.get_legend_handles_labels()
    # ax5.legend(handles=handles, labels=labels, loc='best',
    #            frameon=False, fontsize=12)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    plt.show()


if __name__ == '__main__':
    exp1('pi_vapprfunc')  # pi_tabular pi_vapprfunc pi_vandp
    # for _ in range(4):
    #     for case in [1, 2, 3, 4]:
    #         for alg in ['di', 'indi', 'unify', 'base1', 'base2']:
    #             exp2(alg, case)  # di, indi, unify, base1, base2
    # grab_dirl_data(['di', 'indi', 'unify', 'base1', 'base2'], 4)  # ['di', 'indi', 'unify', 'di_with_init']
    # cal_mean_and_std()
