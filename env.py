#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/3/24
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: main.py
# =====================================

import numpy as np
import matplotlib.pyplot as plt
from utils import rotate_coordination, make_one_hot, prob2greedy
import pylab as pl
import matplotlib.colorbar as cbar
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GridWorld(object):
    def __init__(self, agent_num=256):
        self.agent_num = agent_num
        self.states = np.random.randint(0, 16, size=(agent_num,))
        self.dones = np.zeros(agent_num, dtype=np.bool)
        self.sas = {'00': 0, '01': 1, '02': 0, '03': 0,
                    '10': 0, '11': 2, '12': 1, '13': 1,
                    '20': 1, '21': 3, '22': 2, '23': 4,
                    '30': 2, '31': 3, '32': 3, '33': 3,
                    '40': 4, '41': 4, '42': 2, '43': 6,
                    '50': 6, '51': 5, '52': 5, '53': 5,
                    '60': 7, '61': 5, '62': 4, '63': 6,
                    '70': 8, '71': 6, '72': 7, '73': 10,
                    '80': 8, '81': 7, '82': 8, '83': 9,
                    '90': 9, '91': 10, '92': 8, '93': 9,
                    '100': 9, '101': 10, '102': 7, '103': 11,
                    '110': 11, '111': 11, '112': 10, '113': 13,
                    '120': 12, '121': 13, '122': 12, '123': 12,
                    '130': 12, '131': 14, '132': 11, '133': 13,
                    '140': 13, '141': 15, '142': 14, '143': 14,
                    '150': 14, '151': 15, '152': 16, '153': 15,
                    '160': 16, '161': 16, '162': 16, '163': 15,
                    }
        self.max_l = 50
        self.gamma = 0.9
        self.reset()

    def reset(self):
        self.states = np.where(self.dones,
                               np.random.randint(0, 16, size=(self.agent_num,)),
                               self.states)
        return self.states

    def judge_done(self):
        return np.where(self.states == 16,
                        np.ones(self.agent_num, dtype=np.bool),
                        np.zeros(self.agent_num, dtype=np.bool))

    def step(self, actions, is_reset=True):
        states_list = list(self.states)
        actions_list = list(actions)
        # for i in range(16):
        #     actions_list.append(np.random.choice([0, 1, 2, 3], 1, p=action_prob[i])[0])
        self.states = np.array([self.sas[str(s) + str(a)] for (s, a) in zip(states_list, actions_list)], dtype=np.int32)
        rewards = np.array([1. if str(s) + str(a) == '152' else 0 for (s, a) in zip(states_list, actions_list)],
                           dtype=np.float32)
        self.dones = self.judge_done()
        if is_reset:
            self.reset()
        return self.states, rewards, self.dones

    def value_estimate(self, action_prob, M):
        # action_prob: np.array([[p0, p1, p2, p4], ..., [p0, p1, p2, p4]])
        states = np.arange(16)
        tiled_states = np.tile(states, (1, M))[0]
        action_list = []
        for i in range(16):
            action_list.append(list(np.random.choice([0, 1, 2, 3], 1000*M, p=action_prob[i])))
        action_list.append([0]*(1000*M))
        tiled_actions = np.concatenate([[actions4i.pop(0) for i, actions4i in enumerate(action_list)]
                                        for _ in range(M)], axis=0)
        tiled_returns = np.zeros_like(tiled_states, dtype=np.float32)
        for i in range(self.max_l):
            next_tiled_states = np.array([self.sas[str(s) + str(a)] for (s, a) in zip(tiled_states, tiled_actions)], dtype=np.int32)
            tiled_rewards = np.array([1. if str(s) + str(a) == '152' else 0 for (s, a) in zip(tiled_states, tiled_actions)],
                                      dtype=np.float32)
            tiled_returns += pow(self.gamma, i) * tiled_rewards
            tiled_states = next_tiled_states.copy()
            tiled_actions = np.array(list(map(lambda x: action_list[x].pop(0), list(tiled_states))))
        tiled_returns = np.reshape(tiled_returns, (M, 16))
        estimated_values = np.mean(tiled_returns, axis=0)
        return estimated_values

    def pim(self, values):
        # values is np.array
        values = np.concatenate([values, np.array([0.], dtype=np.float32)])
        states = np.arange(16)
        q_values_of_all_states_all_actions = []
        for action in range(0, 4):
            actions = action * np.ones_like(states, dtype=np.int32)
            next_state = np.array([self.sas[str(s) + str(a)] for (s, a) in zip(states, actions)], dtype=np.int32)
            next_values = values[next_state]
            rewards = np.array([1. if str(s) + str(a) == '152' else 0 for (s, a) in zip(states, actions)],
                               dtype=np.float32)
            q_values_of_all_states_in_this_action = rewards + self.gamma * next_values
            q_values_of_all_states_all_actions.append(q_values_of_all_states_in_this_action)
        q_values_of_all_states_all_actions = np.stack(q_values_of_all_states_all_actions, 0)
        optimal_actions = np.argmax(q_values_of_all_states_all_actions, axis=0)
        action_prob = make_one_hot(optimal_actions)
        return action_prob

    def render(self, values=None, action_prob=None, logdir=None, iter=None):
        l = 3
        big_square_l = l*6
        big_square_w = l*4
        linewidth = 1
        fig = plt.figure()
        for ax in fig.get_axes():
            ax.axis('off')
        ax = plt.axes([0., 0, 1.1, 1])
        ax.axis('off')
        ax.axis("equal")

        def draw_rotate_rec(x, y, a, l, w, c, face='none',):
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=c,
                                        facecolor=face, angle=-(90 - a), linewidth=linewidth))

        # big square
        draw_rotate_rec(0, 0, 0, big_square_l, big_square_w, 'black', face='none')
        # vertical lines
        plt.plot([0, 0], [big_square_w/2, -big_square_w/2], linewidth=linewidth, color='black')
        plt.plot([-l, -l], [big_square_w/2, -big_square_w/2], linewidth=linewidth, color='black')
        plt.plot([-2*l, -2*l], [big_square_w/2, -big_square_w/2], linewidth=linewidth, color='black')
        plt.plot([l, l], [big_square_w / 2, -big_square_w / 2], linewidth=linewidth, color='black')
        plt.plot([2 * l, 2 * l], [big_square_w / 2, -big_square_w / 2], linewidth=linewidth, color='black')
        # horizon lines
        plt.plot([big_square_l/2, -big_square_l/2], [0, 0], linewidth=linewidth, color='black')
        plt.plot([big_square_l/2, -big_square_l/2], [l, l], linewidth=linewidth, color='black')
        plt.plot([big_square_l/2, -big_square_l/2], [-l, -l], linewidth=linewidth, color='black')

        # walls
        draw_rotate_rec(-1.5*l, 1.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(-1.5*l, 0.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(-1.5*l, -1.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(1.5*l, 1.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(0.5*l, -0.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(0.5*l, -1.5*l, 0, l, l, 'black', face='black')
        draw_rotate_rec(1.5*l, -0.5*l, 0, l, l, 'black', face='black')

        # termimal
        draw_rotate_rec(1.5*l, -1.5*l, 0, l, l, 'none', face='red')

        i2xy = {'0': (-2.5*l, 1.5*l), '1': (-2.5*l, 0.5*l), '2': (-2.5*l, -0.5*l), '3': (-2.5*l, -1.5*l),
                '4': (-1.5*l, -0.5*l), '5': (-0.5*l, -1.5*l), '6': (-0.5*l, -0.5*l), '7': (-0.5*l, 0.5*l),
                '8': (-0.5*l, 1.5*l), '9': (0.5*l, 1.5*l), '10': (0.5*l, 0.5*l), '11': (1.5*l, 0.5*l),
                '12': (2.5*l, 1.5*l), '13': (2.5*l, 0.5*l), '14': (2.5*l, -0.5*l), '15': (2.5*l, -1.5*l)}


        fontsize = 10
        if action_prob is not None:
            for si, prob in enumerate(action_prob):
                plt.text(i2xy[str(si)][0]-0.3, i2xy[str(si)][1]+l/2-0.5, '{:.2f}'.format(prob[0]), fontsize=fontsize)
                plt.text(i2xy[str(si)][0]-0.3, i2xy[str(si)][1]-l/2+0.1, '{:.2f}'.format(prob[1]), fontsize=fontsize)
                plt.text(i2xy[str(si)][0]-l/2, i2xy[str(si)][1]-0.2, '{:.2f}'.format(prob[2]), fontsize=fontsize)
                plt.text(i2xy[str(si)][0]+l/2-1.1, i2xy[str(si)][1]-0.2, '{:.2f}'.format(prob[3]), fontsize=fontsize)
                max_i = np.argmax(prob)
                if max_i == 0:
                    plt.arrow(i2xy[str(si)][0], i2xy[str(si)][1]-0.2, 0, 0.3, color='b', head_width=0.2)
                elif max_i == 1:
                    plt.arrow(i2xy[str(si)][0], i2xy[str(si)][1]+0.2, 0, -0.3, color='b', head_width=0.2)
                elif max_i == 2:
                    plt.arrow(i2xy[str(si)][0]+0.2, i2xy[str(si)][1], -0.3, 0, color='b', head_width=0.2)
                elif max_i == 3:
                    plt.arrow(i2xy[str(si)][0]-0.2, i2xy[str(si)][1], 0.3, 0, color='b', head_width=0.2)

        # plot values
        if values is not None:
            normal = pl.Normalize(0., 1.)
            colors = pl.cm.Set2(normal(values))
            # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            for i, (value, c) in enumerate(zip(values, colors)):
                ax.add_patch(plt.Rectangle((i2xy[str(i)][0] - l / 2, i2xy[str(i)][1] - l / 2), l, l,
                                           facecolor=c,
                                           linewidth=linewidth))

            cax, _ = cbar.make_axes(ax)
            cax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.)
            cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.Set2, norm=normal)

        if logdir is not None:
            plt.savefig(logdir + '/iteration{}.pdf'.format(iter))

        plt.show()


def test_grid_world():
    grid_env = GridWorld(1024)
    states = grid_env.reset()
    for _ in range(10):
        actions = np.random.randint(0, 4, size=(1024,))
        next_states, rewards, dones = grid_env.step(actions)
        values = np.random.random(size=(16,))
        action_prob = make_one_hot(np.random.randint(0, 4, size=(16,)))
        grid_env.render(values, action_prob)
        print(next_states)


if __name__ == '__main__':
    test_grid_world()
