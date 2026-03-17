from random import random, randint

import numpy as np

from globals import *
from vanilla_cfr import CFR_Solver

# Implementing https://papers.nips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf

class MCCFR_Solver(CFR_Solver):
    def __init__(self, eps=0.6):
        super().__init__()
        self.eps = eps

    def epsilon_greedy_sampler(self, strat):
        num_actions = strat.shape[0]

        # Make sure prob distro actually sums to 1
        sample_probs = np.full(num_actions, self.eps / num_actions)
        greedy_action = strat.argmax().item()
        sample_probs[greedy_action] += 1. - self.eps

        action = np.random.multinomial(1, sample_probs).nonzero()[0].item()
        return action, sample_probs[action]

    def sample_history(self, game_node):
        history = []
        pi_z = 1. # pi^{\sigma^\prime}(z)
        reach_1, reach_2 = 1., 1. # pi^\sigma(z)

        while not game_node.is_leaf:
            if game_node.player == C:
                action = np.random.multinomial(1, game_node.probs).nonzero()[0].item()
            else:
                info_set = self.get_info_set(game_node)
                #action = np.random.multinomial(1, info_set.strat).nonzero()[0].item()
                #pi_z *= info_set.strat[action]
                action, prob = self.epsilon_greedy_sampler(info_set.strat)

                # Seperate sample prob
                pi_z *= prob

                # Sample reach probs before taking action
                history.append((game_node, action, reach_1, reach_2))

                # From actual strategy prob
                if game_node.player == P1:
                    reach_1 *= info_set.strat[action]
                else:
                    reach_2 *= info_set.strat[action]

            game_node = game_node.children[action]

        return history, game_node.reward(), pi_z, reach_1, reach_2

    def mccfr(self, t, game_node):
        trace, u_i, pi_z, pi_1_z, pi_2_z =  self.sample_history(game_node)
        tail_1, tail_2 = 1., 1.

        # Propagate pi_sigma_i(h, z) backward
        for i in range(len(trace)-1, -1, -1):
            node,action,reach_1,reach_2 = trace[i]

            if node.player == C:
                continue

            infoSet = self.get_info_set(node)

            # Regret update
            if node.player == P1:
                u_i_now = u_i
                pi_not_i_z = pi_2_z # pi_{-i}(z)
                reach_i = reach_1   # pi_i(z[I])
                tail_i_a = tail_1   # pi_i(z[I]a, z)
            elif node.player == P2:
                u_i_now = -u_i
                pi_not_i_z = pi_1_z
                reach_i = reach_2
                tail_i_a = tail_2

            tail_i = tail_i_a * infoSet.strat[action]   # pi_i(z[I], z)
            w_I = (u_i_now * pi_not_i_z) / pi_z         # w_I (obviously)

            for a in range(len(node.children)):
                if a == action:
                    regret = w_I * (tail_i_a - tail_i)
                else:
                    regret = -w_I * tail_i

                infoSet.cum_regret[a] += regret

            iterationsMissed = t - infoSet.last_visit
            infoSet.cum_strat += iterationsMissed * reach_i * infoSet.strat

            infoSet.last_visit = t
            infoSet.update_strat()

            if node.player == P1:
                tail_1 *= infoSet.strat[action]
            else:
                tail_2 *= infoSet.strat[action]