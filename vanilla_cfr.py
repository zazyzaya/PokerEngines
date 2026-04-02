from random import choice

import numpy as np

from globals import *

# Following along from here
# https://web.archive.org/web/20250226120224/https://aipokertutorial.com/the-cfr-algorithm/#comparing-algorithms


class InfoSet:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.cum_strat = np.zeros(num_actions)
        self.cum_regret = np.zeros(num_actions)
        self.strat = self.default_strat()
        self.last_visit = 0

        # For testing, not used in algos
        self.cum_util = 0
        self.visits = 0

    def update_strat(self):
        non_neg_regret = np.maximum(self.cum_regret, 0)
        norm = non_neg_regret.sum()

        if norm == 0:
            self.strat = self.default_strat()
        else:
            self.strat = non_neg_regret / norm

        # If it's been training a really long time, these values will
        # get huge. Periodically adjust them back down to avoid
        # FLOP precision loss
        if np.sum(self.cum_strat) > 1e12:
            self.cum_strat /= 1e6

    def get_avg_strat(self):
        '''
        This is what converges to a NE, not the internal strat
        Though it itself is not used during the "training" runs
        '''
        norm_sum = self.cum_strat.sum()

        if norm_sum:
            avg_strat = self.cum_strat / norm_sum
        else:
            avg_strat = self.default_strat()

        return avg_strat

    def default_strat(self):
        return np.full(self.num_actions, 1./self.num_actions)

class CFR_Solver:
    def __init__(self):
        self.info_sets = dict()

    def get_info_set(self, game_node):
        if (info_set := self.info_sets.get(game_node.infoSet())) is None:
            self.info_sets[game_node.infoSet()] = info_set = InfoSet(len(game_node.children))

        return info_set

    def get_info_set_by_key(self, key, n_children):
        if (info_set := self.info_sets.get(key)) is None:
            self.info_sets[key] = info_set = InfoSet(n_children)

        return info_set

    def eval_game(self, game_node):
        if game_node.is_leaf:
            return game_node.reward()

        if game_node.player == C:
            v = 0
            for p, child in zip(game_node.probs, game_node.children):
                v += p * self.eval_game(child)
            return v

        # Really shouldn't run, as agents should be trained prior to this,
        # but allow it in case we want to get baseline for random strat
        info_set = self.get_info_set(game_node)

        strat = info_set.get_avg_strat()
        v = 0
        for a, child in enumerate(game_node.children):
            v += strat[a] * self.eval_game(child)

        return v

    def cfr(self, game_node, player, pi_1, pi_2):
        pi_i = pi_1 if player == P1 else pi_2
        pi_not_i = pi_2 if player == P1 else pi_1

        if game_node.is_leaf:
            r = game_node.reward()
            if player == P1:
                return r
            return -r

        if game_node.player == C: # Chance node
            child = choice(game_node.children)
            return self.cfr(child, player, pi_1, pi_2)

        info_set = self.get_info_set(game_node)
        v_sigma = 0
        cf_value = np.zeros(len(game_node.children))
        for a,c in enumerate(game_node.children):
            # Propagate into children with prob of getting here * prev prob
            if game_node.player == P1:
                cf_value[a] = self.cfr(c, player, info_set.strat[a] * pi_1, pi_2)
            elif game_node.player == P2:
                cf_value[a] = self.cfr(c, player, pi_1, info_set.strat[a] * pi_2)

            v_sigma += info_set.strat[a] * cf_value[a]

        if game_node.player == player:
            for a in range(len(game_node.children)):
                info_set.cum_regret[a] += pi_not_i * (cf_value[a] - v_sigma)
                info_set.cum_strat[a] += pi_i * info_set.strat[a]

            info_set.update_strat()
            info_set.visits += 1
            info_set.cum_util += v_sigma

        return v_sigma