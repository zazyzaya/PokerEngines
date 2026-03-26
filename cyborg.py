import time

import numpy as np

from globals import P1 as RED, P2 as BLUE, C, AbstractGameNode
from mccfr import MCCFR_Solver
from CybORG_plus_plus.mini_CAGE.minimal import action_mapping, SimplifiedCAGE
from CybORG_plus_plus.mini_CAGE.red_bline_agent import B_line_minimal

GAME_LEN = 30
EVAL_EPISODES = 100

MOVES_FIRST = RED
MOVES_LAST = BLUE

action_map = action_mapping()

class GameNode(AbstractGameNode):
    def __init__(self):
        self.children = [] # Possible actions
        self.probs = [] # Prob of taking that action (if a chance node)
        self.player = None # P1, P2, or C
        self.is_leaf = False # True if game ends at this node
        self.parent = None
        self.cur_reward = 0

    def reward(self):
        '''
        Recursively add the rewards from the end of each
        pair of moves
        '''
        # Root
        if self.parent is None:
            return 0

        # Only red nodes can be leaves (first player)
        elif self.player == MOVES_FIRST:
            return self.cur_reward + self.parent.reward()

        # Blue or chance nodes return parent reward
        else:
            return self.parent.reward()

    def get_depth(self):
        # Base case
        if self.parent is None:
            return 0
        # Red, non-root nodes denote a new turn is starting
        elif self.player == MOVES_FIRST:
            return self.parent.depth + 1
        else:
            return self.parent.depth

def getBlueInfoSet_naive(blue_obs):
    '''
    TODO can we compress this more?
    '''
    # Was going to track previous actions, but I don't
    # think that's relevant... decoy use is tracked already
    # Maybe should track analyze?

    # Extract relevant info from observation
    scan_info = blue_obs[2*-13:-13]
    activity = blue_obs[:2*-13].reshape(13,4)

    # Which machines can be decoyed
    decoy_info = blue_obs[-13:]
    can_decoy = decoy_info.nonzero()[0]
    can_decoy = frozenset(can_decoy)

    # Which machines red probably knows about
    scan_info = scan_info.nonzero()[0]
    scan_info = frozenset(scan_info)

    # Anomalous activity (ignore transient "connections" as they're
    # better captured in scan_info)
    anomalous = frozenset(activity[:,1].nonzero()[0])
    root = frozenset(activity[:,2].nonzero()[0])
    user = frozenset(activity[:,3].nonzero()[0])

    # [1,1] -> Root, [0,1] -> User, [1,0] -> Unk
    unk = root - user
    root = root - unk

    info_set = (
        can_decoy,
        scan_info,
        anomalous,
        root,
        user,
        unk
    )

    return info_set

def getBlueInfoSet(blue_obs):
    '''
    Compresses the 13-host network into a 4-zone macro state.
    Returns a simple string key like "1-0-2-0"
    '''
    # Extract activity (13 hosts, 4 columns: [Scanned, Anomalous, Root, User])
    activity = blue_obs[:2*-13].reshape(13,4)

    # Map to 0 (Clean), 1 (Suspicious), 2 (Compromised)
    threats = np.zeros(13, dtype=int)
    threats[activity[:, 0] > 0] = 1 # Scanned
    threats[activity[:, 1] > 0] = 1 # Anomalous
    threats[activity[:, 2] > 0] = 2 # Root
    threats[activity[:, 3] > 0] = 2 # User

    # Compress by Subnet (Take the max threat level in that zone)
    user_threat = np.max(threats[8:13])  # User hosts (8-12)
    ent_threat = np.max(threats[1:4])    # Ent hosts (1-3)
    op_threat = np.max(threats[4:7])     # Op hosts (4-6)
    opserv_threat = threats[7]           # Op Server (7)

    # Need to include decoy info
    decoy_info = blue_obs[-13:]
    can_decoy = decoy_info.nonzero()[0]
    can_decoy = frozenset(can_decoy)

    # The compressed string key
    return (user_threat, ent_threat, op_threat, opserv_threat, can_decoy)

class BlueNode(GameNode):
    def __init__(self, parent, parent_action, env):
        super().__init__()

        self.player = BLUE
        self.parent = parent
        self.depth = self.get_depth()
        self.is_leaf = BLUE == MOVES_FIRST and self.depth == GAME_LEN
        self.parent_action = parent_action

        obs = env.proc_states['Blue'][0]
        # 1. Extract the exact threat level for all 13 hosts
        obs = env.proc_states['Blue'][0]
        activity = obs[:2*-13].reshape(13,4)

        threats = np.zeros(13, dtype=int)
        threats[activity[:, 0] > 0] = 1 # Scanned
        threats[activity[:, 1] > 0] = 1 # Anomalous
        threats[activity[:, 2] > 0] = 2 # Root
        threats[activity[:, 3] > 0] = 2 # User

        # 2. Targeted Action Pruning
        if not self.is_leaf:
            legal_actions = env.get_mask(
                env.state, env.current_decoys
            )['Blue'].nonzero()[1]

            pruned_actions = []

            for action in legal_actions:
                if action == 0:
                    continue

                # Identify which host this action targets
                host_alloc = (action - 1) % 13

                # CybORG Action Index Guide:
                # 27-39: Remove | 40-52: Restore
                if action >= 27:
                    # ONLY allow Remove/Restore if this specific host is alarming
                    if threats[host_alloc] > 0:
                        pruned_actions.append(action)
                else:
                    # Always allow Analyse and Decoy actions
                    pruned_actions.append(action)

            self.children = np.array(pruned_actions)

        # 3. Build the highly-specific, but pruned InfoSet
        # We hash the exact threat array and the exact pruned actions available
        action_tuple = tuple(self.children) if not self.is_leaf else ()
        infoSet = (tuple(threats), action_tuple)

        self.infoSet = lambda : infoSet

    def __str__(self):
        decoy,scan,root,usr,unk = self.infoSet()
        return (
            'BlueNode\n'
            f'\tCan decoy: {decoy}\n'
            f'\tScanned: {scan}\n'
            f'\tRooted: {root}\n'
            f'\tUser: {usr}\n'
            f'\tUnk: {unk}'
        )

    def __repr__(self):
        return f'<{str(self)}>'

class RedNode(GameNode):
    def __init__(self, parent, parent_action, env, cur_reward):
        super().__init__()

        self.player = RED
        self.parent = parent
        self.depth = self.get_depth()
        self.is_leaf = RED == MOVES_FIRST and self.depth == GAME_LEN
        self.parent_action = parent_action
        self.infoSet = lambda : '' # TODO

        self.cur_reward = cur_reward

class CageSolver(MCCFR_Solver):
    def eval_game(self, game_node):
        rewards = []
        for e in range(EVAL_EPISODES):
            red = B_line_minimal()
            env = SimplifiedCAGE(1) # TODO parallelize
            rew = 0

            for step in range(GAME_LEN):
                obs = env.proc_states
                red_action = red.get_action(obs['Red'])

                blue = BlueNode(None, red_action, env)
                infoSet = self.get_info_set(blue)
                blue_action_idx = np.random.multinomial(
                    1, infoSet.get_avg_strat()
                ).nonzero()[0]

                blue_action = blue.children[blue_action_idx]
                _,r,_,_ = env.step(red_action, blue_action)
                rew += r['Blue']

            rewards.append(rew.item())

        return sum(rewards) / len(rewards)


    def sample_history(self, game_node):
        history = []
        env = SimplifiedCAGE(1)
        pi_z = 1. # pi^{\sigma^\prime}(z)
        reach_1, reach_2 = 1., 1. # pi^\sigma(z)
        red_agent = B_line_minimal()

        while not game_node.is_leaf:
            # Not relevant, but will keep it in for now
            if game_node.player == C:
                action = np.random.multinomial(1, game_node.probs).nonzero()[0].item()

            # For now, only optimize blue strat
            elif game_node.player == BLUE:
                info_set = self.get_info_set(game_node)
                #action = np.random.multinomial(1, info_set.strat).nonzero()[0].item()
                #pi_z *= info_set.strat[action]
                action_idx, prob = self.epsilon_greedy_sampler(info_set.strat)

                # Seperate sample prob
                pi_z *= prob

                # Sample reach probs before taking action
                history.append((game_node, action_idx, reach_1, reach_2))

                # From actual strategy prob
                reach_2 *= info_set.strat[action_idx]

                action = game_node.children[action_idx]
                _, r, _, _ = env.step(
                    game_node.parent_action,
                    np.array([action])
                )
                game_node = RedNode(game_node, action, env, r['Blue'])

            # Red strat
            else:
                action = red_agent.get_action(env.proc_states['Red'])
                prob = red_agent.prob

                history.append((game_node, action, reach_1, reach_2))

                # Strat prob and sample prob are the same
                pi_z *= prob
                reach_1 *= prob

                game_node = BlueNode(game_node, action, env)

        return history, game_node.reward(), pi_z, reach_1, reach_2

def train_mccfr(iters=200_000):
    log = dict()

    scores = []
    solver = CageSolver()
    eval_every = 1_000
    st = time.time()

    tot_sim = 0
    tot_algo = 0

    for t in range(1,iters+1):
        game = RedNode(None, None, None, 0)
        sim_time, algo_time = solver.one_player_mccfr(t, game, BLUE)
        tot_sim += sim_time
        tot_algo += algo_time

        if t and t % eval_every == 0:
            print(f'[{t}] Tracking {len(solver.info_sets)} states...')
            scores.append(solver.eval_game(game))
            print(f"\tAvg score {scores[-1]} ({time.time() - st}s)")
            print(f'\tSim Time: {tot_sim/t}s, Algo time: {tot_algo/t}s')
            st = time.time()


    import matplotlib.pyplot as plt
    x = range(0,iters,eval_every)
    plt.plot(x, scores)

    plt.legend()
    plt.show()

train_mccfr()