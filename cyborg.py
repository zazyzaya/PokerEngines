from argparse import ArgumentParser
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import numpy as np
import ray

from globals import P1 as RED, P2 as BLUE, C, AbstractGameNode
from vanilla_cfr import InfoSet
from mccfr import MCCFR_Solver
from CybORG_plus_plus.mini_CAGE.minimal import action_mapping, SimplifiedCAGE
from CybORG_plus_plus.mini_CAGE.red_bline_agent import B_line_minimal

GAME_LEN = 10

# Want rewards to be positive, so must shift them by the approximate
# floor for worst game (sleep agent.) This way, scores are positive
# for blue agent iff they outperform the sleep agent in some way,
# negative otherwise.
BASELINE_SHIFT = {10: 10, 20: 170, 30: 240, 50: 480, 100: 1135}
BASELINE_RND =   {10: 8,  20: 73,  30: 165, 50: 450, 100: 918}

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
        #threats[activity[:, 0] > 0] = 1 # Scanned (ignore)
        threats[activity[:, 1] > 0] = 1 # Anomalous
        threats[activity[:, 2] > 0] = 3 # Root
        threats[activity[:, 3] > 0] = 2 # User

        # 2. Targeted Action Pruning
        if not self.is_leaf:
            legal_actions = env.get_mask(
                env.state, env.current_decoys
            )['Blue'].nonzero()[1]

            pruned_actions = []

            for action in legal_actions:
                # Never sleep
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

@ray.remote
def parallel_gen_hist_job(info_sets, n_episodes, eps):
    env = SimplifiedCAGE(1)
    red = B_line_minimal()

    return [
        gen_hist_job(info_sets, eps, env, red)
        for _ in range(n_episodes)
    ]

def gen_hist_job(info_sets, eps, env, red_agent):
    game_node = RedNode(None, None, None, 0)

    history = []
    pi_z = 1. # pi^{\sigma^\prime}(z)
    reach_1, reach_2 = 1., 1. # pi^\sigma(z)

    env.reset()
    red_agent.reset()

    while not game_node.is_leaf:
        # Not relevant, but will keep it in for now
        if game_node.player == C:
            action = np.random.multinomial(1, game_node.probs).nonzero()[0].item()

        # For now, only optimize blue strat
        elif game_node.player == BLUE:
            strat = info_sets.get(
                game_node.infoSet(),
                np.full(len(game_node.children), 1/len(game_node.children))
            )

            sample_probs = np.full(len(strat), eps / len(strat))
            greedy_action = strat.argmax()
            sample_probs[greedy_action] += 1. - eps

            action_idx = np.random.multinomial(1, sample_probs).nonzero()[0].item()
            prob = sample_probs[action_idx]

            # Seperate sample prob
            pi_z *= prob

            # Sample reach probs before taking action
            history.append((
                BLUE, game_node.infoSet(), len(game_node.children),
                action_idx, reach_1, reach_2
            ))

            # From actual strategy prob
            reach_2 *= strat[action_idx]

            action = game_node.children[action_idx]
            _, r, _, _ = env.step(
                game_node.parent_action,
                np.array([action])
            )
            game_node = RedNode(game_node, action, env, r['Blue'].item())

        # Red strat
        else:
            action = red_agent.get_action(env.proc_states['Red'])
            prob = red_agent.prob

            history.append((
                RED, game_node.infoSet(), len(game_node.children),
                action, reach_1, reach_2
            ))

            # Strat prob and sample prob are the same
            pi_z *= prob
            reach_1 *= prob

            game_node = BlueNode(game_node, action, env)

    return history, game_node.reward() + BASELINE_SHIFT[GAME_LEN], pi_z, reach_1, reach_2

class CageSolver(MCCFR_Solver):
    def __init__(self):
        super().__init__()
        self.t = 0

    def _play_one_game(self, game_len=GAME_LEN, verbose=False):
        red = B_line_minimal()
        env = SimplifiedCAGE(1) # TODO parallelize
        rew = 0

        for step in range(game_len):
            obs = env.proc_states
            red_action = red.get_action(obs['Red'])

            blue = BlueNode(None, red_action, env)
            infoSet = self.get_info_set(blue)
            blue_action_idx = np.random.multinomial(
                1, infoSet.get_avg_strat()
            ).nonzero()[0]

            blue_action = blue.children[blue_action_idx]
            _,r,_,_ = env.step(red_action, blue_action)

            if verbose:
                print(action_map['Red'][red_action.item()], action_map['Blue'][blue_action.item()], r['Blue'].item())

            rew += r['Blue'].item()

        return rew

    def _play_one_game_rnd(self, game_len=GAME_LEN, verbose=False):
        red = B_line_minimal()
        env = SimplifiedCAGE(1) # TODO parallelize
        rew = 0

        for step in range(game_len):
            obs = env.proc_states
            red_action = red.get_action(obs['Red'])

            blue = BlueNode(None, red_action, env)
            blue_action = np.random.choice(
                blue.children
            )

            _,r,_,_ = env.step(red_action, np.array([blue_action]))

            if verbose:
                print(action_map['Red'][red_action.item()], action_map['Blue'][blue_action.item()], r['Blue'].item())

            rew += r['Blue'].item()

        return rew

    def _play_one_game_sleep(self, game_len=GAME_LEN, verbose=False):
        red = B_line_minimal()
        env = SimplifiedCAGE(1) # TODO parallelize
        rew = 0

        for step in range(game_len):
            obs = env.proc_states
            red_action = red.get_action(obs['Red'])

            _,r,_,_ = env.step(red_action, np.array([0]))

            if verbose:
                print(action_map['Red'][red_action.item()], "Sleep", r['Blue'].item())

            rew += r['Blue'].item()

        return rew

    def eval_game(self, n_episodes, workers):
        rewards = Parallel(n_jobs=1, prefer='processes')(
            delayed(self._play_one_game)()
            for _ in range(n_episodes)
        )

        return sum(rewards) / len(rewards)

    def one_player_mccfr(self, player, n_episodes, workers):
        st = time.time()

        strat_snapshot = {key: info.strat for key, info in self.info_sets.items()}
        strat_ptr = ray.put(strat_snapshot)

        futures = [
            parallel_gen_hist_job.remote(strat_ptr, n_episodes, self.eps)
            for _ in range(workers)
        ]
        histories = ray.get(futures)
        histories = sum(histories, [])

        en = time.time()
        sim_time = en-st

        st = time.time()
        for (trace, u_i, pi_z, pi_1_z, pi_2_z) in histories:
            self.t += 1
            tail_1, tail_2 = 1., 1.

            # Propagate pi_sigma_i(h, z) backward
            for i in range(len(trace)-1, -1, -1):
                cur_player,infoSetKey,n_children,action,reach_1,reach_2 = trace[i]

                if cur_player == C:
                    continue

                if cur_player == player:
                    # Regret update
                    if cur_player == RED:
                        u_i_now = -u_i
                        pi_not_i_z = pi_2_z # pi_{-i}(z)
                        reach_i = reach_1   # pi_i(z[I])
                        tail_i_a = tail_1   # pi_i(z[I]a, z)
                    elif cur_player == BLUE:
                        u_i_now = u_i
                        pi_not_i_z = pi_1_z
                        reach_i = reach_2
                        tail_i_a = tail_2

                    infoSet = self.get_info_set_by_key(infoSetKey, n_children)
                    tail_i = tail_i_a * infoSet.strat[action]   # pi_i(z[I], z)
                    w_I = (u_i_now * pi_not_i_z) / pi_z         # w_I (obviously)

                    for a in range(n_children):
                        if a == action:
                            regret = w_I * (tail_i_a - tail_i)
                        else:
                            regret = -w_I * tail_i

                        # Vanilla CFR
                        #infoSet.cum_regret[a] += regret

                        # CFR+ (Not really meant for MC setting--didn't see any improvemnts)
                        infoSet.cum_regret[a] = max(0.0, infoSet.cum_regret[a] + regret)

                    iterationsMissed = self.t - infoSet.last_visit

                    # Vanilla CFR
                    infoSet.cum_strat += iterationsMissed * reach_i * infoSet.strat

                    # CFR+ # Don't use bc have large batch sizes
                    #weight = (iterationsMissed * (infoSet.last_visit + self.t - 1)) / 2.0
                    #infoSet.cum_strat += weight * reach_i * infoSet.strat

                    infoSet.last_visit = self.t
                    infoSet.update_strat()

                    if player == RED:
                        tail_1 *= infoSet.strat[action]
                    else:
                        tail_2 *= infoSet.strat[action]

        en = time.time()
        return sim_time, en-st

class Buffer:
    def __init__(self, buff_size=25):
        self.arr = [-BASELINE_SHIFT[GAME_LEN] for _ in range(buff_size)]
        self.ptr = 0
        self.len = buff_size

    def push(self, item):
        self.arr[self.ptr] = item
        self.ptr += 1
        self.ptr %= self.len

    def avg(self):
        return sum(self.arr) / self.len

def train_mccfr(iters=200_000):
    solver = CageSolver()
    st = time.time()
    buff = Buffer()
    best = buff.avg()

    with open(f'log{TAG}.txt', 'w+') as f:
        f.write('episodes,explored_states,score\n')

    WORKERS = 100
    N_EPISODES = 100
    for t in range(1,iters+1):
        sim_time, algo_time = solver.one_player_mccfr(BLUE, N_EPISODES, WORKERS)
        n_states = len(solver.info_sets)

        print(f'[{t*WORKERS*N_EPISODES}] Tracking {n_states} states...')
        print(f'\tSim Time: {sim_time:0.2f}s, Algo time: {algo_time:0.2f}s')

        score = solver.eval_game(100, WORKERS)
        buff.push(score)

        print(f"\tAvg score {score:0.2f} ({(time.time() - st):0.2f}s)")
        print(f'\tRolling avg: {buff.avg():0.2f}', end='')
        st = time.time()

        with open(f'cyborg{TAG}.pkl', 'wb+') as f:
            pickle.dump(solver, f)

        if (new_best := buff.avg()) > best:
            print('*')
            with open(f'cyborg{TAG}_best.pkl', 'wb+') as f:
                pickle.dump(solver, f)

            best = new_best
        else:
            print()

        with open(f'log{TAG}.txt', 'a') as f:
            f.write(f'{t*WORKERS*N_EPISODES},{n_states},{score}\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', '--len', type=int, default=30)
    parser.add_argument('--tag')

    args = parser.parse_args()
    GAME_LEN = args.len

    tag = args.tag
    if tag:
        TAG = f'-{tag}'
    else:
        TAG = ''

    ray.init()
    train_mccfr()