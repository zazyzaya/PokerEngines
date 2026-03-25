from globals import P1 as RED, P2 as BLUE, C

from CybORG_plus_plus.mini_CAGE.minimal import action_mapping

GAME_LEN = 10
MOVES_FIRST = RED
MOVES_LAST = BLUE

def get_blue_actionspace(blue_state):
    pass

def get_red_actionspace(red_state):
    pass

action_map = action_mapping()

class GameNode:
    def __init__(self):
        self.children = [] # Possible actions
        self.probs = [] # Prob of taking that action (if a chance node)
        self.player = None # P1, P2, or C
        self.is_leaf = False # True if game ends at this node
        self.parent = None
        self.cur_reward = 0
        self.prev_action = 'decoy' # Spaghetti. Blue agents told to ignore
        self.depth = self.get_depth()

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

class BlueNode(GameNode):
    def __init__(self, parent, state):
        super().__init__()

        self.player = BLUE
        self.is_leaf = BLUE == MOVES_FIRST and self.check_depth() == GAME_LEN
        self.parent = parent

        if not self.is_leaf:
            self.children = get_blue_actionspace(state)

    def infoSet(self, state):
        info = state[-1]

        # Was going to track previous actions, but I don't
        # think that's relevant... decoy use is tracked already
        # Maybe should track analyze?

        # Extract relevant info from observation
        blue_obs = state[0]['Blue'][0]
        decoy_info = blue_obs[-13:]
        scan_info = blue_obs[2*-13:-13]
        activity = blue_obs[:2*-13].reshape(13,4)

        # Which machines can be decoyed
        can_decoy = decoy_info.nonzero()[0]
        can_decoy = frozenset(can_decoy)

        # Which machines red probably knows about
        scan_info = scan_info.nonzero()[0]
        scan_info = frozenset(scan_info)

        # Anomalous activity (ignore transient "connections" as they're
        # better captured in scan_info)
        anomalous = frozenset(activity[:,1].nonzero()[0])
        root = frozenset(activity[:,2].nonzero()[0]) # Need to fix, as can be "UNK" if activity == [1,0]
        user = frozenset(activity[:,3].nonzero()[0])


class RedNode(GameNode):
    def __init__(self, parent, state):
        super().__init__()

        self.player = RED
        self.parent = parent
        self.is_leaf = RED == MOVES_FIRST and self.check_depth() == GAME_LEN

        if not self.is_leaf:
            self.children = get_red_actionspace(state)
