from vanilla_cfr import CFR_Solver
from mccfr import MCCFR_Solver
from globals import P1, P2, C, AbstractGameNode

PASS = 0; BET = 1; NUM_ACTIONS = 2
K=2; Q=1; J=0

class LeafNode(AbstractGameNode):
    def __init__(self, parent, cards, player, history, depth):
        super().__init__()

        self.parent = parent
        self.cards = cards
        self.history = history
        self.player = player
        self.is_leaf = True

    def reward(self):
        # Determine pot value (only 2 if two bets, otherwise 1)
        if self.history[-2:] == f'{BET}{BET}':
            r = 2
        else:
            r = 1

        # If winner is high card
        if self.history[-1] == f'{BET}' or self.history == f'{PASS}{PASS}':
            if self.cards[1] > self.cards[0]:
                return -r
            return r

        # If winner is player that didn't fold
        if self.player == P2:
            return -r
        return r

CARD_MAP = {0:'J', 1:'Q', 2:'K'}
ACTION_MAP = {'0':'P', '1':'B'}
class PlayerNode(AbstractGameNode):
    def __init__(self, parent, cards, player, history, depth):
        super().__init__()

        self.parent = parent
        self.cards = cards
        self.player = player
        self.history = history
        self.depth = depth
        self.children = self.get_children()

    def infoSet(self):
        return f'{CARD_MAP[self.cards[self.player]]}: {"".join([ACTION_MAP[h] for h in self.history])}'

    def get_children(self):
        children = []
        if self.depth == 1:
            children = [
                PlayerNode(self, self.cards, P2, str(PASS), 2),
                PlayerNode(self, self.cards, P2, str(BET), 2)
            ]

        elif self.depth == 2:
            if self.history[-1] == str(BET):
                children = [
                    LeafNode(self, self.cards, P1, self.history + str(PASS), 3),
                    LeafNode(self, self.cards, P1, self.history + str(BET), 3)
                ]
            else:
                children = [
                    LeafNode(self, self.cards, P1, self.history + str(PASS), 3),
                    PlayerNode(self, self.cards, P1, self.history + str(BET), 3)
                ]

        else:
            children = [
                LeafNode(self, self.cards, P2, self.history + str(PASS), 4),
                LeafNode(self, self.cards, P2, self.history + str(BET), 4)
            ]

        return children


class KuhnRoot(AbstractGameNode):
    def __init__(self):
        super().__init__()

        self.parent = None
        self.player = C
        self.depth = 0
        self.children = self.get_children()
        self.probs = [1./len(self.children) for _ in range(len(self.children))]

    def get_children(self):
        return [
            PlayerNode(self, (c1,c2), P1, '', 1)
            for (c1,c2) in [
                (K,Q),(K,J),
                (Q,K),(Q,J),
                (J,K),(J,Q)
            ]
        ]


def train_cfr():
    solver = CFR_Solver()
    game = KuhnRoot()
    util = 0
    iters = 0
    avgs = []
    scores = []

    for t in range(100_000):
        for i in [P1,P2]:
            out = solver.cfr(game, i, 1,1)
            if i == P1:
                util += out
                iters += 1
                avgs.append(util/iters)

                scores.append(solver.eval_game(game))
                print(f"Avg util: {avgs[-1]:0.4f},\tAvg score {scores[-1]}")


    print()
    pairs = list(solver.info_sets.items())
    pairs.sort(key=lambda x : x[1].cum_util / x[1].visits, reverse=True)

    for k,v in pairs:
        print(k,v.cum_util/v.visits, sep='\t')

def train_mccfr(iters=200_000):
    game = KuhnRoot()

    log = dict()

    for eps in [0.1, 0.25, 0.5, 0.75, 0.9]:
        scores = []
        solver = MCCFR_Solver()
        for t in range(iters):
            solver.mccfr(t, game)

            scores.append(solver.eval_game(game))
            print(f"Avg score {scores[-1]}")
        log[eps] = scores

    import matplotlib.pyplot as plt
    x = range(iters)
    for eps,scores in log.items():
        plt.plot(x, scores, label=str(f'Eps: {eps:0.2f}'))

    width = 0.02
    plt.ylim([-0.055-width, -0.055+width])
    plt.axhline(y=-0.055, color='r', linestyle='--', label='Nash Eq (-0.055)')
    plt.legend()
    plt.show()

train_mccfr()