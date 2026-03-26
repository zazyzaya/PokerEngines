P1=0; P2=1; C=3

# CAGE Env
DEFAULT_ARGS = [
    # Defender, ent0, ent1, ent2
    [(1,0,4,1), (1,0,4,1), (5,2,1,1), (5,2,1,1)],

    # Op0, Op1, Op2, OpServ
    [(1,0,0, .1), (1,0,0, .1), (1,0,0, .1), (1,0,4, 1)],

    # User0, User1, User2, User3, User4
    [(0,0,0,0), (2,1,4, .1), (2,1,4, .1), (4,2,2, .1), (5,2,1, .1)]
]

DEFAULT_KWARGS = [
    [dict(),dict(),dict(),{'impactable': True}],
    [dict(),dict(),dict(),dict()],
    [{'red_init': True},dict(),dict(),dict(),dict()]
]

DEFAULT_LINKS = [
    [3,7], # Ent2 -> OpServ
    [9,2], # U1 -> Ent1
    [10,2], # U2 -> Ent1
    [11,3], # U3 -> Ent2
    [12,3]  # U4 -> Ent2
]

HOSTNAMES = [
    'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
    'User0', 'User1', 'User2', 'User3', 'User4'
]

HOST_TO_IDX = {h:i for i,h in enumerate(HOSTNAMES)}

class AbstractGameNode:
    '''
    All game nodes inherit from this
    '''
    pass