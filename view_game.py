import pickle
from cyborg import CageSolver

with open('cyborg_best.pkl', 'rb') as f:
    solver: CageSolver = pickle.load(f)

print(solver._play_one_game(verbose=True))