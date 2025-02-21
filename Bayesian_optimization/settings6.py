# Run arguments 
mode = 'experiment' # {'experiment', 'simulation'}
work_name = 'experiments'
repeat = 50
n_trial = 1000
n_trial_P = 0
random_state = 1
init_file = 'initial_dataset.xlsx'
results_folder = 'results/'
# -------------------------
# Acquisiiton Function
acquisition = 'ucb'  # {'ucb', 'ei', 'poi'}
kappa =  5          # UCB {1, 2.5, 5, 10, 20}
xi = 0.0             # EI & POI
# -------------------------
# GRP kernel
kernel = 'rbf'    # {'rbf', 'matern'}
length_scale = 1   # rfb, matern {0.1, 1, 10, 100}
nu = 1.5             # matern {1, 1.5, 2, 2.5, 3} 
# -------------------------
# Search space
space = {'E': (0.5, 4.0),
         'L': (0, 3000),
         'P': (21, 2080),
         'T': (50, 500),
         'W': (200, 2000)}
E_exp = [0.602, 0.664, 0.677, 1.25, 1.83, 1.83, 1.92, 2.29, 3.55] #Measured Young's modulus by bending test
# -------------------------