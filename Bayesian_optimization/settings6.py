"""
This script contains configuration settings for Bayesian optimization experiments. 
It defines the search space, acquisition function parameters, GP kernel settings, 
and experiment-specific parameters.

Input:
# Run arguments
--------
mode (str): Operation mode, either 'experiment' or 'simulation'.
work_name (str): Name of the saved files.
repeat (int): Number of repetitions for each experiment.
n_trial (int): Number of optimization trials.
n_trial_P (int): Number of trials where parameter P is fixed.
random_state (int): Random seed for reproducibility.
init_file (str): Filename of the initial dataset.
results_folder (str): Directory where results will be stored.

# Acquisition Function:
---------------------
acquisition (str): Method for selecting the next evaluation point {'ucb', 'ei', 'poi'}.
kappa (float): Trade-off parameter for the Upper Confidence Bound (UCB) method {1, 2.5, 5, 10, 20}.
xi (float): Trade-off parameter for Expected Improvement (EI) and Probability of Improvement (POI).

# Gaussian Process Kernel:
------------------------
kernel (str): Type of Gaussian Process kernel {'rbf', 'matern'}.
length_scale (float): Length scale for RBF or Matern kernel {0.1, 1, 10, 100}.
nu (float): Smoothness parameter for the Matern kernel {1, 1.5, 2, 2.5, 3}.

# Search Space:
-------------
space (dict): Defines the parameter ranges for Bayesian optimization.
    - 'E' (tuple): Young's modulus range (GPa).
    - 'L' (tuple): Crystal length range (μm).
    - 'P' (tuple): Light intensity range (mW/cm²).
    - 'T' (tuple): Crystal thickness range (μm).
    - 'W' (tuple): Crystal width range (μm).
E_exp (list): List of experimentally measured Young's modulus values from bending tests.
"""

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