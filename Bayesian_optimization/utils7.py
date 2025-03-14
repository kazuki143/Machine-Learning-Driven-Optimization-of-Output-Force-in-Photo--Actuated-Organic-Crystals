import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import gridspec
plt.rcParams['font.family'] = 'Arial'


class BlackBoxFunc:
    def __init__(self, random_state, E_exp):
        self.random_state = random_state
        self.E_exp = E_exp
        self.mu_E = 2
        self.mu_L = 2000
        self.mu_W = 1200
        self.mu_T = 120
        self.mu_P = 50
        self.var_E = 1.5**2
        self.var_L = 1000**2
        self.var_W = 350**2
        self.var_T = 70**2
        self.var_P = 40**2
        self.coef = 15
        
    def discretize(self, E, L, P, T, W):
        """
        Transform the values of E, L, P, T, and W to be used easily as arguments in the following processes.

        Input:
        --------
        E (float): Young's modulus
        L (float): Crystal length
        P (float): Light intensity
        T (float): Crystal thickness
        W (float): Crystal width

        Return:
        --------
        E (float): The nearest predefined Young's modulus.
        L (int): Rounded crystal length.
        P (int): Rounded light intensity.
        T (int): Rounded crystal thickness.
        W (int): Rounded crystal width.
        """

        E = self.getNearestValue(E)
        L = round(L)
        W = round(W)
        T = round(T)
        P = round(P)
        return E, L, P, T, W
        
    def black_box_function(self, E, L, P, T, W):
        """
        Compute the blocking force in the simulation.
        
        Input:
        --------
        E (float): Young's modulus
        L (int): Crystal length
        P (int): Light intensity
        T (int): Crystal thickness
        W (int): Crystal width

        Return:
        --------
        y (float): The computed blocking force value
        """

        y = self.coef*np.exp(-1*(E - self.mu_E)**2/(2*self.var_E)
                             - 1*(L - self.mu_L)**2/(2*self.var_L)
                             - 1*(P - self.mu_P)**2/(2*self.var_P)
                             - 1*(T - self.mu_T)**2/(2*self.var_T)
                             - 1*(W - self.mu_W)**2/(2*self.var_W))
        return y
    
    def black_box_function_standard(self, E, L, P, T, W, points):
        """
        Compute the blocking force in the simulation with standardized input parameters.

        Input:
        --------
        E (float): Young's modulus
        L (int): Crystal length
        P (int): Light intensity
        T (int): Crystal thickness
        W (int): Crystal width
        points (list): A list containing previously obtained data for each parameter.

        Return:
        --------
        y (float): The computed blocking force value
        """

        y = self.coef*np.exp(- 1*(E*np.std(points[0]) + np.mean(points[0]) - self.mu_E)**2/(2*self.var_E)
                             - 1*(L*np.std(points[1]) + np.mean(points[1]) - self.mu_L)**2/(2*self.var_L)
                             - 1*(P*np.std(points[2]) + np.mean(points[2]) - self.mu_P)**2/(2*self.var_P)
                             - 1*(T*np.std(points[3]) + np.mean(points[3]) - self.mu_T)**2/(2*self.var_T)
                             - 1*(W*np.std(points[4]) + np.mean(points[4]) - self.mu_W)**2/(2*self.var_W))
        return y
    
    def max_minY(self, space):
        """
        Compute the maximum and minimum values of the blocking force within the given search range.

        Input:
        --------
        space (dict):  A dictionary defining the search range for each parameter.

        Return: 
        --------
        y.max() (float): The maximum blocking force within the search range. 
        y.min() (float): The minimum blocking force within the search range.
        """

        n = 30
        var = np.zeros(n*len(space)).reshape(-1, len(space))
        for i, key in enumerate(space):
            var[:, i] = np.linspace(space[key][0], space[key][1], n)
        E, L, P, T, W = np.meshgrid(self.E_exp, var[:,1], var[:,2], var[:,3], var[:,4])
        y = self.black_box_function(E, L, P, T, W)
        return y.max(), y.min()
        
    def discreteGP(self, E, L, P, T, W):
        """
        Predict the blocking force using a Gaussian Process (GP) model with discrete input parameters.

        Input:
        --------
        E (float): Young's modulus
        L (float): Crystal length
        P (float): Light intensity
        T (float): Crystal thickness
        W (float): Crystal width

        Return: 
        --------
        mu (float): Prediction value of the blocking force using a Gaussian Process (GP) model
        """

        assert E in self.E_exp
        assert type(L) == int
        assert type(W) == int
        assert type(T) == int
        assert type(P) == int
        assert P >= 0 and P <= 100
        mu = optimizer._gp.predict(E, L, P, T, W)
        return mu
    
    def getNearestValue(self, num):
        """
        Find the closest value to the given number in `self.E_exp`.

        Input:
        --------
        num (int): The target number for which the closest value in `self.E_exp` is to be found.

        Return:
        --------
        self.E_exp[idx] (float): The value in `self.E_exp` that is closest to `num`.
        """

        idx = np.abs(np.asarray(self.E_exp) - num).argmin()
        return self.E_exp[idx]
    
    def initsampling(self, space):
        """
        In simulation, generate an initial random sample within the given search space.

        Input:
        --------
        space (dict): A dictionary defining the search range for each parameter.

        Return:
        --------
        x_rand: A list containing randomly sampled integer values, one for each parameter in `space`.
        """

        x_rand = []
        for i in range(len(space)):
            rand = np.random.randint(space[i][0], space[i][1], 1)
            x_rand.append(rand.item())
        return x_rand    
    
    def save_max_min_mu(self, maxY, minY, results_folder, save_name):
        """
        Save the maximum and minimum blocking force values to a JSON file.

        Input:
        --------
        maxY (float): The maximum blocking force value.
        minY (float): The minimum blocking force value.
        results_folder (string): The directory where the JSON file will be saved.
        save_name (string): The base name for the saved JSON file.
        """

        save_dic={"max":round(maxY,3),"min":round(minY,3),"mu_E":self.mu_E,"mu_L":self.mu_L,"mu_W":self.mu_W,"mu_T":self.mu_T,"mu_P":self.mu_P}
        path_save_dic = f'{results_folder}/{save_name}_max_min_mu.json'
        json_save_dic = open(path_save_dic, mode="w")
        json.dump(save_dic, json_save_dic)
        json_save_dic.close()

def suggest_E_crystal(E):
    """
    Suggest the closest Young's modulus value to the one proposed by Bayesian optimization.

    Input:
    --------
    E (float): The Young's modulus value suggested by Bayesian optimization.

    Return:
    --------
    closest_value[1] (str): The name of the crystal compound with the closest Young's modulus.
    """

    E_list=[(0.602, "Compuond 2"), (0.664, "Compuond 9"), (0.677, "Compuond 3"), 
     (1.25, "Compuond 5"), (1.83, "Compuond 4"), (1.83, "Compuond 10"), 
     (1.92, "Compuond 1"), (2.29, "Compuond 8"), (3.55, "Compuond 7")]
    closest_value = min(E_list, key = lambda x: abs(x[0]-E))
    return closest_value[1]

def handinput():
    """
    Prompt the user to manually input experimental parameters.

    Input:
    --------
    None

    Returns:
    --------
    E (float): Young's modulus
    L (int): Calculated crystal length
    P (int): Light intensity
    T (int): Crystal thickness
    W (int): Crystal width
    """

    while True:
        try:
            print('Type Youngs modulus (GPa)')
            E = input('>>')
            print('Type crystal length (μm)')
            L_raw = input('>>')
            print('Type crystal raw pixel (px)')
            L_raw_px = input('>>')
            print('Type crystal jig position pixel (px)')
            L_jig = input('>>')
            L = float(L_raw)/float(L_raw_px)*float(L_jig)
            print(f'Crystal length is {L} (μm)')
            print('Type crystal width (μm)')
            W = float(input('>>'))
            print('Type crystal thickness (μm)')
            T = float(input('>>'))
            print('Type UV intensity (mW/cm2)')
            P = float(input('>>'))
            return float(E), int(L), int(P), int(T), int(W)
        except ValueError:
            print("One or more inputs were not valid numbers. Please try again.")    
    
# Raw file (from tact) -> a value
def getmaxY():
    """
    Extract the maximum blocking force (maxY) from an Excel file.

    Input:
    --------
    None

    Returns:
    --------
    maxY (float): The maximum blocking force value.
    """

    try:
        file = input('>>')
        df = pd.read_excel(file, engine='openpyxl')
        for i in range(len(df)):
            if df.iloc[i+1,1]>=0.2:
                save=i
                break
        maxY = df.iloc[(save+1):len(df),1].max()-df.iloc[i+1,1]
    except:
        print('Max_Y was not found. Please input manually.')
        maxY = input('>>')
        maxY = float(maxY)
    return maxY


def posterior(optimizer, x_obs, y_obs, grid):
    """
    Compute the predicted value and the uncertainty using a Gaussian Process (GP) model.

    Input:
    --------
    optimizer (object): An instance of an optimizer containing a Gaussian Process (`_gp`).
    x_obs (): The observed input data points used to update the GP model.
    y_obs (): The observed output values corresponding to `x_obs`.
    grid: The set of query points where the posterior mean and standard deviation are computed.

    Return:
    --------
    `mu`: The predicted mean values for `grid`, shape (m_samples,).
    `sigma`: The predicted standard deviations for `grid`, shape (m_samples,).
    """

    # optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, utility, number, i, key, save_folder, 
            function=None, next_candidate=None):
    """
    Visualize the Gaussian Process (GP) model and acquisition function for a given parameter.
    
    Input:
    --------
    optimizer (object): An optimizer instance that contains the Gaussian Process (`_gp`) and search space.
    utility (object): An acquisition function used to evaluate utility scores for optimization.
    number (int): The number of observed data points to be plotted.
    i (int): The index of the parameter to be visualized.
    key (str): The name of the parameter being visualized.
    save_folder (str): The name of the folder where the output plots will be saved.
    function (callable, optional): The target function to be plotted (default: None).
    next_candidate (dict, optional): The next suggested parameter set for optimization (default: None).

    Return:
    --------
    None
    """

    steps = len(optimizer.space)
    x_range = optimizer.space.bounds[i]
    n_keys = len(optimizer.space.keys)
    n_lin = 1000
    
    x_all = np.zeros(int(n_keys * n_lin)).reshape(-1, n_keys)
    for k in range(n_keys):
        if k == i:
            x = np.linspace(x_range[0], x_range[1], n_lin)
            x_vis = x.reshape(-1, 1)
        else:
            x = np.ones(n_lin)
            x = x*optimizer.res[-1]['params'][optimizer.space.keys[k]]
        x_all[:, k] = x
        
    if function is not None:
        y = function(x_all[:,0], x_all[:,1], x_all[:,2], x_all[:,3], x_all[:,4])
    else:
        y = None

    # Observed points
    y_obs_all = np.array([res['target'] for res in optimizer.res])
    y_obs = y_obs_all[-(number + 1):]
    x_obs_all = np.zeros(int(n_keys * steps)).reshape(-1, n_keys)    
    for k in range(n_keys):
        x = np.array([res['params'][optimizer.space.keys[k]] for res in optimizer.res])
        if k == i:
            x_obs_all_vis = x
        x_obs_all[:, k] = x
    x_obs = x_obs_all_vis[-(number + 1):]
    optimizer.local_max = y.max()
    optimizer.local_min = y.min()
    optimizer.local_obs_max = y_obs.max()
    
    # Set figure
    fig = plt.figure(figsize=(8, 5), tight_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # Plot setting for observed points & GP
    mu, sigma = posterior(optimizer, x_obs_all, y_obs_all, x_all)
    if y is not None:
        axis.plot(x_vis, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='r')
    axis.plot(x_vis, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x_vis, x_vis[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% CI')
    axis.set_xlim((x_range[0], x_range[1]))
    axis.set_ylim((0, 15))
    axis.set_yticks(list(np.arange(0, 15, 2.5)))
    axis.set_ylabel('Target', fontdict={'size':15})
    axis.set_xlabel(key, fontdict={'size':15})

    # Plot setting for acquision function
    utility_res = utility.utility(x_all, optimizer._gp, 0)
    acq.plot(x_vis, utility_res, label='Acquisition Func.', color='purple')
    acq.plot(next_candidate[key], np.max(utility_res), '*', markersize=15, 
             label='Next Candidate', markerfacecolor='gold', markeredgecolor='k', 
             markeredgewidth=1)
    acq.set_xlim((x_range[0], x_range[1]))
    acq.set_ylim((0, 25))
    acq.set_yticks(list(np.arange(0, 25, 5)))
    acq.set_ylabel('Acquisition', fontdict={'size':15})
    acq.set_xlabel(key, fontdict={'size':15})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    os.makedirs(f'results/{save_folder}/', exist_ok=True)
    fig.savefig(f'results/{save_folder}/{key}_{steps}_{number}.png', dpi=150)

    
def optimizer2fig(optimizer, utility, number, save_folder=None, 
                  function=None, next_candidate=None):
    """
    Generate and save Gaussian Process (GP) model visualizations for all parameters.

    Input:
    --------
    optimizer (object): An optimizer instance that contains the Gaussian Process (`_gp`) and search space.
    utility (object): An acquisition function used to evaluate utility scores for optimization.
    number (int): The number of observed data points to be plotted.
    save_folder (str, optional): The name of the folder where the output plots will be saved (default: None).
    function (callable, optional): The target function to be plotted (default: None).
    next_candidate (dict, optional): The next suggested parameter set for optimization (default: None).

    Returns:
    --------
    None
    """

    for i, key in enumerate(optimizer.space.keys):
        if key == 'P':
            plot_gp(optimizer, utility, number, i, key, save_folder, 
                    function, next_candidate)
    
def readjson(path):
    """
    Read and parse a JSON file line by line.

    Input:
    --------
    path (str): The file path of the JSON file to be read.

    Returns:
    --------
    res (list):  A list of parsed JSON objects, where each element corresponds to a line in the file.
    """
    
    res = []
    decoder = json.JSONDecoder()
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            res.append(decoder.raw_decode(line))
            line = f.readline()
    return res