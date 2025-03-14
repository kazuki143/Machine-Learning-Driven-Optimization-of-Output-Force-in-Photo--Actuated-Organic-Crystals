# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, settings6, warnings, shutil
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.gaussian_process.kernels import Matern, RBF
from utils7 import *
from automation.uv_light_control import *
from automation.stage_control import *
warnings.simplefilter('ignore')


def main(mode, work_name, space, acquisition, kappa, xi, kernel, length_scale, nu, 
         E_exp=None, init_file=None, n_trial=100, n_trial_P=20, random_state=0, 
         results_folder=None, repeat=None):
    
    # Make a function to be optimized
    start_time = time.time()
    
    func = BlackBoxFunc(random_state, E_exp)  ### Generating a virtual function
    
    ################################ Simulation mode ###################################
    if mode == 'simulation':
        for number in range(repeat):
            # func = BlackBoxFunc(random_state, E_exp)  ### Generating a virtual function
            maxY, minY = func.max_minY(space)
            print(f'maxY: {maxY}')
            print(f'minY: {minY}')

            # Optimizer
            optimizer = BayesianOptimization(
                f = func.black_box_function,
                pbounds = space,
                verbose = 2,
                random_state = random_state,
                allow_duplicate_points=True
            )
            if kernel == 'matern':
                optimizer._gp.kernel = Matern(length_scale=length_scale, nu=nu)
            elif kernel == 'rbf':
                optimizer._gp.kernel = RBF(length_scale=length_scale)

            # Saving progress
            utility = UtilityFunction(kind=acquisition, kappa=kappa, xi=xi)

            # Set logfile for save
            save_name = f'{work_name}{number+1}_{acquisition}-k{kappa}-xi{xi}_{kernel}-l{length_scale}-nu{nu}-repeat{repeat}-n_trial{n_trial}-maxY{maxY:.3f}-minY{minY:.3f}'
            os.makedirs(results_folder, exist_ok=True)
            logger = JSONLogger(path=f'{results_folder}{save_name}.json')
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            # Save maximum value, minimum value and mu
            func.save_max_min_mu(minY, maxY, results_folder, save_name)

            # Initial samples
            n_init_points = 5
            existing_data = set()
            print('Initial samples')
            for i in range(n_init_points):
                while True:
                    init_point = func.initsampling(optimizer.space.bounds)
                    init_point = {key: num for key, num in zip(optimizer.space.keys, init_point)}
                    init_point = func.discretize(**init_point)
                    init_point = {key: num for key, num in zip(optimizer.space.keys, init_point)}

                    if tuple(init_point.items()) not in existing_data:
                        break

                target = func.black_box_function(**init_point)
                print(i+1, init_point, target)
                optimizer.register(
                    params = init_point,
                    target = target,
                )
                existing_data.add(tuple(init_point.items()))

            # Listing explanatory variables
            space_list=[]
            for k in space.keys():
                space_list.append(k)

            # Optimization loop for E, L, W, T, P
            for i in range(n_trial):
                print(f'Sample {i+1} start')
                
                # Compiling existing samples into a list
                points=[]
                for j in range(len(space)):
                    point = [params[j] for params in optimizer.space.params]
                    points.append(point)

                # Standardizing the obtained data
                standardized_params = [
                    {
                        space_list[m]: (p[m] - np.mean(points[m])) / np.std(points[m])
                        for m in range(len(space))
                    }
                    for p in optimizer.space.params
                ]
                
                # Exploration range in the standardized space
                standardized_pbounds={
                        space_list[m]: ((space[space_list[m]][0] - np.mean(points[m])) / np.std(points[m]), 
                                        (space[space_list[m]][1] - np.mean(points[m])) / np.std(points[m]))
                        for m in range(len(space))
                        }
                
                # Creating a new optimizer instance
                optimizer_standardized = BayesianOptimization(
                    f=func.black_box_function_standard(E=1,L=1,P=1,T=1,W=1,points=points),
                    pbounds=standardized_pbounds,
                    verbose=2,
                    random_state = random_state,
                    allow_duplicate_points=True
                )

                # Registering standardized samples
                for params, target in zip(standardized_params, optimizer.space.target):
                    optimizer_standardized.register(params=params, target=target)

                # Suggesting a new candidate point
                utility = UtilityFunction(kind=acquisition, kappa=kappa, xi=xi)
                next_point_standardized = optimizer_standardized.suggest(utility)

                # Transforming back to the original scale
                next_point_to_probe = {
                    space_list[m]: next_point_standardized[space_list[m]] * np.std(points[m]) + np.mean(points[m])
                    for m in range(len(space))
                }
                next_point_to_probe = func.discretize(**next_point_to_probe)
                next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                # Do assumed experiment at the new point
                target = func.black_box_function(**next_point_to_probe)
                # Save the experimented point
                optimizer.register(
                    params = next_point_to_probe,
                    target = target
                )

                # Optimization loop for P
                optimizer.set_bounds(new_bounds = {
                    'E': (next_point_to_probe['E'], next_point_to_probe['E']),
                    'L': (next_point_to_probe['L'], next_point_to_probe['L']),
                    'W': (next_point_to_probe['W'], next_point_to_probe['W']),
                    'T': (next_point_to_probe['T'], next_point_to_probe['T']),
                    'p': (1,100)
                })

                # Sugget a new point & discretize
                next_point_to_probe = optimizer.suggest(utility)
                next_point_to_probe = func.discretize(**next_point_to_probe)
                next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                # Save figure
                optimizer2fig(optimizer, utility, 0, save_name, func.black_box_function, next_point_to_probe)

                for j in range(n_trial_P):
                    # Exit from loop for P
                    if (optimizer.local_obs_max - optimizer.local_min) >= (optimizer.local_max - optimizer.local_min)*0.9:
                        print(f'---Local maxima was found at {len(optimizer.res)}---')
                        break

                    # Do assumed experiment at the new point
                    target = func.black_box_function(**next_point_to_probe)

                    # Save the experimented point
                    optimizer.register(
                        params = next_point_to_probe,
                        target = target
                    )

                    # Sugget a new point & discretize
                    next_point_to_probe = optimizer.suggest(utility)
                    next_point_to_probe = func.discretize(**next_point_to_probe)
                    next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                    # Save figure
                    optimizer2fig(
                        optimizer, utility, j+1, save_name, func.black_box_function, next_point_to_probe
                    )

                # Exit from E, L, W, T, P
                # Checked the exit condition of the loop
                if (optimizer.max['target']-minY) >= (maxY-minY)*0.9:
                    print(f'---Global maxima was found at {len(optimizer.res)}---')
                    break
    
    ################################ Experiment mode ###################################
    elif mode == 'experiment':
         # Optimizer & dummy object
        optimizer = BayesianOptimization(
            f = func.discreteGP,
            pbounds = space,
            verbose = 2,
            random_state = 1
        )
        
        dummy = BayesianOptimization(
            f = func.discreteGP,
            pbounds = space,
            verbose = 2,
            random_state = 1
        )
        
        # Saving progress
        utility = UtilityFunction(kind=acquisition, kappa=kappa, xi=xi)
        
        # Make results folder
        os.makedirs(results_folder, exist_ok=True)

        # Set logfile for experiment points
        logfile = f'{results_folder}log_{work_name}_exp.json'
        if os.path.exists(logfile):
            backupfile = f'{results_folder}log_{work_name}_exp_backup.json'
            shutil.copy2(logfile, backupfile)
            logger = JSONLogger(path=logfile)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer = load_logs(optimizer, logs=[backupfile])
            print("Logfile(exp) includes {} points.".format(len(optimizer.space)))

            # Added "upper_or_down" to the log file
            with open(backupfile, 'r') as bf, open(logfile, 'w') as lf:
                for line in bf:
                    log_entry = json.loads(line)
                    if 'upper_or_down' not in log_entry:
                        # Added "upper_or_down" from backup if missing "upper_or_down"  
                        for backup_line in open(backupfile, 'r'):
                            backup_entry = json.loads(backup_line)
                            if log_entry['params'] == backup_entry['params'] and 'upper_or_down' in backup_entry:
                                log_entry['upper_or_down'] = backup_entry['upper_or_down']
                                break
                    
                    # Reordered to place "upper_or_down" before "datetime"
                    if 'upper_or_down' in log_entry:
                        ordered_entry = {
                            "target": log_entry["target"],
                            "params": log_entry["params"],
                            "upper_or_down": log_entry["upper_or_down"],
                            "datetime": log_entry["datetime"]
                        }
                    else:
                        ordered_entry = log_entry

                    lf.write(json.dumps(ordered_entry) + '\n')
        else:
            logger = JSONLogger(path=logfile)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            # Initial samples
            df = pd.read_excel(init_file, index_col=0)
            dict_in_list = df.to_dict('records')
            for i in range(len(dict_in_list)):
                init_point = {'E': dict_in_list[i]['E'], 'L': dict_in_list[i]['L'], 'P': dict_in_list[i]['P'], 'T': dict_in_list[i]['T'], 'W': dict_in_list[i]['W']}
                target = dict_in_list[i]['target']
                upper_or_down = dict_in_list[i]['upper_or_down']
                optimizer.register(
                    params=init_point,
                    target=target,
                )

                # Added "upper_or_down" to the log
                with open(f'{results_folder}log_{work_name}_exp.json', 'r+') as f:
                    lines = f.readlines()  # Read all rows
                    last_line = json.loads(lines[-1])  # Retrieved the last row
                    last_line['upper_or_down'] = upper_or_down  # Added "upper_or_down"

                    # Reordered to place "upper_or_down" before "datetime"
                    ordered_entry = {
                        "target": last_line["target"],
                        "params": last_line["params"],
                        "upper_or_down": last_line["upper_or_down"],
                        "datetime": last_line["datetime"]
                    }

                    # Deleted the last row and replaced it with the updated entry
                    lines[-1] = json.dumps(ordered_entry) + '\n'

                    # Rewriting the file from the top
                    f.seek(0)
                    f.writelines(lines)

        # Set logfile for suggested points
        logfile = f'{results_folder}log_{work_name}_suggest.json'
        if os.path.exists(logfile):
            backupfile = f'{results_folder}log_{work_name}_suggest_backup.json'
            shutil.copy2(logfile, backupfile)
            logger = JSONLogger(path=logfile)
            dummy.subscribe(Events.OPTIMIZATION_STEP, logger)
            dummy = load_logs(dummy, logs=[backupfile])
            print("Logfile(suggest) includes {} points.".format(len(dummy.space)))
        else:
            logger = JSONLogger(path=logfile)
            dummy.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Definition of save_name
        save_name = f'{work_name}_{acquisition}-k{kappa}-xi{xi}_{kernel}-l{length_scale}-nu{nu}-n_trial{n_trial}'

        # Optimization loop for E, L, W, T, P
        for i in range(n_trial):
            if i >= 1:
                optimizer.set_bounds(new_bounds=space)

            # Acquiring data for Standardization
            params_list = np.array(optimizer.space.params)
            param_keys = optimizer.space.keys
            means = np.mean(params_list, axis=0)
            stds = np.std(params_list, axis=0)
            
            # Standardization
            standardized_params_list = (params_list - means) / stds
            standardized_space = {
                key: ((space[key][0] - means[idx]) / stds[idx], (space[key][1] - means[idx]) / stds[idx])
                for idx, key in enumerate(param_keys)
            }

            # Creating a new optimizer instance in the standardized space
            optimizer_standardized = BayesianOptimization(
                f=func.discreteGP,
                pbounds=standardized_space,
                verbose=2,
                random_state=1
            )

            # Registering standardized parameters
            for params, target in zip(standardized_params_list, optimizer.space.target):
                standardized_params = {key: param for key, param in zip(param_keys, params)}
                optimizer_standardized.register(params=standardized_params, target=target)

            # Suggesting a new candidate point
            next_point_standardized = optimizer_standardized.suggest(utility)

            # Transforming back to the original scale
            next_point_to_probe = {
                key: next_point_standardized[key] * stds[idx] + means[idx]
                for idx, key in enumerate(param_keys)
            }
            next_point_to_probe = func.discretize(**next_point_to_probe)

            # Reapplying standardization
            next_point_standardized_again = {
                key: (next_point_to_probe[idx] - means[idx]) / stds[idx]
                for idx, key in enumerate(param_keys)
            }

            # Use the standardized optimizer to calculate mu
            items_standardized = np.array([next_point_standardized_again[key] for key in param_keys]).reshape(1, -1)
            mu = optimizer_standardized._gp.predict(items_standardized)
            mu = mu.item()

            print('-----------------')
            print('Moving xy-stage')
            
            control_serial1()
            print('Stoping xy-stage')

            print(f'\nSample Number {i+1}')
            print(f'Suggested next point\n'
                f'E: {next_point_to_probe[0]} GPa, Compound: {suggest_E_crystal(next_point_to_probe[0])}, '
                f'L: {next_point_to_probe[1]} μm, W: {next_point_to_probe[4]} μm, '
                f' T: {next_point_to_probe[3]} μm, P: {next_point_to_probe[2]} mW/cm2\n'
                f'Predicted value: {mu}')

            # Save the suggested point
            dummy.register(
                params = next_point_to_probe,
                target = mu,
            )

            # Do REAL experiment near the new point (IMPORTANT!!!)
            print('-----------------')
            print('Please choose a sample close to the suggested point\n')
            print('(Then, please set crystal.)')
            new_exp_point = handinput()
            new_exp_point = {key: num for key, num in zip(optimizer.space.keys, new_exp_point)}
            
            print('Please adjust jig position and measure generarted force')
            print('When you start irradiating, please enter 1 (upper) or 2 (down).')
            num = int(input(">>"))
            uv_light_serial(round(next_point_to_probe[2]/space['P'][1]*100), num=num)

            print('When you start irradiating the opposite face, please enter 1 (upper) or 2 (down).')
            num = int(input(">>"))
            uv_light_serial(round(next_point_to_probe[2]/space['P'][1]*100), num=num)
            
            # Visualization of generated force
            print('Continue? or Remeasure?')
            measure = input('>>')
            if measure == "Continue":
                print("OK! Continue.")
            else:
                print('-----------------')
                print('Please remeasure generarted force')
                print('When you start irradiation, please enter 1 (upper) or 2 (down).')
                num = input(">>")
                uv_light_serial(round(next_point_to_probe[2]/space['P'][1]*100), num = num)
            
            # When you start irradiation, please enter 1 (upper) or 2 (down).
            print('Which side of the irradiation generates a higher force? Please enter 1 (upper) or 2 (down).')
            upper_or_down = int(input(">>"))

            print('Please measure and specify the result file')
            target = getmaxY()

            # Save the experimented point
            optimizer.register(
                params = new_exp_point,
                target = target,
            )

            # Added "upper_or_down" to the log
            with open(f'{results_folder}log_{work_name}_exp.json', 'r+') as f:
                lines = f.readlines()  # Read all rows
                last_line = json.loads(lines[-1])  # Retrieved the last row
                last_line['upper_or_down'] = upper_or_down  # Added "upper_or_down"

                # Reordered to place "upper_or_down" before "datetime"
                ordered_entry = {
                    "target": last_line["target"],
                    "params": last_line["params"],
                    "upper_or_down": last_line["upper_or_down"],
                    "datetime": last_line["datetime"]
                }

                # Deleted the last row and replaced it with the updated entry
                lines[-1] = json.dumps(ordered_entry) + '\n'

                # Rewriting the file from the beginning
                f.seek(0)
                f.writelines(lines)

            # Adjust irradiation intensity
            for i in range(10):
                print("Do you try other light intensity? (Yes or No)")
                answer = input('>>')
                if answer == "No":
                    print("OK!! Continue.")
                    break
                else:
                    print('Type UV intensity (mW/cm2)')
                    P = float(input('>>'))
                    new_exp_point["P"] = int(P)

                    print('When you start irradiation, please enter 1 (upper) or 2 (down).')
                    num=input(">>")
                    uv_light_serial(round(new_exp_point["P"]/space['P'][1]*100), num=num)

                    print('When you start irradiating the opposite face, please enter 1 (upper) or 2 (down).')
                    num = int(input(">>"))
                    uv_light_serial(round(new_exp_point["P"]/space['P'][1]*100), num=num)

                    # When you start irradiation, please enter 1 (upper) or 2 (down).
                    print('Which side of the irradiation generates a higher force? Please enter 1 (upper) or 2 (down).')
                    upper_or_down = int(input(">>"))

                    print('Please measure and specify the result file')
                    target = getmaxY()

                    # Save the experimented point
                    optimizer.register(
                        params = new_exp_point,
                        target = target,
                    )

                    # Added "upper_or_down" to the log
                    with open(f'{results_folder}log_{work_name}_exp.json', 'r+') as f:
                        lines = f.readlines()  # Read all rows
                        last_line = json.loads(lines[-1])  # Retrieved the last row
                        last_line['upper_or_down'] = upper_or_down  # Added "upper_or_down"

                        # Reordered to place "upper_or_down" before "datetime"
                        ordered_entry = {
                            "target": last_line["target"],
                            "params": last_line["params"],
                            "upper_or_down": last_line["upper_or_down"],
                            "datetime": last_line["datetime"]
                        }

                        # Deleted the last row and replaced it with the updated entry
                        lines[-1] = json.dumps(ordered_entry) + '\n'

                        # Rewriting the file from the beginning
                        f.seek(0)
                        f.writelines(lines)

            print('Moving xy-stage')
            control_serial2()
            print('Stoping xy-stage')

            print("Experiment Continue? (Yes or No)")
            answer2 = input(">>")
            if answer2 == "No":
                break
            ### Breaking the optimization loop ###
    else:
        print('Choose valid mode!')
    
    # Finishing
    total = time.time() - start_time
    minute = total//60
    second = total%60
    print(f'--- {int(minute)} min {int(second)} sec elapsed ---')

if __name__ == '__main__':
    params = {
        'mode': settings6.mode,
        'work_name': settings6.work_name,
        'init_file': settings6.init_file,
        'n_trial': settings6.n_trial,
        'n_trial_P': settings6.n_trial_P,
        'random_state': settings6.random_state,
        'acquisition': settings6.acquisition,
        'results_folder': settings6.results_folder,
        'space': settings6.space,
        'E_exp': settings6.E_exp,
        'kappa': settings6.kappa,
        'xi': settings6.xi,
        'kernel': settings6.kernel,
        'length_scale': settings6.length_scale,
        'nu': settings6.nu,
        'repeat': settings6.repeat
    }
    main(**params)
