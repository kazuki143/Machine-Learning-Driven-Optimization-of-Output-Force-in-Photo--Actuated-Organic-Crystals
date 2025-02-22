# Machine Learning-Driven Optimization of Output Force in Photo-Actuated Organic Crystals

These files contain the machine learning scripts used in "Machine Learning-Driven Optimization of Output Force in Photo-Actuated Organic Crystals". 
I have uploaded files used for LASSO regression and Bayesian optimization. 

## LASSO_regression Folder
`LASSO_regression.ipynb`: File for data loading, pre-processing and LASSO regression

`Young's_modulus_dataset.xlsx`: Dataset for LASSO regression
(This file is the same as "Supplementary Data.xlsx" in "Machine Learning-Driven Optimization of Output Force in Photo-Actuated Organic Crystals".)

## Bayesian_optimization Folder
`main_bayesianoptimization.py`: The main script for performing Bayesian optimization

`settings6.py`: Input the setting

`utils7.py`: Contains functions used for Bayesian optimization

`initial_dataset.xlsx`: Initial data for Bayesian optimization

`./automation/stage_control.py`: Controls the movement of the stage holding the slide glass with a fixed crystal

`./automation/uv_light_control.py`: Controls the UV light irradiation device

> ### Execution command
To perform Bayesian optimization, simply run the following command:

`python main_bayesianoptimization.py`

The command flow for Bayesian optimization is summarized in the figure below. 
![image](https://github.com/user-attachments/assets/1282f18e-2c1c-4efb-a1d8-63c1281f3459)

Additionally, you can watch a recorded video of the actual execution of Bayesian optimization at the following video.
[![image](https://github.com/user-attachments/assets/1b9e5d14-bd44-463f-b919-6564d23ef9a9)](https://youtu.be/iQlV0FbhcLQ)

> ### Output Folder
`./results/`: After executing main_bayesianoptimization.py, the results are saved in this directory as .json files.
