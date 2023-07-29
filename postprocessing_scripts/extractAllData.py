##### Author      : Francesco Massimo (Laboratoire de Physique des Gaz et des Plasmas, CNRS)
##### Purpose     : extract and visualize all data from all runs in the current working directory
##### Last update : 29/07/2023

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import seaborn as sns

starting_directory = os.getcwd()


runs_to_read = []
for file in os.listdir(starting_directory):
    directory_name = file
    if os.path.isdir(directory_name):
        runs_to_read.append(directory_name)

##### assuming that all runs have the same shape, find number of iterations and dimensions of first one
filename = "history_particles_positions_and_function_values.npy"

#filename = "history_particles_positions_and_function_values_iteration_00028.npy"

history_particles_positions_and_function_values = np.load(runs_to_read[0]+"/"+filename)
number_of_iterations,num_samples,num_dimensions = np.shape(history_particles_positions_and_function_values)
num_dimensions = num_dimensions-1
iterations = np.linspace(0,number_of_iterations-1,num=number_of_iterations)

data_optimizations = np.zeros(shape=(number_of_iterations*len(runs_to_read),num_samples,num_dimensions+1))


index_run = 0
for run in runs_to_read:
    history_particles_positions_and_function_values = np.load(run+"/"+filename)
    first_run_index = (index_run)*number_of_iterations
    second_run_index = (index_run)*number_of_iterations+number_of_iterations
    data_optimizations[first_run_index:second_run_index,:,:] = history_particles_positions_and_function_values
    index_run = index_run+1


np.save("data_optimizations.npy",data_optimizations)

### Transform into pandas dataframe
data_optimizations_reshaped = data_optimizations.reshape((len(runs_to_read)*number_of_iterations*num_samples,num_dimensions+1))
#labels = ["delay_behind_laser_micron","bunch_charge_pC","a0","plasma_plateau_density_1_ov_cm3","function_to_optimize"]
labels = ['x_focus_um', 'dopant_N_concentration', 'a0', 'plasma_plateau_density_1_ov_cm3','function_to_optimize']
df = pd.DataFrame(data_optimizations_reshaped, columns = labels)

plt.ion()

new_df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
new_df[new_df["function_to_optimize"] < 0] = 0.1
sns.pairplot(new_df,kind="scatter",height=1,plot_kws=dict(hue=df["function_to_optimize"],palette="blend:gold,dodgerblue",s=5))



# plt.figure()
# df["delay_behind_laser_micron"].plot(kind="hist", weights=df["function_to_optimize"],density="True",bins=15,histtype='step')
# plt.xlabel("delay_behind_laser_micron")
# 
# plt.figure()
# df["bunch_charge_pC"].plot(kind="hist", weights=df["function_to_optimize"],density="True",bins=15,histtype='step')
# plt.xlabel("bunch_charge_pC")
# 
# plt.figure()
# df["a0"].plot(kind="hist", weights=df["function_to_optimize"],density="True",bins=15,histtype='step')
# plt.xlabel("a0")
# 
# plt.figure()
# df["plasma_plateau_density_1_ov_cm3"].plot(kind="hist", weights=df["function_to_optimize"],density="True",bins=15,histtype='step')
# plt.xlabel("plasma_plateau_density_1_ov_cm3")

