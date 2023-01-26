# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Transport of regression-aware autoencoder-derived manifold parameters
#
# For a zero-dimensional reactor testcase
#
# This script saves all the results to .csv files
#
# For plotting and visualizing the results, use the associated Jupyter notebook
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import time
import pandas as pd
import george
from scipy.integrate import odeint
from scipy.interpolate import RBFInterpolator

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, InputLayer
from keras import optimizers
from keras import metrics
from keras import losses
from keras import layers

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold.styles import *

total_script_tic = time.perf_counter()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# User settings
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

data_tag = 'CO-H2-10-1-isobaric-adiabatic-closed-HR'
data_path = '../data/'
results_path = '../results/'
species_to_remove_list = ['N2', 'AR', 'HE']
sample_percentage = 100
random_seed = 100
n_components = 2
scaling = '0to1'
run_RBF = True
run_GPR = True
run_ANN = True
run_KReg = True
reconstruct_thermochemistry = True
start_simulation = 120
n_points = 2000
max_simulation_time = 0.005

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load training data
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

state_space_names = pd.read_csv(data_path + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()
state_space = pd.read_csv(data_path + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
state_space_sources = pd.read_csv(data_path + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
mixture_fraction = pd.read_csv(data_path + data_tag + '-mixture-fraction.csv', sep = ',', header=None).to_numpy()
time_grid = pd.read_csv(data_path + data_tag + '-time.csv', sep = ',', header=None).to_numpy().ravel()

state_space_test_trajectory = pd.read_csv(data_path + data_tag + '-state-space-test-trajectory.csv', sep = ',', header=None).to_numpy()
state_space_sources_test_trajectory = pd.read_csv(data_path + data_tag + '-state-space-sources-test-trajectory.csv', sep = ',', header=None).to_numpy()
time_grid_test_trajectory = pd.read_csv(data_path + data_tag + '-time-test-trajectory.csv', sep = ',', header=None).to_numpy().ravel()

state_space_support_trajectories = pd.read_csv(data_path + data_tag + '-state-space-support-trajectories.csv', sep = ',', header=None).to_numpy()
state_space_sources_support_trajectories = pd.read_csv(data_path + data_tag + '-state-space-sources-support-trajectories.csv', sep = ',', header=None).to_numpy()
time_grid_support_trajectories = pd.read_csv(data_path + data_tag + '-time-support-trajectories.csv', sep = ',', header=None).to_numpy().ravel()

for species_to_remove in species_to_remove_list:

    (species_index, ) = np.where(state_space_names==species_to_remove)
    if len(species_index) != 0:
        print('Removing ' + state_space_names[int(species_index)] + '.')
        state_space = np.delete(state_space, np.s_[species_index], axis=1)
        state_space_sources = np.delete(state_space_sources, np.s_[species_index], axis=1)

        state_space_test_trajectory = np.delete(state_space_test_trajectory, np.s_[species_index], axis=1)
        state_space_sources_test_trajectory = np.delete(state_space_sources_test_trajectory, np.s_[species_index], axis=1)

        state_space_support_trajectories = np.delete(state_space_support_trajectories, np.s_[species_index], axis=1)
        state_space_sources_support_trajectories = np.delete(state_space_sources_support_trajectories, np.s_[species_index], axis=1)

        state_space_names = np.delete(state_space_names, np.s_[species_index])
    else:
        print(species_to_remove + ' already removed from the data set.')

(n_observations, n_variables) = np.shape(state_space)

if sample_percentage == 100:
    sample_data = False
else:
    sample_data = True

if sample_data:
    idx = np.zeros((n_observations,)).astype(int)
    sample_random = preprocess.DataSampler(idx, random_seed=random_seed, verbose=False)
    (idx_sample, _) = sample_random.random(sample_percentage)

    state_space = state_space[idx_sample,:]
    state_space_sources = state_space_sources[idx_sample,:]
    mf = mf[idx_sample,:]

    (n_observations, n_variables) = np.shape(state_space)

print('\nThe data set has ' + str(n_observations) + ' observations.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate low-dimensional manifold parameters
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

AE_basis = np.array([[ 0.44541207, -0.53351498],
       [ 0.21240635,  0.03709494],
       [ 0.23040931,  0.0683625 ],
       [ 0.04429741,  0.27895769],
       [ 0.17797215,  0.02717828],
       [-0.77823412, -0.22460838],
       [ 0.01838836,  0.13045858],
       [-0.06177867, -0.02877426],
       [-0.22748888,  0.42358291],
       [ 0.02015193, -0.61245435],
       [-0.08835387, -0.08909094]])

(X_CS, centers, scales) = preprocess.center_scale(state_space, scaling=scaling)
S_CS = state_space_sources / scales

state_space_test_trajectory_CS = (state_space_test_trajectory - centers)/scales
state_space_sources_test_trajectory_CS = state_space_sources_test_trajectory / scales

state_space_support_trajectories_CS = (state_space_support_trajectories - centers)/scales
state_space_sources_support_trajectories_CS = state_space_sources_support_trajectories / scales

X_AE = np.dot(X_CS, AE_basis)
S_AE = np.dot(S_CS, AE_basis)

X_AE_CS, center_AE, scale_AE = preprocess.center_scale(X_AE, scaling='-1to1')

AE_state_space_test_trajectory = np.dot(state_space_test_trajectory_CS, AE_basis)
AE_state_space_test_trajectory_CS = (AE_state_space_test_trajectory - center_AE) / scale_AE
AE_source_terms_test_trajectory = np.dot(state_space_sources_test_trajectory_CS, AE_basis)

AE_state_space_support_trajectories = np.dot(state_space_support_trajectories_CS, AE_basis)
AE_state_space_support_trajectories_CS = (AE_state_space_support_trajectories - center_AE) / scale_AE
AE_source_terms_support_trajectories = np.dot(state_space_sources_support_trajectories_CS, AE_basis)

AE_source_terms_support_trajectories_CS, centers_AE_source_terms, scales_AE_source_terms = preprocess.center_scale(AE_source_terms_support_trajectories, scaling='-1to1')

(n_observations, _) = AE_state_space_support_trajectories.shape

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Build nonlinear regression closure models
#
# We benchmark four different techniques: RBF, GPR, ANN, and kernel regression
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# RBF - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_RBF:

    print('Training RBF closure model...')

    AE_source_terms_test_trajectory_CS = (AE_source_terms_test_trajectory - centers_AE_source_terms) / scales_AE_source_terms

    tic = time.perf_counter()

    rbf_model = RBFInterpolator(AE_state_space_support_trajectories_CS, AE_source_terms_support_trajectories_CS, epsilon=20, kernel='linear', )

    toc = time.perf_counter()

    print(f'\tRBF training time: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    AE_source_terms_predicted_rbf = rbf_model(AE_state_space_test_trajectory_CS)
    AE_source_terms_predicted_rbf = preprocess.invert_center_scale(AE_source_terms_predicted_rbf, centers_AE_source_terms, scales_AE_source_terms)

    np.savetxt(results_path + 'best-AE-RBF-predicted-source-terms.csv', (AE_source_terms_predicted_rbf), delimiter=',', fmt='%.16e')

# GPR - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_GPR:

    print('Training GPR closure model...')

    sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=100, verbose=False)
    (idx_train_gpr, idx_test_gpr) = sample_random.random(40)

    kernel = george.kernels.ExpSquaredKernel(0.01, ndim=n_components)

    tic = time.perf_counter()

    gpr_model = george.GP(kernel)
    gpr_model.compute(AE_state_space_support_trajectories_CS[idx_train_gpr,:], yerr=1.25e-12,)

    toc = time.perf_counter()

    print(f'\tGPR training time: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    AE_source_terms_predicted_gpr = np.zeros_like(AE_source_terms_test_trajectory)
    AE_source_terms_predicted_gpr[:,0], _ = gpr_model.predict(AE_source_terms_support_trajectories_CS[idx_train_gpr,0], AE_state_space_test_trajectory_CS, return_var=False)
    AE_source_terms_predicted_gpr[:,1], _ = gpr_model.predict(AE_source_terms_support_trajectories_CS[idx_train_gpr,1], AE_state_space_test_trajectory_CS, return_var=False)
    AE_source_terms_predicted_gpr = preprocess.invert_center_scale(AE_source_terms_predicted_gpr, centers_AE_source_terms, scales_AE_source_terms)

    np.savetxt(results_path + 'best-AE-GPR-predicted-source-terms.csv', (AE_source_terms_predicted_gpr), delimiter=',', fmt='%.16e')

# ANN - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_ANN:

    print('Training ANN closure model...')

    sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=100, verbose=False)
    (idx_train, idx_test) = sample_random.random(80)

    tic = time.perf_counter()

    tf.random.set_seed(random_seed)

    ann_model = Sequential([
    Dense(5, input_dim=n_components, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(10, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(15, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(10, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(5, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(n_components, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros')
    ])

    ann_model.compile(tf.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])

    monitor = EarlyStopping(monitor='loss',
                        min_delta=1e-6,
                        patience=100,
                        verbose=0,
                        mode='auto',
                        restore_best_weights=True)

    history_model = ann_model.fit(AE_state_space_support_trajectories_CS[idx_train,:], AE_source_terms_support_trajectories_CS[idx_train,:],
                epochs=2000,
                batch_size=100,
                shuffle=True,
                validation_data=(AE_state_space_support_trajectories_CS[idx_test,:], AE_source_terms_support_trajectories_CS[idx_test,:]),
                callbacks=[monitor],
                verbose=0)

    toc = time.perf_counter()

    print(f'\tANN training time: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    AE_source_terms_predicted_ann = ann_model.predict(AE_state_space_test_trajectory_CS)
    AE_source_terms_predicted_ann = preprocess.invert_center_scale(AE_source_terms_predicted_ann, centers_AE_source_terms, scales_AE_source_terms)

    np.savetxt(results_path + 'best-AE-ANN-predicted-source-terms.csv', (AE_source_terms_predicted_ann), delimiter=',', fmt='%.16e')

# Kernel regression - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_KReg:

    print('Training kernel regression closure model...')

    tic = time.perf_counter()

    kreg_model = analysis.KReg(AE_state_space_support_trajectories_CS, AE_source_terms_support_trajectories_CS)

    toc = time.perf_counter()

    print(f'\tKernel regression training time: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    AE_source_terms_predicted_kreg = kreg_model.predict(AE_state_space_test_trajectory_CS, bandwidth=0.0001)
    AE_source_terms_predicted_kreg = preprocess.invert_center_scale(AE_source_terms_predicted_kreg, centers_AE_source_terms, scales_AE_source_terms)

    np.savetxt(results_path + 'best-AE-KReg-predicted-source-terms.csv', (AE_source_terms_predicted_kreg), delimiter=',', fmt='%.16e')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Evolve the reduced-order model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

AE_initial_condition = AE_state_space_test_trajectory[start_simulation,:]
t_coordinates = np.linspace(time_grid_test_trajectory[start_simulation],max_simulation_time,n_points)

# RBF - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_RBF:

    print('Running ROM with RBF closure model...')

    def AE_source_terms_rbf_model(query):

        query_CS = (query - center_AE) / scale_AE

        predicted = rbf_model(query_CS)

        predicted = preprocess.invert_center_scale(predicted, centers_AE_source_terms, scales_AE_source_terms)

        return predicted

    def RHS_ODE(X, time_vector):

        query_list = []
        for i in range(0,n_components):
            query_list.append(X[i])
        query = np.array([query_list])

        dZdt_list = []
        for i in range(0,n_components):
            dZidt = AE_source_terms_rbf_model(query)[:,i]
            dZdt_list.append(dZidt)
        dZdt = np.array([dZdt_list])

        return dZdt.ravel()

    tic = time.perf_counter()
    numerical_solution_rbf_model = odeint(RHS_ODE, AE_initial_condition.ravel(), t_coordinates)
    toc = time.perf_counter()
    print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    np.savetxt(results_path + 'best-AE-RBF-numerical-solution.csv', (numerical_solution_rbf_model), delimiter=',', fmt='%.16e')

    AE_source_terms_from_numerical_solution = AE_source_terms_rbf_model(numerical_solution_rbf_model)

    np.savetxt(results_path + 'best-AE-RBF-AE-source-terms-from-numerical-solution.csv', (AE_source_terms_from_numerical_solution), delimiter=',', fmt='%.16e')

# GPR - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_GPR:

    print('Running ROM with GPR closure model...')

    def AE_source_terms_gpr_model(query):

        query_CS = (query - center_AE) / scale_AE

        predicted = np.zeros_like(query)
        predicted[:,0], _ = gpr_model.predict(AE_source_terms_support_trajectories_CS[idx_train_gpr,0], query_CS, return_var=False)
        predicted[:,1], _ = gpr_model.predict(AE_source_terms_support_trajectories_CS[idx_train_gpr,1], query_CS, return_var=False)
        predicted = preprocess.invert_center_scale(predicted, centers_AE_source_terms, scales_AE_source_terms)

        return predicted

    def RHS_ODE(X, time_vector):

        query_list = []
        for i in range(0,n_components):
            query_list.append(X[i])
        query = np.array([query_list])

        dZdt_list = []
        for i in range(0,n_components):
            dZidt = AE_source_terms_gpr_model(query)[:,i]
            dZdt_list.append(dZidt)
        dZdt = np.array([dZdt_list])

        return dZdt.ravel()

    tic = time.perf_counter()
    numerical_solution_gpr_model = odeint(RHS_ODE, AE_initial_condition.ravel(), t_coordinates)
    toc = time.perf_counter()
    print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    np.savetxt(results_path + 'best-AE-GPR-numerical-solution.csv', (numerical_solution_gpr_model), delimiter=',', fmt='%.16e')

    AE_source_terms_from_numerical_solution = AE_source_terms_gpr_model(numerical_solution_gpr_model)

    np.savetxt(results_path + 'best-AE-GPR-AE-source-terms-from-numerical-solution.csv', (AE_source_terms_from_numerical_solution), delimiter=',', fmt='%.16e')

# ANN - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_ANN:

    print('Running ROM with ANN closure model...')

    def AE_source_terms_ann_model(query):

        query_CS = (query - center_AE) / scale_AE

        predicted = ann_model.predict(query_CS)

        predicted = preprocess.invert_center_scale(predicted, centers_AE_source_terms, scales_AE_source_terms)

        return predicted

    def RHS_ODE(X, time_vector):

        query_list = []
        for i in range(0,n_components):
            query_list.append(X[i])
        query = np.array([query_list])

        dZdt_list = []
        for i in range(0,n_components):
            dZidt = AE_source_terms_ann_model(query)[:,i]
            dZdt_list.append(dZidt)
        dZdt = np.array([dZdt_list])

        return dZdt.ravel()

    tic = time.perf_counter()
    numerical_solution_ann_model = odeint(RHS_ODE, AE_initial_condition.ravel(), t_coordinates)
    toc = time.perf_counter()
    print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    np.savetxt(results_path + 'best-AE-ANN-numerical-solution.csv', (numerical_solution_ann_model), delimiter=',', fmt='%.16e')

    AE_source_terms_from_numerical_solution = AE_source_terms_ann_model(numerical_solution_ann_model)

    np.savetxt(results_path + 'best-AE-ANN-AE-source-terms-from-numerical-solution.csv', (AE_source_terms_from_numerical_solution), delimiter=',', fmt='%.16e')

# Kernel regression - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if run_KReg:

    print('Running ROM with kernel regression closure model...')

    def AE_source_terms_kreg_model(query):

        query_CS = (query - center_AE) / scale_AE

        predicted = kreg_model.predict(query_CS, bandwidth=0.001)

        predicted = preprocess.invert_center_scale(predicted, centers_AE_source_terms, scales_AE_source_terms)

        return predicted

    def RHS_ODE(X, time_vector):

        query_list = []
        for i in range(0,n_components):
            query_list.append(X[i])
        query = np.array([query_list])

        dZdt_list = []
        for i in range(0,n_components):
            dZidt = AE_source_terms_kreg_model(query)[:,i]
            dZdt_list.append(dZidt)
        dZdt = np.array([dZdt_list])

        return dZdt.ravel()

    tic = time.perf_counter()
    numerical_solution_kreg_model = odeint(RHS_ODE, AE_initial_condition.ravel(), t_coordinates)
    toc = time.perf_counter()
    print(f'\tTime it took: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    np.savetxt(results_path + 'best-AE-KReg-numerical-solution.csv', (numerical_solution_kreg_model), delimiter=',', fmt='%.16e')

    AE_source_terms_from_numerical_solution = AE_source_terms_kreg_model(numerical_solution_kreg_model)

    np.savetxt(results_path + 'best-AE-KReg-AE-source-terms-from-numerical-solution.csv', (AE_source_terms_from_numerical_solution), delimiter=',', fmt='%.16e')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruct the thermo-chemistry from the evolved manifold parameters
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if reconstruct_thermochemistry:

    selected_state_variables = [0,1,2,3,4,5,6,7,8,9,10]
    state_space_CS, state_space_centers, state_space_scales = preprocess.center_scale(state_space_support_trajectories[:,selected_state_variables], scaling='0to1')

    tic = time.perf_counter()

    sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=100, verbose=False)
    (idx_train, idx_test) = sample_random.random(80)

    tf.random.set_seed(random_seed)

    ann_model_state = Sequential([
    Dense(5, input_dim=n_components, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(10, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(15, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(20, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(15, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    Dense(len(selected_state_variables), activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')
    ])

    ann_model_state.compile(tf.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])

    monitor = EarlyStopping(monitor='loss',
                        min_delta=1e-6,
                        patience=100,
                        verbose=0,
                        mode='auto',
                        restore_best_weights=True)

    history = ann_model_state.fit(AE_state_space_support_trajectories_CS[idx_train,:], state_space_CS[idx_train,:],
                epochs=2000,
                batch_size=100,
                shuffle=True,
                validation_data=(AE_state_space_support_trajectories_CS[idx_test,:], state_space_CS[idx_test,:]),
                callbacks=[monitor],
                verbose=0)

    toc = time.perf_counter()

    print(f'\tANN training time: {(toc - tic)/60:0.1f} minutes.\n' + '-'*40)

    state_space_predicted = ann_model_state.predict(AE_state_space_support_trajectories_CS)
    state_space_predicted = preprocess.invert_center_scale(state_space_predicted, state_space_centers, state_space_scales)

    if run_RBF:

        numerical_solution_rbf_model_CS = (numerical_solution_rbf_model - center_AE) / scale_AE
        state_space_ROM_predicted_rbf = ann_model_state.predict(numerical_solution_rbf_model_CS)
        state_space_ROM_predicted_rbf = preprocess.invert_center_scale(state_space_ROM_predicted_rbf, state_space_centers, state_space_scales)
        np.savetxt(results_path + 'best-AE-RBF-predicted-state-space.csv', (state_space_ROM_predicted_rbf), delimiter=',', fmt='%.16e')

    if run_GPR:

        numerical_solution_gpr_model_CS = (numerical_solution_gpr_model - center_AE) / scale_AE
        state_space_ROM_predicted_gpr = ann_model_state.predict(numerical_solution_gpr_model_CS)
        state_space_ROM_predicted_gpr = preprocess.invert_center_scale(state_space_ROM_predicted_gpr, state_space_centers, state_space_scales)
        np.savetxt(results_path + 'best-AE-GPR-predicted-state-space.csv', (state_space_ROM_predicted_gpr), delimiter=',', fmt='%.16e')

    if run_ANN:

        numerical_solution_ann_model_CS = (numerical_solution_ann_model - center_AE) / scale_AE
        state_space_ROM_predicted_ann = ann_model_state.predict(numerical_solution_ann_model_CS)
        state_space_ROM_predicted_ann = preprocess.invert_center_scale(state_space_ROM_predicted_ann, state_space_centers, state_space_scales)
        np.savetxt(results_path + 'best-AE-ANN-predicted-state-space.csv', (state_space_ROM_predicted_ann), delimiter=',', fmt='%.16e')

    if run_KReg:

        numerical_solution_kreg_model_CS = (numerical_solution_kreg_model - center_AE) / scale_AE
        state_space_ROM_predicted_kreg = ann_model_state.predict(numerical_solution_kreg_model_CS)
        state_space_ROM_predicted_kreg = preprocess.invert_center_scale(state_space_ROM_predicted_kreg, state_space_centers, state_space_scales)
        np.savetxt(results_path + 'best-AE-KReg-predicted-state-space.csv', (state_space_ROM_predicted_kreg), delimiter=',', fmt='%.16e')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

total_script_toc = time.perf_counter()

print(f'\tTotal time it took to run this script: {(total_script_toc - total_script_tic)/60:0.1f} minutes.\n' + '-'*40)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
