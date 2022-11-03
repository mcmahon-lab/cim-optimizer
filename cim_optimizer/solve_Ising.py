# Reference CIM File
# Last Updated: September 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/

from cmath import inf
import matplotlib.pyplot as plt
import signal
import csv
import copy
import os
import sys
import glob
import time
import torch
import numpy as np
from bohb import BOHB
import bohb.configspace as cs
from dataclasses import dataclass
import warnings
import argparse
from cim_optimizer import AHC, AHC_tuning, extAHC, extAHC_tuning, CAC, CAC_tuning, CIM_helper

NoneType = type(None)

# Helper Functions

def rsm(A):
    # Calculate root square sum of array A.
    s = 0.0
    for a in A:
        s += a**2
    return np.sqrt(s)


def get_binary(A):
    # Get binary form of an array.
    v = 2 * (np.real(A) > 0) - 1
    return v


def rms(A):
    s = 0.0
    for a in A:
        s += a**2
    return np.sqrt(s / (A.shape[0]))


def load_adjMatrix_from_rudy(filepathstr, delimiter='\t', index_start=0, preset_size=1):
    return CIM_helper.load_adjMatrix_from_rudy(filepathstr, delimiter=delimiter, index_start=index_start, preset_size=preset_size)


class Ising:
    r""" Class for the Ising problem, including nested CIM solver classes.

    The Ising problem is specified to reduce the following Hamiltonian:
    
    .. math::
        
        \text{argmin}_{\vec{s}} \left(-(\sum_{1\leq i<j<N} J_{ij} s_i s_j) - \sum_i(h_i s_i) \right)

    Attributes
    ----------
    J : ndarray
        Ising spin-spin coupling matrix.
    h : ndarray
        External-field vector (individual weights).

    Examples
    ----------
    >>> J = np.zeros((5,5))
    >>> h = np.zeros(5)
    >>> problem = Ising(J,h)

    The Ising class uses a nested solver class for invoking CIM solvers.

    >>> problem.solve()
    Ising.solver(problem=<__main__.Ising object at 0x10a478040>, .... )
    >>> Ising(J,h).solve(target_energy = 0)
    Ising.solver(problem=<__main__.Ising object at 0x10a478040>, .... )
    """

    def __init__(
        self,
        J: np.ndarray,  # J spin-spin coupling matrix
        h: np.ndarray = None   # h external-field vector
    ) -> None:

        if h is None:
            h = np.zeros(J.shape[0])
        # Assertion statements:
        assert J.ndim == 2, "J_coupling must be a 2-dimensional array!"
        assert h.ndim == 1, "h_external must be a 1-dimensional vector!"
        assert J.shape[0] == J.shape[1], \
            "J must be a square matrix!"
        assert np.array_equal(J, J.transpose()), \
            "J must be a symmetric matrix!"
        assert J.shape[0] == h.shape[0], \
            "J and h must have an equal number of spins!"
        # Initialize Ising problem:
        self.J = J
        self.h = h
        # Bind solve attribute to nested solver class:
        self.solve = lambda **kw: Ising.solver(self, **kw)

    def __str__(self) -> str:
        """Represent the Ising class object as a string."""
        return f'IsingProblem({self.J},{self.h})'

    class Result(dict):
        """ Represents the result of a CIM run.

        Attributes
        ----------
        lowest_energy_spin_config : ndarray
            Ising spin configuration found that had the lowest energy across all
            runs and all timesteps.
        spin_config_all_runs : ndarray
            Spin configurations of the configurations achieving the lowest
            energies in each run.
        spin_trajectories : ndarray
            Array of spin configuration for each step in the CIM and each run.
        lowest_energy : int
            Lowest Ising energy found across all runs.
        energies : ndarray
            Lowest Ising energy met for each run.
        energy_evolution : ndarray
            Evolution of Ising energy for each timestep in the CIM and each run.
        reached_target_energy: ndarray
            Array of booleans for whether the target Ising energy was reached
            for each run.
        num_timesteps_completed : ndarray
            Roundtrip at which the CIM has stopped; either after solving the
            problem to the target energy or reaching a maximum execution time,
            for each run.
        time : ndarray
            Time elapsed for each CIM run, measured using wall-clock time.
        """

        def plot_spin_trajectories(self, plot_type="spins", trajectories_to_plot: list = []) -> None:
            """Plot spin trajectories over time for a given run or all runs."""

            # if trajectories_to_plot is not defined, trajectories for all runs are plotted
            input_data = self.result_data
            trajectory_data = input_data[2]
            num_runs = len(input_data[2])
            if len(trajectories_to_plot) == 0:
                num_runs_to_plot = num_runs
                trajectories_to_plot = np.arange(num_runs_to_plot)
            else:
                num_runs_to_plot = len(trajectories_to_plot)
            num_timesteps_per_run = ((trajectory_data[0]).shape)[1]
            N = ((trajectory_data[0]).shape)[0]
            # Spin trajectory plot
            if plot_type == "spins":
                axs = tuple([] for _ in range(num_runs_to_plot))
                fig, axs = plt.subplots(
                    num_runs_to_plot, 1, figsize=(10, 2*num_runs_to_plot), sharex=True)
                fig.text(0.55, -0.05, 'Time t', ha='center', va='center')
                fig.text(0.05, 0.5, 'Spin Amplitude', ha='center',
                         va='center', rotation='vertical')
                colors = plt.cm.PuOr(1-np.linspace(0, 1, N))
                if num_runs_to_plot == 1:
                    axs = (axs,)
                for z in range(num_runs_to_plot):
                    for x in range(N):
                        (axs[z]).plot(range(num_timesteps_per_run), (trajectory_data[trajectories_to_plot[z]])[x, range(
                            num_timesteps_per_run)], label=f"Spin {x+1}", color=colors[x], alpha=(0.99))

            # Energy plot
            if plot_type == "energy":
                energy_axs = tuple([] for _ in range(num_runs_to_plot))
                fig, energy_axs = plt.subplots(
                    num_runs_to_plot, 1, figsize=(10, 2*num_runs_to_plot), sharex=True)
                fig.text(0.55, -0.05, 'Time t', ha='center', va='center')
                fig.text(0.05, 0.5, 'Ising Energy', ha='center',
                         va='center', rotation='vertical')
                colors = plt.cm.viridis(np.linspace(0, 1, num_runs_to_plot))
                if num_runs_to_plot == 1:
                    energy_axs = (energy_axs,)
                for z in range(num_runs_to_plot):
                    (energy_axs[z]).plot(range(num_timesteps_per_run), ((input_data[3])[trajectories_to_plot[z]]), label=f"Run {z}", color=colors[z], alpha=(0.8))
            return None

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as e:
                raise AttributeError(key) from e

        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __dir__(self):
            return list(self.keys())

    @dataclass
    class solver:
        """ Class for CIM solver parameters and tools.

        Runs one of three Coherent Ising Machine simulations with amplitude
        control for problems with or without external fields.

        Attributes
        -------------
        problem : __main__.Ising object
            Ising object corresponding to the given problem. This attribute is
            manually set by when using the nested class from an Ising object:
            >>> Ising(J, h).solve(<args>)
        target_energy : int, optional
            Assumed target/ground energy for the solver to reach, used to stop
            before num_runs runs have been completed.
        num_runs : int, default=100
            Maximum number of runs to attempt by the CIM, either running the
            set repeated number of repetitions or stopping if the target energy
            is met.
        num_timesteps_per_run : int, default=1000
            Roundtrip number per run, representing the number of MVM's per run.
        max_wallclock_time_allowed : int, default=10000000
            Seconds passed by the CIM before quitting if num_runs or
            target_energy have not been met.
        stop_when_target_energy_reached : bool, default=True
            Stop if target energy reached before completing all the CIM runs.
        amplitude_control_scheme, optional
            Choice of amplitude control scheme to use, uses CAC for problems
            without external fields and AHC/CAC for problems with external
            fields by default.
        custom_feedback_schedule, optional
            Option to specify a custom function or array of length
            num_timesteps_per_run to use as a feedback schedule.
        custom_pump_schedule, optional
            Option to specify a custom function or array of
            length num_timesteps_per_run to use as a pump schedule.
        hyperparameters_autotune : bool, default=True
            If True then: Based on max_wallclock_time_allowed and num_runs,
            dedicate a reasonable amount of time to finding the best
            hyperparameters to use with the CIM.
        ahc_noext_time_step : float, default=0.05
            Time step for each iteration, based on Eq(X) of citation above.
        ahc_noext_r : float, default=0.2
            AHC hyperparameter
        ahc_noext_beta : float, default=0.05
            AHC hyperparameter
        ahc_noext_eps : float, default=0.07
            AHC hyperparameter
        ahc_noext_mu : float, default=1
            AHC hyperparameter
        ahc_noext_noise : float, default=0
            AHC hyperparameter
        ahc_nonlinearity, optional
            AHC hyperparameter
        cac_time_step : float, default=0.05
            Time step for each iteration, based on Eq(X) of citation above.
        cac_r : float, default=-4.04
            CAC hyperparameter
        cac_alpha : float, default=3.0
            CAC hyperparameter
        cac_beta : float, default=0.25
            CAC hyperparameter
        cac_gamma : float, default=0.00011
            CAC hyperparameter
        cac_delta : float, default=10
            CAC hyperparameter
        cac_mu : float, default=1
            CAC hyperparameter
        cac_rho : float, default=3
            CAC hyperparameter
        cac_tau : float, default=1000
            CAC hyperparameter
        cac_noise : float, default=0
            CAC hyperparameter
        cac_nonlinearity, default=np.tanh
            CAC hyperparameter
        ahc_ext_time_stop : int, default=9000
            Roundtrip number per run, representing time horizon.
        ahc_ext_nsub : int, default=1
            extAHC hyperparameter
        ahc_ext_alpha : float, default=1.0
            extAHC hyperparameter
        ahc_ext_delta : float, default=0.25
            extAHC hyperparameter
        ahc_ext_eps : float, default=0.333
            extAHC hyperparameter
        ahc_ext_lambd : float, default=0.001
            extAHC hyperparameter
        ahc_ext_pi : float, default=-0.225
            extAHC hyperparameter
        ahc_ext_rho : float, default=1.0
            extAHC hyperparameter
        ahc_ext_tau : float, default=100
            extAHC hyperparameter
        ahc_ext_F_h : float, default=2.0
            extAHC hyperparameter
        ahc_ext_noise : float, default=0
            extAHC hyperparameter
        ahc_ext_nonlinearity, default=torch.tanh
            extAHC hyperparameter
        return_lowest_energies_found_spin_configuration : bool, default=False
            Return a vector where for each run, it gives the spin configuration
            that was found during that run that had the lowest energy.
        return_lowest_energy_found_from_each_run : bool, default=True
            Return a vector with the lowest energy found for each run.
        return_spin_trajectories_all_runs : bool, default=True
            Return the Ising spin trajectory for every run.
        return_number_of_solutions : int, default=1000
            Number of best solutions to return; must be <= num_runs.
        suppress_statements : bool, default=False
            Print details of each solver call to screen/REPL.
        use_GPU : bool, default=False
            Option to use GPU acceleration using PyTroch libraries.
        use_CAC : bool, default=True
            Option to select CAC or AHC solver for no external field.
        chosen_device : torch.device, default=torch.device('cpu')
            Device used for torch-based computations.
            
        Returns
        ----------
        Result
            Returns Ising.Result dictionary objects with run/solution details.
        """
        # ISING PROBLEM
        problem: __qualname__
        # Assumed target/ground state energy
        target_energy: float = float('-inf')
        # RUNTIME ARGUMENTS
        num_runs: int = 1     # Maximum number of runs to attempt
        num_timesteps_per_run: int = 1000   # Roundtrip number per run
        max_wallclock_time_allowed: int = 10000000  # Seconds before quitting
        stop_when_target_energy_reached: bool = True  # Stop if target energy reached. If code is being parallelized, will be ignored.
        amplitude_control_scheme: NoneType = None  # Manual choice of AHC vs. CAC
        custom_feedback_schedule: NoneType = None     # Custom feedback schedule
        custom_pump_schedule: NoneType = None     # Custom pump schedule
        num_parallel_runs : int = 1
        random_number_function = np.random.normal()  # Random Number Generator Function
        # HYPERPARAMETER OPTIMIZATION SCHEME
        hyperparameters_autotune: bool = False
        # AHC-NO-EXTERNAL-FIELD HYPERPARAMETERS
        ahc_noext_time_step: float = 0.05
        ahc_noext_r: float = 0.2
        ahc_noext_beta: float = 0.05
        ahc_noext_eps: float = 0.07
        ahc_noext_mu: float = 1
        ahc_noext_noise: float = 0
        ahc_nonlinearity: NoneType = None 
        # CAC-NO EXTERNAL-FIELD HYPERPARAMETERS
        cac_time_step: float = 0.05
        cac_r: float = 0.2
        cac_alpha: float = 1.0
        cac_beta: float = 0.0625
        cac_gamma: float = 9.375e-05
        cac_delta: float = 9
        cac_mu: float = 1
        cac_rho: float = 1
        cac_tau: float = 7200
        cac_noise: float = 0
        cac_nonlinearity: torch = torch.tanh
        cac_amplitude_ramp: NoneType = None
        # AHC/CAC-EXTERNAL-FIELD HYPERPARAMETERS (VRP(TW) Solver)
        ahc_ext_time_step: float = 0.015625
        ahc_ext_nsub: int = 1
        ahc_ext_alpha: float = 1.0
        ahc_ext_delta: float = 0.25
        ahc_ext_eps: float = 0.333
        ahc_ext_lambd: float = 0.001
        ahc_ext_pi: float = -0.225
        ahc_ext_rho: float = 1.0
        ahc_ext_tau: float = 100
        ahc_ext_F_h: float = 2.0
        ahc_ext_noise: float = 0
        ahc_ext_nonlinearity: torch = torch.tanh
        # RETURN CONFIGRATION
        return_lowest_energies_found_spin_configuration: bool = False
        return_lowest_energy_found_from_each_run: bool = True
        return_spin_trajectories_all_runs: bool = True
        return_number_of_solutions: int = 1000
        suppress_statements: bool = False    # Print details to screen
        # MULTITHREADING AND GPU CONFIGURATION
        use_GPU: bool = False
        use_CAC: bool = True
        chosen_device: torch.device = torch.device('cpu')
        result: __qualname__ = None

        def __post_init__(self):
            # Assertion statements:
            assert self.target_energy <= 0, "Target Ising energy must be non-positive!"
            # Run Ising solver immediately after initialization of parameters
            self.solve()

        def solve(self):
            """Solve the Ising problem using the default implementation."""
            #########################################################################################
            # Lay out problem parameters and runtime parameters.
            J = self.problem.J  # J coupling matrix (internal-field Hamiltonian)
            h = self.problem.h  # h external field terms (Zeeman term, diagonal Hamiltonian)
            target_energy = self.target_energy,  # target energy, if known. If unspecified, will default to -inf.
            # Runtime Arguments
            num_runs = self.num_runs #Number of runs total to complete.
            num_timesteps_per_run = self.num_timesteps_per_run # Number of discrete iteratiosn per run. Proportional to the number of MVM's per run.
            max_wallclock_time_allowed = self.max_wallclock_time_allowed # Specify time allowed in seconds before timeout.
            # CIM & Problem Type Controls
            amplitude_heterogeneity_correction = self.amplitude_control_scheme # Sets error variable rate of change, beta, to zero, effectively eliminating AHC.
            # Return config
            num_solutions_to_return = self.return_number_of_solutions # The number of solutions to return. Solutions are sorted in order of ascending Ising energy reached.
            return_spin_trajectories_all_runs = self.return_spin_trajectories_all_runs # Sets whether all spin evolution data is returned for all runs.
            return_verbose = not self.suppress_statements # If set to False, the function prints only essential user data for running (e.g. success probability, number of runs completed, lowest Ising energy sampled).
            return_optimal_spins_found_from_each_run = self.return_lowest_energies_found_spin_configuration 
            return_lowest_energy_found_from_each_run = self.return_lowest_energy_found_from_each_run
            use_CAC = self.use_CAC # Sets whether or not the 2021 Chaotic Amplitude Control scheme is used for modulating target amplitude and error-variable-rate-of-change.
            #########################################################################################
            N = J.shape[1]

            # Start runtime clock, initialize list of spin amplitude trajectory data.
            if amplitude_heterogeneity_correction == False:
                ahc_beta = 0
            run_data = [None] * num_runs
            run_counter = 0
            success_counter = 0
            start_time = time.time()
            # Set up GPU/CPU device.
            if self.use_GPU:
                self.chosen_device = torch.device('cuda')
            else:
                self.chosen_device = torch.device('cpu')
            # Partition number of runs specified into batches.
            num_batches = int(num_runs/self.num_parallel_runs)
            if h is None or (h == np.zeros(N)).all():  # Non-External Field Case
                h = np.zeros(N)
                print("No External Field Detected")
                if self.use_CAC == False:  # AHC Case
                    if self.hyperparameters_autotune == False: # Case in which Hyperband is not called.
                        # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                        for z in range(num_batches):
                            curr_batch = self.CIM_AHC_GPU()
                            for y in range(self.num_parallel_runs):
                                run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                            run_counter += self.num_parallel_runs
                    else:
                        tuned_parameters = self.tune_AHC()
                        tuned_parameters.pop('J') # Remove J matrix for OOP call.
                        print(f"Tuned parameters: {tuned_parameters}.")
                        # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                        for z in range(num_batches):
                            curr_batch = self.CIM_AHC_GPU(**tuned_parameters)
                            for y in range(self.num_parallel_runs):
                                run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                            run_counter += self.num_parallel_runs
                else:
                    if self.hyperparameters_autotune == False:
                        # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                        for z in range(num_batches):
                            curr_batch = self.CIM_CAC_GPU()
                            for y in range(self.num_parallel_runs):
                                run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                            run_counter += self.num_parallel_runs
                    else:
                        tuned_parameters = self.tune_CAC()
                        tuned_parameters.pop('J') # remove J matrix for OOP call
                        print(f"Tuned parameters: {tuned_parameters}.")
                        # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                        for z in range(num_batches):
                            curr_batch = self.CIM_CAC_GPU(**tuned_parameters)
                            for y in range(self.num_parallel_runs):
                                run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                            run_counter += self.num_parallel_runs
            else:  # External-Field Case
                print("External Field Detected")
                if self.hyperparameters_autotune == False:
                    # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                    for z in range(num_batches):
                        curr_batch = self.CIM_ext_f_AHC_GPU()
                        for y in range(self.num_parallel_runs):
                            run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                        run_counter += self.num_parallel_runs
                else: 
                    tuned_parameters = self.tune_AHC_ext_f()
                    tuned_parameters.pop('J') # Remove J matrix for OOP call.
                    tuned_parameters.pop('h') # Remove h matrix OOP call.
                    print(f"Tuned parameters: {tuned_parameters}.")
                    # Runs the solver in z batches in parallel, and concatenates batches into one flattened run_data list.
                    for z in range(num_batches):
                        curr_batch = self.CIM_ext_f_AHC_GPU(**tuned_parameters)
                        for y in range(self.num_parallel_runs):
                            run_data[z*self.num_parallel_runs + y] = [curr_batch[0][y,:], curr_batch[1][y,:,:], curr_batch[2], curr_batch[3][y,:], curr_batch[4][y,:,:]]
                        run_counter += self.num_parallel_runs

            # Compile Return Data, initialize return data lists.
            time_elapsed = (time.time() - start_time) # Stop wall clock.
            return_data = []
            spin_trajectories = []
            energy_trajectories = []
            all_min_energies = []
            all_min_spin_configs = []
            # Compute optimal Ising energies from each run, and appends them to the list of minimum Ising energies. 
            for z in range(num_runs):
                curr_opt_sig = (run_data[z])[0]
                all_min_spin_configs.append(curr_opt_sig)
                curr_opt_energy = -1/2 * (J.dot(curr_opt_sig.T).T).dot(curr_opt_sig.T) - h[None,:].dot(curr_opt_sig.T)
                curr_opt_energy = curr_opt_energy.item() # stripping nested numpy array
                target_energy = self.target_energy # grabbing value from tuple 
                all_min_energies.append(curr_opt_energy)
                spin_trajectories.append((run_data[z])[1])
                energy_trajectories.append((run_data[z])[3]) # Insert Ising energy evolution depending on the spin amplitude's spin configuration.
                if (target_energy is not None) and (curr_opt_energy <= target_energy): # Evaluates whether the target energy has been reached for each run.
                    success_counter += 1
            # Compile sorted minimum spin configurations and their corresponding energies.
            best_spin_configs = np.unique(
                np.array(all_min_spin_configs), axis=0, return_index=True)
            best_ising_energies = (np.array(all_min_energies))[
                best_spin_configs[1]]
            best_run = (run_data[(np.argmin(np.array(all_min_energies)))])[1]
            best_run_energy_trajectory = (
                run_data[(np.argmin(np.array(all_min_energies)))])[3]
            min_list = sorted(list(zip(best_ising_energies, best_spin_configs[0])), key=lambda k: k[0])[
                : num_solutions_to_return]
            best_ising_energies = [k[0] for k in min_list]
            best_spin_configs = [k[1] for k in min_list]
            best_spin_configs = np.array(best_spin_configs)
            best_ising_energies = np.array(best_ising_energies)
            # Run Data Structure:
            # [Optimal Spin Configuration, Spin Trajectories, Time Stopped, Energy Trajectories, (Error Variable Trajectories), Min_List]
            # Return Data Structure:
            # [Minimum Ising Energy Found, Optimal Spin Configuration, Spin Trajectories, Ising Energy Over Time, List of Min. Energies and/or Corresponding Optimal Spin Configurations, Wall Clock Time Elapsed]
            return_data.append(best_ising_energies[0])
            return_data.append(best_spin_configs[0])
            if return_spin_trajectories_all_runs:
                return_data.append(np.array(spin_trajectories))
                return_data.append(np.array(energy_trajectories, dtype=float))
            else:
                return_data.append([best_run])
                return_data.append([best_run_energy_trajectory])
            if return_lowest_energy_found_from_each_run:
                return_data.append(all_min_energies)
            if return_optimal_spins_found_from_each_run:
                return_data.append(all_min_spin_configs)

            # Prints Information about success probability and runtime statistics.
            if return_verbose:
                print(f"Target Ising Energy: {target_energy}.")
                print(f"Best Ising Energy Found: {return_data[0]}.")
                print(f"Corresponding Spin Configuration: {return_data[1]}.")
                print(f"Time Elapsed: {time_elapsed}.")
                print(f"Number of Runs Completed: {run_counter}.")
                if target_energy is not None and target_energy > -1e20:
                    print(
                        f"Success Probability: {success_counter/run_counter}.")
            result = self.problem.Result() # Sets attributes in Result class object for problem hamiltonian J, h.
            result.__setattr__("lowest_energy", return_data[0])
            result.__setattr__("lowest_energy_spin_config", return_data[1])
            result.__setattr__("spin_trajectories", return_data[2])
            result.__setattr__("energy_evolution", return_data[3])
            result.__setattr__("energies", all_min_energies)
            result.__setattr__("spin_config_all_runs", all_min_spin_configs)
            result.__setattr__("time", time_elapsed)
            result.__setattr__("result_data", return_data)
            self.result = result
            return result

        def CIM_AHC_GPU(self, **kwargs):
            """CIM solver with amplitude-heterogeneity correction; no external h field.
            Complete docstring and function information located in AHC.py.
            """            
            
            return AHC.CIM_AHC_GPU(self.num_timesteps_per_run, self.problem.J, batch_size = self.num_parallel_runs, time_step=self.ahc_noext_time_step, r=self.ahc_noext_r, beta=self.ahc_noext_beta, eps=self.ahc_noext_eps, mu=self.ahc_noext_mu, noise = self.ahc_noext_noise, custom_fb_schedule=self.custom_feedback_schedule, custom_pump_schedule=self.custom_pump_schedule, random_number_function=self.random_number_function, ahc_nonlinearity=self.ahc_nonlinearity, device=self.chosen_device)
        
        def CIM_CAC_GPU(self, **kwargs):
            """CIM solver with chaotic amplitude control; no external h field.
            Complete docstring and function information located in CAC.py.
            """
            
            return CAC.CIM_CAC_GPU(self.num_timesteps_per_run, self.problem.J, batch_size = self.num_parallel_runs, time_step = self.cac_time_step, r=self.cac_r, alpha = self.cac_alpha, beta = self.cac_beta, gamma = self.cac_gamma, delta = self.cac_delta, mu = self.cac_mu, rho = self.cac_rho, tau = self.cac_tau, noise = self.cac_noise, custom_fb_schedule = self.custom_feedback_schedule, custom_pump_schedule = self.custom_pump_schedule, cac_nonlinearity = self.cac_nonlinearity, device = self.chosen_device)
        
        def CIM_ext_f_AHC_GPU(self, **kwargs):
            """CIM solver with amplitude-heterogeneity correction with external-field.
            Complete docstring and function information located in extAHC.py.
            """

            return extAHC.CIM_ext_f_AHC_GPU(self.num_timesteps_per_run, self.problem.J, self.problem.h, batch_size = self.num_parallel_runs, nsub = self.ahc_ext_nsub, dt = self.ahc_ext_time_step, F_h = self.ahc_ext_F_h, alpha = self.ahc_ext_alpha, delta = self.ahc_ext_delta, eps = self.ahc_ext_eps, lambd = self.ahc_ext_lambd, pi = self.ahc_ext_pi, rho = self.ahc_ext_rho, tau = self.ahc_ext_tau, noise = self.ahc_ext_noise, custom_fb_schedule = self.custom_feedback_schedule, custom_pump_schedule = self.custom_pump_schedule, ahc_ext_nonlinearity= self.ahc_ext_nonlinearity, device = self.chosen_device)
        
        def tune_AHC_ext_f(self):
            """Hyperparameter tuning implementation of extAHC via BOHB.
            Complete docstring and function information located in extAHC_tuning.py
            """

            return extAHC_tuning.tune_AHC_ext_f(self.num_timesteps_per_run, self.problem.J, self.problem.h, batch_size = self.num_parallel_runs, nsub = self.ahc_ext_nsub, dt = self.ahc_ext_time_step, F_h = self.ahc_ext_F_h, alpha = self.ahc_ext_alpha, delta = self.ahc_ext_delta, eps = self.ahc_ext_eps, lambd = self.ahc_ext_lambd, pi = self.ahc_ext_pi, rho = self.ahc_ext_rho, tau = self.ahc_ext_tau, noise = self.ahc_ext_noise, custom_fb_schedule = self.custom_feedback_schedule, custom_pump_schedule = self.custom_pump_schedule, ahc_ext_nonlinearity= self.ahc_ext_nonlinearity, device = self.chosen_device)

        def tune_AHC(self):
            """Hyperparameter tuning implementation of AHC via BOHB.
            Complete docstring and function information located in AHC_tuning.py
            """
            
            return AHC_tuning.tune_AHC(self.num_timesteps_per_run, self.problem.J, batch_size = self.num_parallel_runs, time_step=self.ahc_noext_time_step, r=self.ahc_noext_r, beta=self.ahc_noext_beta, eps=self.ahc_noext_eps, mu=self.ahc_noext_mu, noise = self.ahc_noext_noise, custom_fb_schedule=self.custom_feedback_schedule, custom_pump_schedule=self.custom_pump_schedule, random_number_function=self.random_number_function, ahc_nonlinearity=self.ahc_nonlinearity, device=self.chosen_device)

        def tune_CAC(self):
            """Hyperparameter tuning implementation of CAC via BOHB.
            Complete docstring and function information located in CAC_tuning.py
            """
            
            return CAC_tuning.tune_CAC(self.num_timesteps_per_run, self.problem.J, batch_size = self.num_parallel_runs, time_step = self.cac_time_step, r=self.cac_r, alpha = self.cac_alpha, beta = self.cac_beta, gamma = self.cac_gamma, delta = self.cac_delta, mu = self.cac_mu, rho = self.cac_rho, tau = self.cac_tau, noise = self.cac_noise, custom_fb_schedule = self.custom_feedback_schedule, custom_pump_schedule = self.custom_pump_schedule, cac_nonlinearity = self.cac_nonlinearity, device = self.chosen_device)

        def brute_force(self):
            return CIM_helper.brute_force(self.J, self.h)
