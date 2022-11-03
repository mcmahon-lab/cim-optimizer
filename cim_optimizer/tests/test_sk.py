"""
PyTest Unit Tests for Reference CIM Specification with SK problem instances
Test the functionality of the CIM to ensure proper solutions returned.
(C) Copyright The Contributors 2022
Contributors: /CITATION.cff
This code is licensed under Creative Commons Attribution 4.0 International.
https://creativecommons.org/licenses/by/4.0/
"""

import sys
from pathlib import Path
cimOpt_path_str = str(Path.cwd())
sys.path.append(cimOpt_path_str)
import numpy as np
import csv
import copy
import matplotlib.pyplot as plt
import time
import torch
from cim_optimizer.solve_Ising import *
from cim_optimizer.CIM_helper import load_adjMatrix_from_rudy

inst_path_str_SK1 = str(Path.cwd()) + "\\instances\\SK_Instances_NPZ\\"

#The CIM-solver being tested within this test class is the Chaotic Amplitude Control Algorithm, found in /cim_optimizer/CAC.py.
class Test_SK:
    """ Test Sherrington-Kirkpatrick (SK) problems without external-field terms.
    Uses dense SK problems generated for benchmarking in the following article:
    http://doi.org/10.1126/sciadv.aau0823
    """

    def test_N10(self):
        N = 10
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        test_result = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 10000, target_energy = ground_state_energy, return_spin_trajectories_all_runs=True)

        solved_energy = test_result.result.lowest_energy
        assert solved_energy <= 0.8*ground_state_energy

    def test_N20(self):
        N = 20
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        test_result = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 10000, target_energy = ground_state_energy, return_spin_trajectories_all_runs=True)

        solved_energy = test_result.result.lowest_energy
        assert solved_energy <= 0.8*ground_state_energy


    def test_N30(self):
        N = 30
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        test_result = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 10000, target_energy = ground_state_energy, return_spin_trajectories_all_runs=True)

        solved_energy = test_result.result.lowest_energy
        assert solved_energy <= 0.8*ground_state_energy

    def test_N100_runtime(self):
        N = 100
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 2500, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1)
        
        run_time = solution.result.time
        # expecting at worse 2 minutes to reach a near-ground solution
        assert run_time < 2*60

    def test_N100(self):
        N = 100
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 2500, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1)
                # expecting at worse 5 minutes to reach a near-ground solution
        solved_energy = solution.result.lowest_energy
        assert solved_energy <= 0.8*ground_state_energy

    def test_N500_runtime(self):
        N = 500
        sk_id = 1
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 50000, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1)
        
        run_time = solution.result.time
        # expecting at worse 5 minutes to reach a near-ground solution
        assert run_time < 5*60

    def test_N1000_runtime(self):
        N = 1000
        sk_id = 1
        loaded_energies = np.load(inst_path_str_SK1 + f"SK1_N={N}_ground_energies.npz")['arr_0']
        ground_state_energy = (loaded_energies)[sk_id-1]
        J = - np.load(inst_path_str_SK1 + f"SK1_N={N}_{sk_id-1}.npz")
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 5, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1)
        
        run_time = solution.result.time
        # expecting at worse 10 minutes to reach a near-ground solution
        assert run_time < 10*60