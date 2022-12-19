"""
PyTest Unit Tests for Reference CIM Specification with GSet problem instances

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

inst_path_str_Gset = str(Path.cwd()) + "\\instances\\Stanford_Gset\\"

#The CIM-solver being tested within this test class is the Chaotic Amplitude Control Algorithm, found in /cim_optimizer/CAC.py.
class Test_GSet:
    """ Test MAX-CUT problems from the Stanford G-Set found at
    https://web.stanford.edu/~yyye/yyye/Gset/.
    """

    def test_G1(self):
        filepath = inst_path_str_Gset + "G1.txt"
        f = 11624  # problem-specific maximization function output
        # source: http://dx.doi.org/10.1016/j.engappai.2012.09.001 Tbl 2
        # number of edges in G1 graph (number of lines in txt file * 2)
        e = 38352
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        ground_energy = (e - 4 * f) / 2     # Calculate MAX-CUT ground energy
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 100000, target_energy =  ground_energy, return_spin_trajectories_all_runs=True,
                                cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        solved_energy = solution.result.lowest_energy
        assert solved_energy <= 0.8*ground_energy

    def test_G2(self):
        filepath = inst_path_str_Gset + "G2.txt"
        f = 11620  # problem-specific maximization function output
        # source: http://dx.doi.org/10.1016/j.engappai.2012.09.001 Tbl 2
        # number of edges in G2 graph (number of lines in txt file * 2)
        e = 38352
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        ground_energy = (e - 4 * f) / 2     # Calculate MAX-CUT ground energy
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 100000, target_energy = ground_energy, return_spin_trajectories_all_runs=True,
                                cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        solved_energy = solution.result.lowest_energy
        assert solved_energy <= 0.8*ground_energy

    def test_G70(self):
        filepath = inst_path_str_Gset + "G70.txt"
        f = 9499  # problem-specific maximization function output
        # source: http://dx.doi.org/10.1016/j.engappai.2012.09.001 Tbl 2
        # number of edges in G2 graph (number of lines in txt file * 2)
        e = 19998
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        ground_energy = (e - 4 * f) / 2     # Calculate MAX-CUT ground energy
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 1000, target_energy = ground_energy, return_spin_trajectories_all_runs=True,
                                cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        solved_energy = solution.result.lowest_energy
        assert solved_energy <= 0.8*ground_energy

    def test_G1_runtime(self):
        filepath = inst_path_str_Gset + "G1.txt"
        f = 11624  # problem-specific maximization function output
        e = 38352  # number of edges in G1 graph (number of lines in txt file * 2)
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        ground_energy = (e - 4 * f) / 2     # Calculate MAX-CUT ground energy
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 5, target_energy =  ground_energy, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        run_time = solution.result.time
        # time bound obtained from https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040607
        # expecting at worse 2x worse performance
        assert run_time < 2 * 12.69 

    def test_G2_runtime(self):
        filepath = inst_path_str_Gset + "G2.txt"
        f = 11620  # problem-specific maximization function output
        e = 38352  # number of edges in G1 graph (number of lines in txt file * 2)
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        ground_energy = (e - 4 * f) / 2     # Calculate MAX-CUT ground energy
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 5, target_energy =  ground_energy, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        run_time = solution.result.time
        # time bound obtained from https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040607
        # expecting at worse 2x worse performance
        assert run_time < 2 * 49.07

    def test_G42_runtime(self):
        filepath = inst_path_str_Gset + "G2.txt"
        # source: http://dx.doi.org/10.1016/j.engappai.2012.09.001 Tbl 2
        J = - load_adjMatrix_from_rudy(filepath, delimiter=' ', index_start=1)[0]
        N = J.shape[0]
        solution = Ising(J).solve(num_runs = 1, num_timesteps_per_run = 5, return_spin_trajectories_all_runs=True,
                                cac_time_step=0.05, cac_r = 0.2, cac_alpha = 1.0, cac_beta = 3*N/(np.sum(np.abs(J))), cac_gamma = 0.075/N,
                                cac_delta = 9, cac_tau = 9*N, cac_rho = 1, hyperparameters_randomtune = False)
        run_time = solution.result.time
        # time bound obtained from https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040607
        # expecting at worse 2x worse performance
        assert run_time < 2 * 199.26