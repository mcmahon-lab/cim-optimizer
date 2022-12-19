"""
PyTest Unit Tests for Reference CIM Specification with MIRP problem instances

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

inst_path_str_MIRP = str(Path.cwd()) + "\\instances\\MIRP_TestSet\\"
inst_path_str_VRPTW = str(Path.cwd()) + "\\instances\\VRPTW_TestSet\\"

#The CIM-solver being tested within this test class is the External Field-Amplitude Heterogeneity Correction Algorithm, found in /cim_optimizer/extaAHC.py.
class Test_MIRP:
    """ Test Maritime Inventory Routing Problems (MIRP) with external-field terms.
    """

    def test_vrptw_sb_308_f(self):
        # time horizon 60 sequence-based problem
        J, h = load_adjMatrix_from_rudy(inst_path_str_VRPTW + "test_sb_308_f.rudy", delimiter='\t', index_start=1, preset_size=308)
        J = -J # flip sign to compute minima instead of maxima
        ground_state_energy = -2757.50
        time_span = 10000
        test_result = Ising(J, h).solve(num_runs = 1, num_timesteps_per_run = time_span, target_energy = ground_state_energy, hyperparameters_randomtune = False)

        solved_energy = test_result.result.lowest_energy
        assert solved_energy <= ground_state_energy * 0.8
    
    def test_vrptw_pb_687_f(self):
        # time horizon 60 sequence-based problem
        J, h = load_adjMatrix_from_rudy(inst_path_str_VRPTW + "test_pb_687_f.rudy", delimiter='\t', index_start=1, preset_size=687)
        J = -J # flip sign to compute minima instead of maxima
        ground_state_energy = -55872.50
        time_span = 50000
        test_result = Ising(J, h).solve(num_runs = 4, num_parallel_runs = 2, num_timesteps_per_run = time_span, target_energy = ground_state_energy, hyperparameters_randomtune = False)
        # confirm batching performance
        solved_energy = test_result.result.lowest_energy
        assert solved_energy <= ground_state_energy * 0.8
