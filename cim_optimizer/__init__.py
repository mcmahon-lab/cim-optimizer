# Reference CIM File
# Last Updated: September 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/

"""
This is the top level of cim-optimizer, containing functions
for optimization of Ising-based problems that can be 
represented with or without an external field.

Code details
------------
"""
import cim_optimizer.solve_Ising # contains class Ising and subclass results, solver
import cim_optimizer.CIM_helper # contains helper functions for solve_Ising and preprocessing functions
from cim_optimizer.optimal_params import * # contains optimal hyperparameters for well-known problems
# import all solvers
import cim_optimizer.AHC_tuning
import cim_optimizer.AHC
import cim_optimizer.CAC_tuning
import cim_optimizer.CAC
import cim_optimizer.extAHC_tuning
import cim_optimizer.extAHC