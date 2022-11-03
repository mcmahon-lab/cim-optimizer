# CIM Optimizer Library 

All code in this folder is dedicated to the internal workings of the CIM simulator. The main class `Ising`, along with its sublcasses `Result` and `solver` are all found in `solve_Ising.py`. Helper functions are located in `CIM_helper.py`. The dynamics of our CIM simulators are located in files `AHC/CAC/extAHC.py`, while tuning functions are found in `AHC/CAC/extAHC_tuning.py`.

## `solve_Ising.py`

The `Result` nested class contains all metadata stored about each CIM run, including internal dynamics, wall clock time, graph functions, etc. The `solver` nested class contains adjustable hyperparameters, along with batching functions and CIM solver calls.

## `CIM_helper.py`

Contains helper functions such as `load_adjMatrix_from_rudy` for loading `.rudy` files, `get_binary_list` for converting an integer into a binary list, and `brute_force` for computing the minimum energy of all spin configuration for an Ising problem. 

## `AHC/CAC/extAHC.py`

`extAHC.py` and `AHC.py` contain implementations of amplitude-heterogeneity correction proposed by Leleu et al. with and without external fields, respectively. `CAC.py` contains the implementation of chaotic amplitude control proposed by Leleu et al. (which does not support external fields). The default hyperparameters for the CAC solver are set from the supplementary information found within the work listed here: (https://doi.org/10.1038/s42005-021-00768-0), while the default hyperparameters for the non-external field AHC solver are set from the supplementary information found within the work listed here: (https://doi.org/10.1103/PhysRevLett.122.040607).

## `AHC/CAC/extAHC_tuning.py`

All three scripts contain a `evaluate_AHC/CAC/extAHC.py` function, which computes the minimum energy achieved from a set of hyperparameters and utilizes it as an objective function for BOHB, along with `tune_AHC/CAC/extAHC.py`, which loads the initial and tuned hyperparameters into the search space. 

## `optimal_params.py`

Contains optimal hyperparameters for well-defined instances including MAX-CUT, SK, and GSet, and returns these values as dictionaries for easy use.