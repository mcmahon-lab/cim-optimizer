# Hyperparameter Tuning of CAC algorithm using Bayesian Optimization Hyperband
# Created September 2022
# Last Updated Oct 17th, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import torch 
from cim_optimizer import CAC
import numpy as np
from bohb import BOHB
import bohb.configspace as cs

def evaluate_CAC(params, n_iterations):
    """ Compute objective function that we are minimized. 
    Defined as min energy achieved divided by iteration count.
    
    Attributes
    ----------
    params : dictionary
        Arguments for CAC algorithm. Includes fixed hyperparameters
        and configurable hyperparameters.
    n_iterations : int, default=1
        Number of parallel runs set by bohb-hpo.

    Returns
    ----------
    loss : double
        Average of minimum energies found acrosss each run. Used as 
        the objective to minimize for Bayesian optimization Hyperband.
    """

    # min energy achieved
    energies = CAC.CIM_CAC_GPU(**params)[3]
    min_energy = np.amin(energies) 

    loss = min_energy/n_iterations # loss defined as average of the minimum ising energies found across each run
    return loss # minimize loss

def tune_CAC(T_time, J, batch_size=1, time_step=0.05, r=-4.04, alpha=3.0, beta=0.25, gamma=0.00011, delta=10, mu=1, rho=3, tau=1000, noise=0, H0=None, stop_when_solved=False, num_sol=10, custom_fb_schedule=None, custom_pump_schedule=None, cac_nonlinearity=torch.tanh, device=torch.device('cpu')):
    """ BOHB implementation of CIM simulator with chaotic amplitude control; no h field

    Attributes
    ----------
    T_time : int
        Roundtrip number per run, representing time horizon.
    J : ndarray
        Ising spin-spin coupling matrix.
    batch_size : int, default=1
        Number of trials to run simultaneously as a batch; helpful for
        parallelism.
    time_step : float, default=0.05
        Number of steps per T_time.
    H0 : int, optional
        Assumed target/ground energy for the solver to reach, used for 
        early stopping.
    stop_when_solved : bool, default=False
        Stop run if ground state energy is achieved (not applicable
        for BOHB runs).
    num_sol : int, default=10
        Number of best solutions to return; must be <= num_runs.
    custom_feedback_schedule : optional, default=None
        Option to specify a custom function or array of length
        time_step to use as a feedback schedule.
    custom_pump_schedule : optional, default=None
        Option to specify a custom function or array of
        length time_step to use as a pump schedule.
    random_number_function : optional, default=np.random.normal()
        Random number generator function for nonlinearity.  
    device : optional, default=torch.device('cpu')
        Device for pytorch tensor computations.
    r : float, default=-4.04
        CAC hyperparameter
    alpha : float, default=3.0
        CAC hyperparameter
    beta : float, default=0.25
        CAC hyperparameter
    gamma : float, defualt=0.00011
        CAC hyperparameter
    delta : float, default=10
        CAC hyperparameter
    mu : float, default=1
        CAC hyperparameter
    rho : float, default=3
        CAC hyperparameter
    tau : float, default=1000
        CAC hyperparameter
    noise : float, default=0
        CAC hyperparameter
    cac_nonlinearity : default=torch.tanh 
        CAC hyperparameter

    Returns
    ----------
    configspace : dictionary
        Returns the tuned hyperparameter selection determined
        by BOHB from the final call.  
    """
    
    # Load Hyperparameters into configSpace
    T_time = cs.CategoricalHyperparameter('T_time',[T_time])
    J = cs.CategoricalHyperparameter('J',[J])
    time_step = cs.CategoricalHyperparameter('time_step',[time_step])
    gamma = cs.CategoricalHyperparameter('gamma',[gamma])
    delta = cs.CategoricalHyperparameter('delta', [delta])
    mu = cs.CategoricalHyperparameter('mu', [mu])
    tau = cs.CategoricalHyperparameter('tau', [tau])
    noise = cs.CategoricalHyperparameter('noise', [noise])
    H0 = cs.CategoricalHyperparameter('H0', [H0])
    stop_when_solved = cs.CategoricalHyperparameter('stop_when_solved', [stop_when_solved])
    num_sol = cs.CategoricalHyperparameter('num_sol', [num_sol])
    custom_fb_schedule = cs.CategoricalHyperparameter('custom_fb_schedule', [custom_fb_schedule])
    custom_pump_schedule = cs.CategoricalHyperparameter('custom_pump_schedule', [custom_pump_schedule])
    device = cs.CategoricalHyperparameter('device', [device])
    batch_size = cs.CategoricalHyperparameter('batch_size', [batch_size])
    cac_nonlinearity = cs.CategoricalHyperparameter('cac_nonlinearity', [cac_nonlinearity])

    # Load configurable hyperparameters into configspace; defaulted to ±20%
    beta = cs.UniformHyperparameter('beta', min(beta*0.8, beta*1.2), max(beta*.8, beta*1.2))
    r = cs.UniformHyperparameter('r', min(r*0.8, r*1.2), max(r*.8, r*1.2))
    alpha = cs.UniformHyperparameter('alpha', min(alpha*0.8, alpha*1.2), max(alpha*.8, alpha*1.2))
    rho = cs.UniformHyperparameter('rho', min(rho*0.8, rho*1.2), max(rho*.8, rho*1.2))
    
    configspace = cs.ConfigurationSpace([T_time, J, time_step, r, alpha, beta, gamma, delta, mu, rho, tau, noise, H0, stop_when_solved, num_sol, device, batch_size, custom_fb_schedule, custom_pump_schedule, cac_nonlinearity])
    # Run BOHB with max_budget=10 and min_budget=1
    opt = BOHB(configspace, evaluate_CAC, max_budget=10, min_budget=1)
    logs = opt.optimize()

    # return final hyperparameter space as a dictionary
    return configspace.sample_configuration().to_dict()