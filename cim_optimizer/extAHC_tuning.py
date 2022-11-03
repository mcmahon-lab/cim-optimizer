# Hyperparameter Tuning of external field AHC algorithm using Bayesian Optimization Hyperband
# Created October 2022
# Last Updated Oct 17th, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import torch 
from cim_optimizer import extAHC
import numpy as np
from bohb import BOHB
import bohb.configspace as cs

def evaluate_AHC_ext_f(params, n_iterations):
    """ Compute objective function that we are minimized. 
    Defined as min energy achieved divided by iteration count.
    
    Attributes
    -----------
    params : dictionary
        Arguments for AHC algorithm with external fields. Includes 
        fixed hyperparameters and configurable hyperparameters.
    n_iterations : int, default=1
        Number of parallel runs set by bohb-hpo.

    Returns
    ----------
    loss : double
        Average of minimum energies found acrosss each run. Used as 
        the objective to minimize for Bayesian optimization Hyperband.
    """

    # min energy achieved
    energies = extAHC.CIM_ext_f_AHC_GPU(**params)[3]
    min_energy = np.amin(energies) 

    loss = min_energy/n_iterations # loss defined as average of the minimum ising energies found across each run
    return loss #minimize loss

def tune_AHC_ext_f(time_stop, J, h, batch_size = 1, nsub = 1, dt=0.015625, F_h=2.0, alpha=1.0, beta=0.05, delta=0.25, eps=0.333, lambd=0.001, pi=-0.225, rho=1.0, tau= 100, noise=0, ahc_ext_nonlinearity = None, custom_fb_schedule=None, custom_pump_schedule=None, device=torch.device('cpu')):
    """ CIM solver with amplitude-heterogeneity correction and external-field.

    Attributes
    -----------
    time_stop : int
        Roundtrip number per run, representing time horizon.
    J : ndarray
        Ising spin-spin coupling matrix.
    h : ndarray
        External-field vector (individual weights)
    batch_size : int, default=1
        Number of trials to run simultaneously as a batch; helpful for
        parallelism.
    dt : float, default=0.015625
        Number of steps per T_time.
    custom_feedback_schedule : optional, default=None
        Option to specify a custom function or array of length
        time_step to use as a feedback schedule.
    custom_pump_schedule : optional, default=None
        Option to specify a custom function or array of
        length time_step to use as a pump schedule.
    device : optional, default=torch.device('cpu')
        Device for pytorch tensor computations. 
    nsub : int, default=1
        extAHC hyperparameter
    F_h : float, default=2.0
        extAHC hyperparameter
    alpha : float, default=1.0
        extAHC hyperparameter
    beta : float, default=0.05
        extAHC hyperparameter
    delta : float, default=0.25
        extAHC hyperparameter
    eps : float, default=0.333
        extAHC hyperparameter
    lambd : float, default=0.001
        extAHC hyperparameter
    pi : float, default=-0.225
        extAHC hyperparameter
    rho : float, default=1.0
        extAHC hyperparameter
    tau : int, default=1000
        extAHC hyperparameter
    noise : float, default=0
        extAHC hyperparameter
    ahc_ext_nonlinearity : default=None
        extAHC hyperparameter
    
    Returns
    ----------
    configspace : dictionary
        Returns the tuned hyperparameter selection determined
        by BOHB from the final call.  
    """

    # Load Hyperparameters into configSpace
    time_stop = cs.CategoricalHyperparameter('time_stop', [time_stop])
    J = cs.CategoricalHyperparameter('J', [J])
    h = cs.CategoricalHyperparameter('h', [h])
    batch_size = cs.CategoricalHyperparameter('batch_size',[batch_size])
    nsub = cs.CategoricalHyperparameter('nsub', [nsub])
    dt = cs.CategoricalHyperparameter('dt', [dt])
    beta = cs.CategoricalHyperparameter('beta', [beta])
    delta = cs.CategoricalHyperparameter('delta', [delta])
    lambd = cs.CategoricalHyperparameter('lambd', [lambd])
    rho = cs.CategoricalHyperparameter('rho', [rho])
    tau = cs.CategoricalHyperparameter('tau', [tau])
    ahc_ext_nonlinearity = cs.CategoricalHyperparameter('ahc_ext_nonlinearity', [ahc_ext_nonlinearity])
    custom_fb_schedule = cs.CategoricalHyperparameter('custom_fb_schedule', [custom_fb_schedule])
    custom_pump_schedule = cs.CategoricalHyperparameter('custom_pump_schedule', [custom_pump_schedule])
    device = cs.CategoricalHyperparameter('device', [device])

    # Load configurable hyperparameters into configspace; defaulted to ±20%
    eps = cs.UniformHyperparameter('eps', min(eps*0.8, eps*1.2), max(eps*.8, eps*1.2))
    F_h = cs.UniformHyperparameter('F_h', min(F_h*0.8, F_h*1.2), max(F_h*.8, F_h*1.2))
    pi = cs.UniformHyperparameter('pi', min(pi*0.8, pi*1.2), max(pi*.8, pi*1.2))
    alpha = cs.UniformHyperparameter('alpha', min(alpha*0.8, alpha*1.2), max(alpha*.8, alpha*1.2))    
    noise = cs.UniformHyperparameter('noise', min(noise*0.8, noise*1.2), max(noise*.8, noise*1.2))

    configspace = cs.ConfigurationSpace([time_stop, J, h, batch_size, nsub, dt, beta, delta, lambd, rho, tau, ahc_ext_nonlinearity, custom_fb_schedule, custom_pump_schedule, device, eps, F_h, pi, alpha, noise])
    # Run BOHB with max_budget=10 and min_budget=1
    opt = BOHB(configspace, evaluate_AHC_ext_f, max_budget=10, min_budget=1)
    logs = opt.optimize()
    # Return final hyperparameter space as a dictionary
    return configspace.sample_configuration().to_dict()