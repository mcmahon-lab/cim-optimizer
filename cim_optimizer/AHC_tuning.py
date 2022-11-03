# Hyperparameter Tuning of AHC algorithm using Bayesian Optimization Hyperband
# Created September 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import torch 
from cim_optimizer import AHC
import numpy as np
from bohb import BOHB
import bohb.configspace as cs

def evaluate_AHC(params, n_iterations):
    """ Compute objective function that we are minimized. 
    Defined as min energy achieved divided by iteration count.

    Attributes
    -----------
    params : dictionary
        Arguments for AHC algorithm. Includes fixed hyperparameters
        and configurable hyperparameters.
    n_iterations : int, default=1
        Number of parallel runs set by bohb-hpo.

    Returns
    --------
    loss : double
        Average of minimum energies found acrosss each run. Used as 
        the objective to minimize for Bayesian optimization Hyperband.
    """

    # min energy achieved
    energies = AHC.CIM_AHC_GPU(**params)[3]
    min_energy = np.amin(energies)

    loss = min_energy/n_iterations # loss defined as average of the minimum ising energies found across each run
    return loss # minimize loss

def tune_AHC(T_time, J, batch_size = 1, time_step=0.05, r=0.2, beta=0.05, eps=0.07, mu=1, noise=0, custom_fb_schedule=None, custom_pump_schedule=None, random_number_function=None, ahc_nonlinearity=None, device=torch.device('cpu')):
    """ BOHB implementation of CIM simulator with amplitude-heterogeneity correction; no h field

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
    r : float, default=0.2
        AHC hyperparameter
    beta : float, default=0.05
        AHC hyperparameter
    eps : float, default=0.07
        AHC hyperparameter
    mu : float, default=1
        AHC hyperparameter
    noise : float, default=0
        AHC hyperparameter
    ahc_nonlinearity : torch function
        AHC hyperparameter, default=torch.tanh 

    Returns
    ----------
    configspace : dictionary
        Returns the tuned hyperparameter selection determined
        by BOHB from the final call.  
    """
    
    # Load Hyperparameters into configspace
    T_time = cs.CategoricalHyperparameter('T_time',[T_time])
    J = cs.CategoricalHyperparameter('J',[J])
    time_step = cs.CategoricalHyperparameter('time_step',[time_step]) 
    random_number_function = cs.CategoricalHyperparameter('random_number_function', [random_number_function])
    device = cs.CategoricalHyperparameter('device', [device])
    batch_size = cs.CategoricalHyperparameter('batch_size',[batch_size])
    custom_fb_schedule = cs.CategoricalHyperparameter('custom_fb_schedule', [custom_fb_schedule])
    custom_pump_schedule = cs.CategoricalHyperparameter('custom_pump_schedule', [custom_pump_schedule])
    ahc_nonlinearity = cs.CategoricalHyperparameter('ahc_nonlinearity', [ahc_nonlinearity])

    # Load configurable hyperparameters into configspace; defaulted to ±20%
    eps = cs.UniformHyperparameter('eps', min(eps*0.8, eps*1.2), max(eps*.8, eps*1.2))
    beta = cs.UniformHyperparameter('beta', min(beta*0.8, beta*1.2), max(beta*.8, beta*1.2))
    r = cs.UniformHyperparameter('r', min(r*0.8, r*1.2), max(r*.8, r*1.2))
    mu = cs.UniformHyperparameter('mu', min(mu*0.8, mu*1.2), max(mu*.8, mu*1.2))
    noise = cs.UniformHyperparameter('noise', min(noise*0.8, noise*1.2), max(noise*.8, noise*1.2))

    configspace = cs.ConfigurationSpace([T_time, J, time_step, random_number_function, device, batch_size, custom_fb_schedule, custom_pump_schedule, ahc_nonlinearity, eps, beta, r, mu, noise])
    # Run BOHB with max_budget=10 and min_budget=1
    opt = BOHB(configspace, evaluate_AHC, max_budget=10, min_budget=1)
    logs = opt.optimize()

    # return final hyperparameter space as a dictionary
    return configspace.sample_configuration().to_dict()