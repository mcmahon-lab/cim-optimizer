# Hyperband PyTorch Implementation of CAC Algorithm (with GPU-acceleration)
# Created August 2022
# Last Updated October 31st, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import torch
import numpy as np
import math
import time
torch.backends.cudnn.benchmark = True

def CIM_CAC_GPU(T_time, J, batch_size=1, time_step=0.05, r=None, alpha=3.0, beta=0.25, gamma=0.00011, delta=10, mu=1, rho=3, tau=1000, noise=0, H0=None, stop_when_solved=False, num_sol=10, custom_fb_schedule=None, custom_pump_schedule=None, cac_nonlinearity=torch.tanh, device=torch.device('cpu')):
    """CIM solver with chaotic amplitude control; no external h field.

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
        Stop run if ground state energy is achieved.
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
        Target device for tensor computations.
    r : float, default=0.8-(N/220)**2 
        CAC hyperparameter
    alpha: float, default=3.0
        CAC hyperparameter
    beta : float, default=0.25
        CAC hyperparameter
    gamma: float, default=0.00011
        CAC hyperparameter
    delta: float, default=10.0
        CAC hyperparameter
    mu : float, default=1.0
        CAC hyperparameter
    rho: float, default=3.0
        CAC hyperparameter
    tau: float, default=1000
        CAC hyperparameter
    noise : float, default=0.0
        CAC hyperparameter
    cac_nonlinearity : default=torch.tanh 
        CAC hyperparameter
    
    Returns
    ----------
    sig_opt : NumPy array
        2-d NumPy array of batches x optimal spins where optimal spins
        are defined by the spin configuration that achieved the lowest
        energy up to that point.
    spin_amplitude_trajectory : NumPy array
        3-d NumPy array of batches x spins x time steps.
    t : int 
        Number of time steps: defined as int(T_time/time_step).
    energy_plot_data : NumPy array
        2-d NumPy array defined as batches x energy_achieved.
    error_var_data : NumPy array
        3-d NumPy array defined as batches x error variables x time steps.
    """

    #  Reference: 
    #  [1] Leleu, T., Khoyratee, F., Levi, T. et al. Scaling advantage of chaotic 
    #      amplitude control for high-performance combinatorial optimization. 
    #      Commun Phys 4, 266 (2021). https://doi.org/10.1038/s42005-021-00768-0


    #Compute instance sizes, cast Ising problem matrix to torch tensor.
    MAX_FLOAT = 500
    EMAX_FLOAT = 32
    J = torch.from_numpy(J)
    J = J.float().to(device)
    N = J.size()[1]
    if r == None:
        r = 0.8-(N/220)**2
    #Initialize plot and runtime data arrays.
    xi = torch.zeros(batch_size).to(device)
    t_c = torch.zeros(batch_size).to(device)
    H = torch.zeros(batch_size).to(device)
    spin_amplitude_trajectory = torch.zeros(batch_size, N, T_time).to(device)
    error_var_data = torch.zeros(batch_size, N, T_time).to(device)
    energy_plot_data = torch.zeros(batch_size, T_time).to(device)
    t_opt = torch.zeros(batch_size).to(device)
    #Initialize Spin-Amplitude Vectors and Auxiliary Variables
    x = 0.001 * torch.rand(batch_size, N).to(device) - 0.0005
    error_var = torch.ones(batch_size, N).to(device).float()
    effective_tau = tau/time_step

    #Configure ramp schedules.
    if custom_fb_schedule is None:
        beta = torch.ones(T_time).to(device)*beta
    else:
        beta = custom_fb_schedule(torch.arange(0, T_time).to(device))
    if custom_pump_schedule is None:
        r = torch.ones(T_time).to(device)*r
    else:
        r = custom_pump_schedule(torch.arange(0, T_time).to(device))

    #Compute initial Ising energy and spin states.
    sig = ((2 * (x > 0) - 1).float()).to(device)
    H = (-1/2*(torch.bmm(sig.view(batch_size, 1, N), (sig @ J).view(batch_size, N, 1)))[:,:,0]).view(batch_size)

    H_opt = H
    sig_opt = sig
    a = alpha * torch.ones(batch_size).to(device)
    #Simulate Time-Evolution of Spin Amplitudes
    for t in range(T_time):
        #Save spin states at current iteration.
        x_ = x
        spin_amplitude_trajectory[:, :, t] = x_
        error_var_data[:, :, t] = error_var
        sig = ((2 * (x > 0) - 1).float())
        H = (-1/2*(torch.bmm(sig.view(batch_size, 1, N), (sig @ J).view(batch_size, N, 1)))[:,:,0]).view(batch_size)
        
        #Euler step for equations of motion of spin amplitudes.
        x_squared = x**2
        MVM = x @ J 
        x += time_step*(x*((r[t]-1) - mu*x_squared))
        x += time_step*beta[t]*(MVM * error_var) 
        x += beta[t]*noise*(torch.rand(batch_size, N, device=device)-0.5) 

        #Save current Ising energy.
        energy_plot_data[:, t] = H

        #Euler step for equations of motion of error variables.
        delta_e = -xi[:, None]*(x_**2 - a[:, None])*error_var 
        error_var += delta_e*time_step 

        #Normalize auxiliary error variables.
        error_var[error_var > EMAX_FLOAT] = EMAX_FLOAT
        
        #Modulate target amplitude, error variable rate of change parameters depending on Ising energy.
        xi += gamma*time_step
        dH = H - H_opt 
        a = alpha + rho*cac_nonlinearity(delta*dH) 

        #Use boolean-array indexing to update ramp schedules and minimum Ising energy.
        t_c[t_c < t - effective_tau] = t
        xi[t_c < t - effective_tau] = 0
        t_opt[H < H_opt] = t
        t_c[H < H_opt] = t
        H_opt = torch.minimum(H_opt, H) 

    #Parse and Return Solutions
    spin_amplitude_trajectory = spin_amplitude_trajectory.cpu()
    spin_plot_data = 2 * (spin_amplitude_trajectory > 0) - 1

    energy_plot_data = energy_plot_data.cpu()

    for k in range(batch_size):
        sig_opt[k,:] = spin_plot_data[k,:,t_opt.long()[k]]
    sig_opt = sig_opt.cpu()
    error_var_data = error_var_data.cpu()
    return (sig_opt.numpy(), spin_amplitude_trajectory.numpy(), t, energy_plot_data.numpy(), error_var_data.numpy())