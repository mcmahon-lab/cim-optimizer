# Hyperband PyTorch Implementation of AHC Algorithm (with GPU-acceleration)
# Created August 2022
# Last Updated Sept 23rd, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/

import torch
import numpy as np
torch.backends.cudnn.benchmark = True

def CIM_AHC_GPU(T_time, J, batch_size = 1, time_step=0.05, r=0.2, beta=0.05, eps=0.07, mu=1, noise=0, custom_fb_schedule=None, custom_pump_schedule=None, random_number_function=None, ahc_nonlinearity=None, device=torch.device('cpu')):
    """ CIM solver with amplitude-heterogeneity correction; no h field

    Attributes
    -----------
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
    ahc_nonlinearity : default=torch.tanh 
        AHC hyperparameter
    
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
    #  hyperparameters are obtained from the supplemental material of https://doi.org/10.1103/PhysRevLett.122.040607

    #  Reference: 
    #  [1] T. Leleu, Y. Yamamoto, P. L. McMahon, and K. Aihara, Destabilization
    #      of local minima in analog spin systems by correction of amplitude
    #      heterogeneity. Phys Rev Lett 122, 040607 (2019). 
    #      https://doi.org/10.1103/PhysRevLett.122.040607


    #Compute instance sizes, cast Ising problem matrix to torch tensor.
    J = torch.from_numpy(J)
    J = J.float().to(device)
    N = J.size()[1]

    #Initialize plot arrays and runtime variables.
    end_ising_energy = (1e20*torch.ones(batch_size)).to(device)

    target_a_baseline = 0.2
    target_a = (target_a_baseline*torch.ones(batch_size)).to(device)

    ticks = int(T_time/time_step)
    spin_amplitude_trajectory = torch.zeros(batch_size, N, ticks).to(device)
    error_var_data = torch.zeros(batch_size, N, ticks).to(device)
    energy_plot_data = torch.zeros(batch_size, ticks).to(device)
    t_opt = torch.zeros(batch_size).to(device)
    EMAX_FLOAT = 32.0

    #Initialize Spin-Amplitude Vectors and Auxiliary Variables
    x = 0.001 * torch.rand(batch_size, N).to(device) - 0.0005
    error_var = torch.ones(batch_size, N).to(device).float()
    etc_flag = torch.ones(batch_size, N).to(device)
    sig_ = ((2 * (x > 0) - 1).float()).to(device)
    sig_opt = sig_

    #Configure ramp schedules, random number function.
    if random_number_function is None:
        random_number_function = lambda c : torch.rand(c,3)

    if custom_fb_schedule is None:
        eps = (torch.ones(ticks)*eps).to(device)
    else:
        eps = custom_fb_schedule((torch.arange(0, ticks, time_step))).to(device)
    if custom_pump_schedule is None:
        r = (torch.ones(ticks)*r).to(device)
    else:
        r = (custom_pump_schedule(np.arange(0, ticks, time_step))).to(device)

    if ahc_nonlinearity is None:
        ahc_nonlinearity = lambda c : torch.pow(c,3)

    #Spin evolution euler-step iteration loop.
    for t in range(ticks):
        #Update spin states and Ising energy.
        sig = ((2 * (x > 0) - 1).float())
        sig_ = sig 

        #Save current Ising energy.
        curr_ising_energy = (-1/2*(torch.bmm(sig.view(batch_size, 1, N), (sig @ J).view(batch_size, N, 1)))[:,:,0]).view(batch_size)
        energy_plot_data[:, t] = curr_ising_energy

        #Simulate Time-Evolution of Spin Amplitudes
        spin_amplitude_trajectory[:, :, t] = x
        error_var_data[:, :, t] = error_var
        x_squared = x**2
        MVM = x @ J 
        x += time_step*(x*((r[t]-1) - mu*x_squared))
        x += time_step*eps[t]*(MVM*error_var)
        x += eps[t]*noise*(torch.rand(N, device=device)-0.5) 

        #Modulate target amplitude, error variable rate of change parameters depending on Ising energy.
        delta_a = eps[t]*torch.mean((sig @ J)*sig*etc_flag, 1)
        target_a = target_a_baseline + delta_a
        x_squared = x**2

        #Euler step for equations of motion of error variables.
        error_var += time_step * \
            (-beta*((x_squared) - target_a[:, None])*error_var)

        #Normalize auxiliary error variables.
        error_var[error_var > EMAX_FLOAT] = EMAX_FLOAT
        
        #Use boolean-array indexing to update ramp schedules and minimum Ising energy.
        comparison = torch.any(sig_ != sig, 1)
        etc_flag[comparison, :] = error_var[comparison, :]
        t_opt[curr_ising_energy < end_ising_energy] = t
        end_ising_energy = torch.minimum(end_ising_energy, curr_ising_energy) 

    #Parse and Return Solutions
    sig = ((2 * (x > 0) - 1).float())
    spin_amplitude_trajectory = spin_amplitude_trajectory.cpu()
    spin_plot_data = 2 * (spin_amplitude_trajectory > 0) - 1
    energy_plot_data = energy_plot_data.cpu()
    error_var_data = error_var_data.cpu()
    for k in range(batch_size):
        sig_opt[k,:] = spin_plot_data[k,:,t_opt.long()[k]]
    sig_opt = sig_opt.cpu()

    return (sig_opt.numpy(), spin_amplitude_trajectory.numpy(), t, energy_plot_data.numpy(), error_var_data.numpy())