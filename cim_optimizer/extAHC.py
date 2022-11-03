# Hyperband PyTorch Implementation of External-Field AHC Solver (with GPU-acceleration)
# Created September 2022
# Last Updated October 31st, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import torch
import numpy as np
import copy
torch.backends.cudnn.benchmark = True

def CIM_ext_f_AHC_GPU(time_stop, J, h, batch_size = 1, nsub = 1, dt=0.015625, F_h=2.0, alpha=1.0, beta=0.05, delta=0.25, eps=0.333, lambd=0.001, pi=-0.225, rho=1.0, tau= 100, noise=0, ahc_ext_nonlinearity = None, custom_fb_schedule=None, custom_pump_schedule=None, device=torch.device('cpu')):
    """ CIM solver with amplitude-heterogeneity correction and external-field.

    Attributes
    ----------
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
    sig_opt : NumPy array
        2-d NumPy array of batches x optimal spins where optimal spins
        are defined by the spin configuration that achieved the lowest
        energy up to that point.
    spin_amplitude_trajectory : NumPy array
        3-d NumPy array of batches x spins x time steps.
    t : int 
        Number of time steps: defined as int(time_stop/dt).
    energy_plot_data : NumPy array
        2-d NumPy array defined as batches x energy_achieved.
    error_var_data : NumPy array
        3-d NumPy array defined as batches x error variables x time steps.
    """

    # Hyperparameters are fitted to MIRP feasibility instances from Exxon-Mobil.

    #  Reference: 
    #  [1] T. Leleu, Y. Yamamoto, P. L. McMahon, and K. Aihara, Destabilization
    #      of local minima in analog spin systems by correction of amplitude
    #      heterogeneity. Phys Rev Lett 122, 040607 (2019). 
    #      https://doi.org/10.1103/PhysRevLett.122.040607



    #Compute instance sizes, cast Ising problem matrix to torch tensor.
    J = torch.from_numpy(J)
    J = J.float().to(device)
    h = torch.from_numpy(h)
    h = h.float().to(device)
    N = J.size()[1]
    if h is None:
        h = torch.zeros(batch_size, N).float().to(device)
    #Initialize plot and runtime data arrays.
    energy_plot_data = torch.zeros(batch_size, time_stop).to(device)
    error_var_data = torch.zeros(batch_size, N, time_stop).to(device)
    spin_amplitude_trajectory = torch.zeros(batch_size, N, time_stop).to(device)
    # Constants, Commented Values are Default Values set by Hyperband Benchmarking on Generated MIRP Instances.
    MAX_FLOAT = 500
    EMAX_FLOAT = 32
    #Initialize Spin-Amplitude Vectors and Auxiliary Variables
    theta0 = (2*(torch.rand(batch_size,N) - 0.5)).to(device)
    x = (0.14*(torch.cos(2*torch.pi*theta0))).to(device)

    #Compute initial Ising energy and spin states.
    sig = ((2 * (x > 0) - 1).float()).to(device)
    mu = x @ J
    h_ = sig @ J

    H = (-1/2*(torch.bmm(sig.view(batch_size, 1, N), (sig @ J).view(batch_size, N, 1)))[:,:,0]).view(batch_size) - sig @ h#
    e = (0.9*torch.ones(batch_size,N) + 0.6*torch.rand(batch_size,N)).to(device)
    g = (0.9*torch.ones(batch_size,N) + 0.6*torch.rand(batch_size,N)).to(device)

    ic = torch.ones(batch_size).to(device)
    beta = torch.zeros(batch_size).to(device)
    p = pi*torch.ones(batch_size).to(device)
    a = alpha*torch.ones(batch_size).to(device)
    #Configure ramp schedules.
    if custom_fb_schedule is None:
        eps = (torch.ones(time_stop)*eps).to(device)
    else:
        eps = custom_fb_schedule((torch.arange(0, time_stop, dt))).to(device)
    if custom_pump_schedule is None:
        pi = (torch.ones(time_stop)*pi).to(device)
    else:
        pi = custom_pump_schedule(torch.arange(0, time_stop)).to(device)
        rho = 0
    if ahc_ext_nonlinearity is None:
        ahc_ext_nonlinearity = lambda c : torch.pow(c,3)
    # Optimal energy, Sim. Time of Opt. Energy, and Opt. Spin Configuration
    H_opt = H
    sig_opt = sig
    t_opt = torch.zeros(batch_size).float().to(device)
    Jm = ((abs(J) > 0).int()).to(device)
    norm_ = torch.ones(N).to(device)
    norm = (norm_.float() @ Jm.float()).to(device)
    # Time-keeping variables
    dt_sub = dt/nsub
    effective_tau = tau/dt
    #Simulate Time-Evolution of Spin Amplitudes
    for t in range(time_stop):
        #Save spin states at current iteration.
        spin_amplitude_trajectory[:, :, t] = x
        error_var_data[:, :, t] = e
        x_prev = copy.deepcopy(x)
        e_prev = copy.deepcopy(e)
        g_prev = copy.deepcopy(g)
        
        for z in range(nsub):
            x_squared = x**2

            dxdt = (p-1)[:,None]*x - ahc_ext_nonlinearity(x) 
            dxdt += noise * (torch.rand(batch_size, N, device=device)-0.5) 
            dedt = -beta[:, None]*(x_squared - a[:, None])*e_prev
            g_ = 5*g_prev 
            xmax = torch.sqrt(a/2)
            theta = torch.log(torch.cosh(g_*x_prev))/g_
            theta += torch.abs(xmax)[:, None]
            theta += - torch.log(torch.cosh(g_*xmax[:, None]))/g_

            mx = F_h*(theta @ Jm.float())/norm
            dtheta = x_prev**2 - a[:, None]/2
            dgdt = beta[:, None]*(dtheta*g_prev)
            #Euler step for equations of motion of spin amplitudes.
            x = x + dxdt*dt_sub
            #Euler step for equations of motion of error variables.
            e = e + dedt*dt_sub
            g = g + dgdt*dt_sub

            #Normalize auxiliary error variables.
            g[g > EMAX_FLOAT] = EMAX_FLOAT
            e[e > EMAX_FLOAT] = EMAX_FLOAT

            e = torch.abs(e)
            g = torch.abs(g)

        # Compute Ising Coupling Off-Diagonal
        x = x + (eps[t]*dt)*e_prev*(x_prev @ J)
        # Compute Zeeman Term Feedback
        x = x + (eps[t]*dt)*e_prev*mx*h/2
        # Compute Binary form of X
        sig = ((2 * (x > 0) - 1).float())

        # Compute h Prime, which is J dotted with the spin signs.
        h_ = sig @ J

        #Save current Ising energy.
        H = (-1/2*(torch.bmm(sig.view(batch_size, 1, N), (h_).view(batch_size, N, 1)))[:,:,0]).view(batch_size) - sig @ h #
        energy_plot_data[:, t] = H

        # Difference in energy (between current state and best state found so far)
        dH = (H-H_opt)

        #Modulate target amplitude, error variable rate of change parameters depending on Ising energy.
        a = alpha + rho*torch.tanh(delta*dH)
        # Linear Gain Modulation
        p = pi[t] - rho*torch.tanh(delta*dH)
        # Increment Beta, Error Variable Rate of Change
        beta = beta + lambd*dt
        #Use boolean-array indexing to update ramp schedules, beta, and minimum Ising energy.
        ic[ic < t - effective_tau] = t
        beta[ic < t - effective_tau] = 0
        t_opt[H < H_opt] = t
        ic[H < H_opt] = t
        H_opt = torch.minimum(H_opt, H) 
    #Parse and Return Solutions
    H_opt = H_opt.cpu()
    spin_amplitude_trajectory = spin_amplitude_trajectory.cpu()
    t = time_stop
    spin_plot_data = 2 * (spin_amplitude_trajectory > 0) - 1
    for k in range(batch_size):
        sig_opt[k,:] = spin_plot_data[k,:,t_opt.long()[k]]
    energy_plot_data = energy_plot_data.cpu()
    sig_opt = sig_opt.cpu()
    error_var_data = error_var_data.cpu()
    return (sig_opt.numpy(), spin_amplitude_trajectory.numpy(), t, energy_plot_data.numpy(), error_var_data.numpy())