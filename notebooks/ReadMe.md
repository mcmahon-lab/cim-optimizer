# CIM Optimizer Notebooks 

This directory contains notebooks which call functions from the [lib folder](../cim_optimizer) to showcase our Ising solvers.

**Example 1** introduces the concept of a coherent Ising machine (CIM), along with how to use our implemented simulators and observe results. We discuss loading problems, brute force solving, solving with the CIM, graphs of spin dynamics and energy evolution, and GPU acceleration.  

![Example 1 - Ising Energy Plot](/docs/gallery/example_1_ising_energy.png "Example 1 - Ising Energy Plot")

**Example 2** introduces hyperparameters and hyperparamter tuning with the CIM. We showcase the performance boost with well-selected hyperparameters, updated spin dynamics, and a Bayesian Optimization Hyperband package that allows for tuning of relevant hyperparameters across CAC, AHC, and external field AHC solvers.

![Example 2 - Spin Configuration Plot](/docs/gallery/example_2_spin_trajectories.png "Example 2 - Spin Configuration Plot")

**Example 3** introduces an external field problem (Maritime Inventory Routing Problem), and showcases the external field AHC solver along with hyperparmater tuning for this class of problems.