Introduction to the Coherent Ising Machine
===========================================

The Ising Problem is an NP-Hard combinatorial optimization problem commonly used as a benchmark for state-of-the art heuristic solvers. Often, these Ising problems also map to optimization problems found in widespread industrial applications. Each Ising problem is defined by an :math:`N \times N` coupling matrix :math:`\boldsymbol{J} \in \mathbb{R}^{N \times N}`, and may include an external field :math:`N`-vector :math:`\boldsymbol{\vec{h}}`. The problem involves finding :math:`N` different binary states :math:`\sigma_{i} \in \{+1, -1\}`, referred to as spins, to minimize the cost function known as the Ising Hamiltonian:

.. math::
    H(\boldsymbol{\vec{\sigma}}) = - \sum_{1\leq i<  j\leq N} J_{ij} \sigma_{i} \sigma_{j} - \sum_{1\leq i \leq N} h_{i} \sigma_{i}

In choosing the :math:`N`-vector of spins :math:`\boldsymbol{\vec{\sigma}}`, the global minimum cost function value of the Ising Hamiltonian is known as the ground-state energy, and the corresponding set of binary spins that. Several common methods exist within traditional computers; brute force, random sampling, Gibbs sampling, stochastic gradient descent, and Monte Carlo Markov Chain method can all potentially find the ground-state of a given Ising Hamiltonian. The measurement-feedback Coherent Ising Machine (CIM) is especially effective at solving Ising problems up to sizes of up to :math:`N=100,000` spins. 

This repository provides simulation tools for the classical analogue of the CIM, with several dynamical-systems inspired adaptations, as well as an option to use the simplified CIM equations of motions without the use of any auxiliary amplitude-control variables. The aforementioned CIM-solvers include the solvers proposed by Timothee Leleu, referred to as the Amplitude Heterogeneity Correction (AHC), Chaotic Amplitude Control (CAC), and external field-Amplitude Heterogeneity Correction (extAHC) solvers. Within each solver, the binary spins :math:`\boldsymbol{\vec{\sigma}}` are relaxed to analog spin amplitudes :math:`\boldsymbol{\vec{x}}` that represent the in-phase quadrature amplitudes of Optical Parametric Oscillator pulses within the experimental measurement-feedback CIM. The general systems of :math:`2N` equations of motion for each of these is as follows:

.. math::
    \frac{dx_{i}}{dt} = (p-1)x_{i} - \mu x_{i}^3 + \epsilon e_{i} \sum_{1\leq j\leq N} J_{ij} x_{j} 

.. math:: 
    \frac{de_{i}}{dt} = - \beta (x_{i}^2 - a)e_{i}


Here, :math:`p` and :math:`\mu` characterize the linear gain and the nonlinearity present within an OPO cavity, :math:`\epsilon` controls the feedback coupling coefficient, :math:`e_i` defines the error variables used for amplitude control, and :math:`\beta` controls the rate of change of the error variables in response to variances in spin amplitudes from the target amplitude :math:`\sqrt{a}`. 

In cases where an external field vector (also known as a Zeeman vector, or bias vector) :math:`\boldsymbol{\vec{h}}` is present in the Ising problem, the equations of motion of the external field-AHC solver include an additional feedback term:

.. math::
    \frac{dx_{i}}{dt} = (p-1)x_{i} - \mu x_{i}^3 + \epsilon e_{i} \sum_{1\leq j\leq N} J_{ij} x_{j} + F_h(\boldsymbol{\vec{x}}, \boldsymbol{J}, t) h_i

Here, :math:`F_{h}` is the external-field feedback coefficient, represented as a function of the current spin amplitudes, the norm of the coupling matrix, with a potential time-dependence on :math:`t`, the number of discrete iterations passed since the initialization of the spins amplitudes within a run of the CIM.