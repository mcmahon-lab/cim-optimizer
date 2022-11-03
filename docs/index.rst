.. cim-optimizer-local documentation master file, created by
   sphinx-quickstart on Tue Nov  1 17:16:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html

cim-optimizer
=============


This repository contains a reference implementation of a simulator of the Coherent Ising Machine (CIM). The CIM was `developed <https://doi.org/10.1038/s42254-022-00440-8>`_ as a photonic machine for heuristically solving Ising-optimization problems. Simulations of the CIM can be thought of as an unconventional, dynamical-systems-based, heuristic algorithm for solving Ising problems, which can compete with more conventional Ising-solving algorithms (such as simulated annealing, parallel tempering, and branch-and-bound).

There are two main intended audiences for this repository:


#. People who would like to use a state-of-the-art implementation of the CIM algorithm to heuristically solve Ising problems (for example, to benchmark the CIM approach against other heuristic approaches).
#. People who would like to study the workings of the CIM approach through simulation, and/or would like to have a quantitative model of CIM performance to make predictions about how future CIM hardware implementations will perform.

Most of the code in this repository resides within a Python file ``solve_Ising.py``. This repository is written in Python, and all input and result data for users is formatted in NumPy, while the source code uses PyTorch libraries for GPU acceleration. Several demonstration notebooks are provided in the Examples subsection, and showcase how to configure and run the solver function.

The goal of the CIM (and its simulation and variants) is to heuristically optimize the following :math:`N`-variable objective function (the classical :math:`N`-spin Ising Hamiltonian):

.. math::
   H = -\sum_{1\leq i < j \leq N} J_{ij}\sigma_i \sigma_j - \sum_{1 \leq i \leq N} h_i \sigma_i

where :math:`J` is an :math:`N \times N` coupling matrix and :math:`h` is an :math:`N`-dimensional vector representing the Zeeman external field. Each Ising spin is represented as :math:`\sigma_i \in \{ -1, 1\}`.

This repository uses solver algorithms adapted from:


* Discrete-Time Measurement-Feedback Coherent Ising Machine


     P.L. McMahon\ *, A. Marandi*\ , Y. Haribara, R. Hamerly, C. Langrock, S. Tamate, T. Inagaki, H. Takesue, S. Utsunomiya, K. Aihara, R.L. Byer, M.M. Fejer, H. Mabuchi, Y. Yamamoto. A fully programmable 100-spin coherent Ising machine with all-to-all connections. *Science* **354**\ , No. 6312, 614 - 617 (2016). https://doi.org/10.1126/science.aah5178


* Amplitude-Heterogeneity-Correction variant of the CIM algorithm


     T. Leleu, Y. Yamamoto, P.L. McMahon, and K. Aihara, Destabilization of local minima in analog spin systems by correction of amplitude heterogeneity. *Physical Review Letters* **122**\ , 040607 (2019). https://doi.org/10.1103/PhysRevLett.122.040607


* Chaotic-Amplitude-Control variant of the CIM algorithm


     T. Leleu, F. Khoyratee, T. Levi, R. Hamerly, T. Kohno, K. Aihara. Scaling advantage of chaotic amplitude control for high-performance combinatorial optimization. Commun Phys **4**\ , 266 (2021). https://doi.org/10.1038/s42005-021-00768-0


All of the algorithms implemented are for classical models of the CIM. See https://doi.org/10.1364/QIM.2017.QW3B.2 and https://doi.org/10.1117/12.2613817 for examples of discussions of quantum models of the CIM, which are not implemented in this repository.

Please see the references within the cited papers for a fuller picture of the history and development of the Coherent-Ising-Machine appraoch to heuristically solving Ising problems, which was begun at Stanford University in the group of Yoshihisa Yamamoto circa 2010.

Getting Started (The Short Version)
===================================

Installation
------------

.. code-block::

   pip install cim-optimizer

Usage
-----

.. code-block::

   from cim_optimizer.solve_Ising import *
   import numpy as np
   N = 20 # number of spins
   J = np.random.randint(-100,100,size=(N,N)) # spin-spin-coupling matrix of a random Ising instance
   h = np.random.randint(-100,100,size=(N)) # external-field vector of a random Ising instance
   solution = Ising(J, h).solve()
   print(solution)

Getting Started (The Longer Version)
====================================


* For background on CIMs, metadata, and GPU acceleration check out :doc:`Example 1<example_1>`.
* Hyperparameters and hyperparameter tuning (with BOHB) is showcased in :doc:`Example 2<example_2>`.
* An example solving the Maritime Inventory Routing Problem, which with the Ising mapping used, includes non-zero external-field terms: :doc:`Example 3<example_3>`.

Requirements
============


* Requires Python Version >= 3.7
* Requires PyTorch Version 1.12.1 to be compiled with CUDA 11.6 (for GPU acceleration). See Pytorch's `installation page <https://pytorch.org/>`_ for more information.
* Requires `BOHB-HPO Version 0.5.2 <https://pypi.org/project/BOHB-HPO/>`_ 
* For an exhaustive list of requirements, see the :doc:`requirements.txt<https://github.com/mcmahon-lab/cim-optimizer/blob/master/requirements.txt>` file.

Contributors
============

Francis Chen, Brian Isakov, Tyler King, Timothée Leleu, Peter McMahon, Tatsuhiro Onodera

Funding acknowledgements
========================

The development of this open-source implementation of CIM algorithm variants was partially supported by an NSF Expeditions award (CCF-1918549).

How to cite
===========

If you use this code in your research, please consider citing it. You can retrieve an APA or BibTeX citation by clicking 'Cite this repository' on the sidebar in GitHub, or you can view the raw citation data in `CITATION.cff <https://github.com/mcmahon-lab/cim-optimizer/blob/master/CITATION.cff>`_.

License
=======

The code in this repository is released under the following license:

`Creative Commons Attribution 4.0 International <https://creativecommons.org/licenses/by/4.0/>`_

A copy of this license is given in this repository as `license.txt <https://github.com/mcmahon-lab/cim-optimizer/blob/master/license.txt>`_.


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   install
   problem_description

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   example_1
   example_2
   example_3


.. toctree::
   :maxdepth: 4
   :caption: cim-optimizer API
   :hidden:

   code 
   solve_Ising
   AHC_tuning
   AHC
   CAC_tuning
   CAC
   extAHC_tuning
   extAHC
   optimal_params
   CIM_helper