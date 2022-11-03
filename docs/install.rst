Installation and Downloads
==========================

cim-optimizer requires Python version 3.7, 3.8, 3.9, or 3.10. Installation of cim-optimizer can be done via pip:

.. code-block::

   pip install cim-optimizer

Compiling from source
----------------------

cim-optimizer has the following dependencies:


* `Python <http://python.org/>`_ >= 3.7
* `NumPy <http://numpy.org/>`_  >= 1.19.2 (>= 1.22.3 for GitHub Actions pytests)
* `matplotlib <https://matplotlib.org/>`_ >= 3.5.2
* `Torch <https://pytorch.org/>`_ >= 1.12.1 (and additionally CUDA 11.6 for GPU acceleration)

To compile the latest developments, clone the git repository and install using pip in development mode:

.. code-block::

   git clone https://github.com/mcmahon-lab/cim-optimizer.git
   cd cim_optimizer && python -m install -e .

PyTorch Support
---------------

To use cim-optimizer with PyTorch using CPU, it is sufficient to run

.. code-block::

   pip install torch

However, if GPU acceleration is desired, it is expected to instead run 

.. code-block::

   pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

Please refer to `PyTorch <https://pytorch.org/>`_\ 's website for additional information.

Software Tests
--------------

To confirm the accuracy of cim-optimizer's performance, users can locally run the test suite via pytest. To perform this, pytest must be installed via:

.. code-block::

   pip install pytest