# Instances 

Several classes of Ising problem instances are provided with the repo, organized into different directories:

## `MC_Instances_NPZ`
Contains dense unweighted MAX-CUT instances, generated from Erdős–Rényi graphs under the G(n, p=0.5) model, based on examples from: 
> Hamerly, R., Inagaki, T., McMahon, P. L., Venturelli, D., Marandi, A., Onodera, T., Ng, E., Langrock, C., Inaba, K., Honjo, T., Enbutsu, K., Umeki, T., Kasahara, R., Utsunomiya, S., Kako, S., Kawarabayashi, K., Byer, R. L., Fejer, M. M., Mabuchi, H., … Yamamoto, Y. (2019). Experimental investigation of performance differences between coherent Ising machines and a quantum annealer. In Science Advances (Vol. 5, Issue 5). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/sciadv.aau0823

for MAX-CUT instances with <=100 spins. These instances (and their solutions) are stored in the .npz format, a compressed file format using gzip compression, readable with Python's numpy module.

## `SK_Instances_NPZ`
Contains dense Sherrington-Kirkpatrick instances with elements $J_{ij} \in \{ -1, 1\}$, generated from Erdős–Rényi graphs under the G(n, p=0.5) model, based on examples from: 
> Hamerly, R., Inagaki, T., McMahon, P. L., Venturelli, D., Marandi, A., Onodera, T., Ng, E., Langrock, C., Inaba, K., Honjo, T., Enbutsu, K., Umeki, T., Kasahara, R., Utsunomiya, S., Kako, S., Kawarabayashi, K., Byer, R. L., Fejer, M. M., Mabuchi, H., … Yamamoto, Y. (2019). Experimental investigation of performance differences between coherent Ising machines and a quantum annealer. In Science Advances (Vol. 5, Issue 5). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/sciadv.aau0823

for SK1 instances with <=100 spins. These instances (and their solutions) are stored in the .npz format, a compressed file format using gzip compression, readable with Python's numpy module.

## `Stanford_Gset`
Stanford Gset results are sourced from Yinyu Ye's [personal website](http://web.stanford.edu/~yyye/yyye/Gset/). These consist of MAX-CUTs, weighted and unweighted, of sizes N=800 and greater. These instances are stored in the .txt format, as delimited plain text files.

## `MIRP_TestSet`
Contains a class of Ising-formulated general vehicle routing problems, known as Maritime Inventory Routing Problems (MIRP). These instances were generated using with CPLEX code from Timothée Leleu based on the formulation in the following paper:
>Braekers, K., Ramaekers, K., & Van Nieuwenhuyse, I. (2016). The vehicle routing problem: State of the art classification and review. In Computers &amp; Industrial Engineering (Vol. 99, pp. 300–313). Elsevier BV. https://doi.org/10.1016/j.cie.2015.12.007

These instances are stored in the .rudy format, a form of delimited text file which can be ready by our helper function load_adjMatrix_from_rudy(). Each row of the file specifies a row, and column of the coupling matrix, as well as the corresponding element value. The solutions, files with the "_SOL" suffix, are delimited plain text files. Each solution file contains the ground-state Ising energy, as well as the zero-indexed spin numbers which should be spin-up (i.e. positive) to achieve this ground-state Ising energy given the coupling matrix and external field. All other spin numbers should be spin-down.

## `VRPTW_TestSet`
Contains Ising Problem formulation of Vehicle Routing Problems with Time Windows (VRPTW), based on examples from:
>S. Harwood, C. Gambella, D. Trenev, A. Simonetto, D. Bernal and D. Greenberg, "Formulating and Solving Routing Problems on Quantum Computers," in IEEE Transactions on Quantum Engineering, vol. 2, pp. 1-17, 2021, Art no. 3100118, doi: 10.1109/TQE.2021.3049230

These instances are stored in the .rudy format, a form of delimited text file which can be ready by our helper function load_adjMatrix_from_rudy(). Each row of the file specifies a row, and column of the coupling matrix, as well as the corresponding element value.

