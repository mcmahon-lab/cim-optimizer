# Instances 

Several classes of Ising problem instances are provided with the repo, organized into different directories:

## `MC_Instances_NPZ`
Contains dense unweighted MAX-CUT instances, generated from Erdős–Rényi graphs under the G(n, p=0.5) model, based on examples from: 
> Hamerly, R., Inagaki, T., McMahon, P. L., Venturelli, D., Marandi, A., Onodera, T., Ng, E., Langrock, C., Inaba, K., Honjo, T., Enbutsu, K., Umeki, T., Kasahara, R., Utsunomiya, S., Kako, S., Kawarabayashi, K., Byer, R. L., Fejer, M. M., Mabuchi, H., … Yamamoto, Y. (2019). Experimental investigation of performance differences between coherent Ising machines and a quantum annealer. In Science Advances (Vol. 5, Issue 5). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/sciadv.aau0823
for MAX-CUT instances with less than 100 spins.

## `SK_Instances_NPZ`
Contains dense Sherrington-Kirkpatrick instances with elements $J_{ij} \in \{ -1, 1\}$, generated from Erdős–Rényi graphs under the G(n, p=0.5) model, based on examples from: 
> Hamerly, R., Inagaki, T., McMahon, P. L., Venturelli, D., Marandi, A., Onodera, T., Ng, E., Langrock, C., Inaba, K., Honjo, T., Enbutsu, K., Umeki, T., Kasahara, R., Utsunomiya, S., Kako, S., Kawarabayashi, K., Byer, R. L., Fejer, M. M., Mabuchi, H., … Yamamoto, Y. (2019). Experimental investigation of performance differences between coherent Ising machines and a quantum annealer. In Science Advances (Vol. 5, Issue 5). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/sciadv.aau0823
for SK1 instances with less than 100 spins. 

## `Stanford_Gset`
Stanford Gset results are sourced from Yinyu Ye's [personal website](http://web.stanford.edu/~yyye/yyye/Gset/). These consist of MAX-CUTs, weighted and unweighted, of sizes N=800 and greater. 

## `MIRP_TestSet`
Contains a class of Ising-formulated general vehicle routing problems, known as Maritime Inventory Routing Problems (MIRP). These instances were generated using with CPLEX code from Timothée Leleu based on the formulation in the following paper:
>Braekers, K., Ramaekers, K., & Van Nieuwenhuyse, I. (2016). The vehicle routing problem: State of the art classification and review. In Computers &amp; Industrial Engineering (Vol. 99, pp. 300–313). Elsevier BV. https://doi.org/10.1016/j.cie.2015.12.007

## `VRPTW_TestSet`
Contains Ising Problem formulation of Vehicle Routing Problems with Time Windows (VRPTW), based on examples from:
>S. Harwood, C. Gambella, D. Trenev, A. Simonetto, D. Bernal and D. Greenberg, "Formulating and Solving Routing Problems on Quantum Computers," in IEEE Transactions on Quantum Engineering, vol. 2, pp. 1-17, 2021, Art no. 3100118, doi: 10.1109/TQE.2021.3049230
