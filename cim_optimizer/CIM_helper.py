#Helper Functions for the CIM Reference Functions
#McMahon Lab
#Last Updated: September 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/

import copy
import torch
import numpy as np
import csv

def load_adjMatrix_from_rudy(filepathstr, delimiter='\t', index_start=0, preset_size=1, start_at_first_row=False):
    """Loads a coupling matrix J and external-field bias vector h from a delimited file.
    
    Arguments
    ----------
    filepathstr : str
        Path name for the instance to be loaded.
    delimiter : optional, str, default=0
        The delimiter between row/column co-ordinates in the J-matrix, as well as for the element value given the row-column co-ordinates.
    index_start : optional, int, default=0
        Option to specify whether the delimited file uses zero-indexed.
    preset_size : optional, int, default=1
        Option to specify the dimension N of the J-matrix and h-vector
    start_at_first_row : optional, bool, default=False
        Some delimited files contain a first row specifying the size of the J-matrix, N. If set to False (default), the first row will therefore not be read as an element in the Ising Hamiltonian.
    
    Returns
    --------
    J : ndarray
        Ising spin-spin coupling matrix
    h : ndarray
        External-field vector (individual weights)
    """

    with open(filepathstr, 'r') as f:
        read_array = csv.reader(f)
        if not start_at_first_row:
            row1 = next(read_array)
            matrix_size = int(row1[0].split()[0])
        if preset_size != 1:
            matrix_size = preset_size
        read_matrix = np.zeros((matrix_size, matrix_size))
        read_vector = np.zeros(matrix_size)
        for row in read_array:
            coord = list(map(float, (row[0].split(delimiter))))
            if int(coord[0]-index_start) == int(coord[1]-index_start):
                read_vector[int(coord[0]-index_start)] = float(coord[2])
            else:
                read_matrix[int(coord[0])-index_start,
                            int(coord[1])-index_start] = float(coord[2])
        final_matrix = read_matrix.T + read_matrix
    return final_matrix, read_vector

def get_binary_list(num, length):
    """Intermediate helper function for brute_force()."""
    bit_string = "{0:b}".format(num)
    num_str_len = len(bit_string)
    num_list = np.zeros(length)
    if num == 0:
        return num_list
    filled_string = (length - num_str_len) * '0' + bit_string
    return list(map(int, filled_string))

def brute_force(J, h=None):             
    """Basic brute-force solver for confirming ground state energy of smaller Ising problems."""
    N = J.shape[0]
    if h is None:
        h = np.zeros(N)
    min_energy = 1e20
    for z in range(2**N):
        spins = np.array(get_binary_list(z, N))
        spins = 2 * (spins > 0) - 1
        ising_energy = -1/2*(J.dot(spins)).dot(spins) - h.dot(spins)

        if ising_energy < min_energy:
            min_energy = ising_energy
            opt_spins = copy.deepcopy(spins)
    return opt_spins, min_energy
        