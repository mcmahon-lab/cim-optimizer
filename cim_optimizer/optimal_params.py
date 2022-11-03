# Optimal hyperparameters for well-defined instances including MAX-CUT, SK, and GSet
# Created October 2022
# Last Updated October 27th, 2022
# (C) Copyright The Contributors 2022
# Contributors: /CITATION.cff
#
# This code is licensed under Creative Commons Attribution 4.0 International.
# https://creativecommons.org/licenses/by/4.0/
import numpy as np

def sk_50_rand_params():
    """hyperparameters for random sk graph with size N=50"""
    sk_50_rand_dic = {
        "num_runs" : 1,
        "num_timesteps_per_run" : 2500,
        "cac_time_step" : 0.04,
        "cac_r" : 0.3,
        "cac_alpha" : 0.7,
        "cac_beta" : 0.25,
        "cac_gamma" : 0.010,
        "cac_delta" : 12,
        "cac_mu" : 0.8,
        "cac_rho" : 1.2,
        "cac_tau" : 150
    }
    return sk_50_rand_dic

def sk_100_fc_params():
    """hyperparameters for fully connected sk graph with size N=100"""
    sk_100_fc_dic = {
        "num_runs" : 1,
        "num_timesteps_per_run" : 3000,
        "cac_time_step" : 0.05,
        "cac_r" : -0.2,
        "cac_alpha" : 0.7,
        "cac_beta" : 0.3,
        "cac_gamma" : 0.010,
        "cac_delta" : 12,
        "cac_mu" : 0.8,
        "cac_rho" : 1.2,
        "cac_tau" : 150
    } 
    return sk_100_fc_dic

def maxcut_100_params():
    """hyperparameters for 50% edge density MAX-CUT instance size N=100"""
    maxcut_100_dic = {
        "num_runs" : 1,
        "num_timesteps_per_run" : 2500,
        "cac_time_step" : 0.04,
        "cac_r" : -0.3,
        "cac_alpha" : 0.7,
        "cac_beta" : 0.25,
        "cac_gamma" : 0.010,
        "cac_delta" : 12,
        "cac_mu" : 0.8,
        "cac_rho" : 1.2,
        "cac_tau" : 150
    } 
    return maxcut_100_dic

def maxcut_200_params():
    """hyperparameters for 50% edge density MAX-CUT instance size N=200"""
    # same as MAX-CUT instance size N=100
    return maxcut_100_params()

def maxcut_500_params():
    """hyperparameters for 50% edge density MAX-CUT instance size N=500"""
    maxcut_500_dic = {
        "num_runs" : 1,
        "num_timesteps_per_run" : 25000,
        "cac_time_step" : 0.02,
        "cac_r" : 0.9,
        "cac_alpha" : 1.1,
        "cac_beta" : 0.35,
        "cac_gamma" : 0.0005,
        "cac_delta" : 15,
        "cac_mu" : 0.7,
        "cac_rho" : 1,
        "cac_tau" : 200
    } 

def G1_params(J):
    """hyperparameters for Stanford GSet problem 1 (G1)"""
    N = J.shape[0] 
    G1_dic = {
        "num_runs" : 1,
        "num_timesteps_per_run" : 200000,
        "cac_r" : 0.2,
        "cac_alpha" : 1.0,
        "cac_beta" : 3*N/(np.sum(np.abs(J))),
        "cac_gamma" : 0.075/N,
        "cac_delta" : 9,
        "cac_rho" : 1,
        "cac_tau" : 9*N
    }
    return G1_dic

def G2_params(J):
    """hyperparameters for Stanford GSet problem 2 (G2)"""
    N = J.shape[0] 
    G2_dic = {
        "num_runs" : 1, 
        "num_timesteps_per_run" : 200000, 
        "cac_r" : 0.2, 
        "cac_alpha" : 1.0, 
        "cac_beta" : 3*N/(np.sum(np.abs(J))), 
        "cac_gamma" : 0.075/N, 
        "cac_delta" : 7,
        "cac_mu" : 1, 
        "cac_rho" : 1,
        "cac_tau" : 7*N 
    }
    return G2_dic

def G42_params(J):
    """hyperparameters for Stanford GSet problem 42 (G42)"""
    N = J.shape[0]
    G42_dic = {
        "num_runs" : 1, 
        "num_timesteps_per_run" : 200000, 
        "cac_r" : 0.1, 
        "cac_alpha" : 1.0, 
        "cac_beta" : 3*N/(np.sum(np.abs(J))), 
        "cac_gamma" : 0.065/N, 
        "cac_delta" : 7,
        "cac_mu" : 1, 
        "cac_rho" : 1,
        "cac_tau" : 7*N 
    }
    return G42_dic
