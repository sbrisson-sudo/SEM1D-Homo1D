#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:31:42 2021

@author: sylvain
"""

import numpy as np

import os, sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))

path_output = f"{path}/data"

from SEM1D import SEM1D, compute_total_continuous_space, compute_total_space
from homo1D import Homo1D

# PARAMETRES COMMUNS A TOUTES LES SIMULATIONS

H       = 3000
total_time = 1
f0      = 25
recep_list = [H/3]
rho0    = 3000
G0      = 40e9
eps     = 0.1
 

# 2-SCALE HOMOGENEISATION

Nx  = 600
Np  = 3
mesh_detailed    = np.linspace(0,H,Nx+1)

n = 2 # periodicité de l'hétérogénéité (en nombre d'éléments)
lambda_medium = n * H/Nx

rho1 = np.array([0.9 if k%(n*(Np+1)) < n*(Np+1)//2 else 1.1 for k in range((Np+1)*Nx)]) * rho0
G1   = np.array([0.9 if k%(n*(Np+1)) < n*(Np+1)//2 else 1.1 for k in range((Np+1)*Nx)]) * G0

complete_mesh = compute_total_continuous_space(mesh_detailed, Np)   
k_medium      = 1 / lambda_medium
k2 = 0.5 * k_medium
k1 = 0.5 * k2
fmax = 3*f0

homo = Homo1D(complete_mesh, rho1, G1, k1, k2, fmax)
print(homo)
homo.run_homo()
homo.save_properties(path_output)

# PLOT des propriétés homogénéisées

if False:
    homo.plot_rho()
    homo.plot_mu()
    homo.plot_wavelet()
    homo.plot_wavelet_freq()
    
# Run de SEM sur milieu homogénéisé

Nx  = 200
Np  = 4
Nt  = 12500
mesh2    = np.linspace(0,H,Nx+1)
output_mesh = compute_total_space(mesh2, Np)

rho_star = homo.get_rho_star(output_mesh)
G_star   = homo.get_mu_star(output_mesh)


homo_sem = SEM1D(mesh2, rho_star, G_star, eps, Np, "homo1D", path_output,f0=f0, src="ricker",  recep_list = recep_list, tscheme="newmark", Nt=Nt, fr_save=50)
print(homo_sem)
homo_sem.run()
homo_sem.save_medium_properties()


    
    
    
    
    
    
    
    
    

