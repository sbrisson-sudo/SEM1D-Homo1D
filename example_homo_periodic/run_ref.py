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

from SEM1D import SEM1D


# PARAMETRES COMMUNS A TOUTES LES SIMULATIONS

H       = 3000
total_time = 1
f0      = 25
recep_list = [H/3]
rho0    = 3000
G0      = 40e9
eps     = 0.1

# REFERENCE : milieu à hétérogeneité de petite échelle 

Nx  = 600
Np  = 3
mesh_detailed    = np.linspace(0,H,Nx+1)

# Construction milieu periodique, tel que lamnda_min > 2 * lamnda (1)

n = 2 # periodicité de l'hétérogénéité (en nombre d'éléments)

rho1 = np.array([0.9 if k%(n*(Np+1)) < n*(Np+1)//2 else 1.1 for k in range((Np+1)*Nx)]) * rho0
G1   = np.array([0.9 if k%(n*(Np+1)) < n*(Np+1)//2 else 1.1 for k in range((Np+1)*Nx)]) * G0


# On vérifie la condition (1)

lamnda_min = np.min(np.sqrt(G1/rho1)) / (3*f0) 
print("{:<35}{}".format("Longueur d'onde minimale",lamnda_min))
print("{:<35}{}".format("Periode spatiale du milieu",n*H/Nx))

path_output = f"{path}/data"

ref = SEM1D(mesh_detailed, rho1, G1, eps, Np, "ref" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", time=total_time, fr_save=100)
print(ref)
ref.save_medium_properties()
ref.run()

    
    
    
    
    
    
    
    

