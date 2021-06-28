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

from random import random


# PARAMETRES COMMUNS A TOUTES LES SIMULATIONS

H       = 3000
total_time = 1
f0      = 100
recep_list = [H/3]
rho0    = 3000
G0      = 40e9
eps     = 0.1

# REFERENCE : milieu à hétérogeneité de petite échelle non periodique
# On construit un milieu avec aléatoire (constante + bruit blanc), constant sur un element 

Nx  = 600
Np  = 3
mesh_detailed    = np.linspace(0,H,Nx+1)

# Construction milieu periodique, tel que lamnda_min > 2 * lamnda (1)

rho_random = np.ones((Np+1)*Nx) * rho0
G_random = np.ones((Np+1)*Nx) * G0

for i in range(Nx):
    rd_rho = random() * rho0 /10
    rd_G = random() * G0 /10
    for j in range(Np+1):
        rho_random[i*(Np+1) + j] += rd_rho
        G_random[i*(Np+1) + j] += rd_G
        
path_output = f"{path}/data"

ref = SEM1D(mesh_detailed, rho_random, G_random, eps, Np, "ref_random" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", time=total_time, fr_save=100)
print(ref)
ref.save_medium_properties()
ref.run()

    
    
    
    
    
    
    
    

