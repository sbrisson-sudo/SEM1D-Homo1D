#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:31:42 2021

@author: sylvain brisson
"""

import numpy as np

import os, sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(path))

from SEM1D import SEM1D



print(f"Looking for the Sem1D package in {os.path.dirname(path)}")

# PARAMETRES COMMUNS A TOUTES LES SIMULATIONS

H       = 3000
total_time = 1
f0      = 25
recep_list = [H/3]
rho0    = 3000
G0      = 40e9
eps     = 0.1

# HOMOGENEISATION NAIVE : moyenne arithmétique

Nx  = 200
Np  = 4
Nt  = 12500
mesh_simple    = np.linspace(0,H,Nx+1)

rho2 = np.ones((Np+1)*Nx) * rho0
G2   = np.ones((Np+1)*Nx) * G0

path_output = f"{path}/data"

mean_arith = SEM1D(mesh_simple, rho2, G2, eps, Np, "mean_arith" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", Nt=Nt, fr_save=50)
print(mean_arith)    
mean_arith.run()

# HOMOGENEISATION MILIEU PERIODIQUE (solution analytique)

rho3 = np.ones((Np+1)*Nx) * rho0
G3   = np.ones((Np+1)*Nx) * 1/ (1/(2*0.9*G0) + 1/(2*1.1*G0))

mean_geo = SEM1D(mesh_simple, rho3, G3, eps, Np, "mean_geo" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", Nt=Nt, fr_save=50)
print(mean_geo)    
mean_geo.run()

 