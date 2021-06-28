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
f0      = 100
recep_list = [H/3]
rho0    = 3000
G0      = 40e9
eps     = 0.1

Nx  = 200
Np  = 4
Nt  = 12500
mesh_simple    = np.linspace(0,H,Nx+1)

# chargement des propriétés materielles

path_output = f"{path}/data"

try: 
    rho = np.loadtxt(f"{path_output}/density_ref_random.asc", skiprows=2, delimiter='\t')[:,1]
    G   = np.loadtxt(f"{path_output}/shear-modulus_ref_random.asc", skiprows=2, delimiter='\t')[:,1]
except:
    print(f"ERROR: density_ref_random.asc or shear-modulus_ref_random.asc not found in {path_output}\n\t-> you have to run the reference first and update the path and the file names if you have changed them")
    exit(1)
    
# HOMOGENEISATION NAIVE : moyenne arithmétique

rho2 = np.ones((Np+1)*Nx) * rho.mean()
G2   = np.ones((Np+1)*Nx) * G.mean()

mean_arith = SEM1D(mesh_simple, rho2, G2, eps, Np, "mean_arith_rd" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", Nt=Nt, fr_save=50)
print(mean_arith) 
mean_arith.save_medium_properties()   
mean_arith.run()

# HOMOGENEISATION NAIVE 2 : moyenne harmonique pour le coefficient de cisaillemenrt milieu non periodique

def hmean(A):
    return A.shape[0] * 1/((1/A).sum())

rho3 = rho2
G3   = np.ones((Np+1)*Nx) * hmean(G)

mean_geo = SEM1D(mesh_simple, rho3, G3, eps, Np, "mean_geo_rd" ,path_output, f0=f0, src="ricker", recep_list = recep_list, tscheme="newmark", Nt=Nt, fr_save=50)
print(mean_geo)   
mean_geo.save_medium_properties() 
mean_geo.run()

 