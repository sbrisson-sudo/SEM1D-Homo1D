#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:31:42 2021

@author: sylvain
"""

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from SEM1D import SEM1D, compute_total_continuous_space
from homo1D import Homo1D
from plot1Dnc import read_netcdf


if __name__ == "__main__":
    
    # Comparaison milieu periodique avec milieu homogenisé (moyenne arithmétique)
    
    Nx  = 200
    Np  = 4
    Nt  = 10000
    eps = 0.1
    rho0 = 3000
    G0  = 40e9
    
    H   = 10000
    mesh = np.linspace(0,H,Nx+1)
    
    # Milieu periodique
    
    rho1 = np.array([0.9 if k%20 < 10 else 1.1 for k in range((Np+1)*Nx)]) * rho0
    G1   = np.array([0.9 if k%20 < 10 else 1.1 for k in range((Np+1)*Nx)]) * G0
    
    SEM1D(mesh, rho1, G1, eps, Nt, Np, "data/ref.nc").run()
        
    # Moyenne arithmétique
    
    rho2 = np.ones((Np+1)*Nx) * rho0
    G2   = np.ones((Np+1)*Nx) * G0
    
    SEM1D(mesh, rho2, G2, eps, Nt, Np, "data/homo1.nc").run()
    
    # 2-scale homogeneisation
    
    complete_mesh = compute_total_continuous_space(mesh, Np)
    lamnda_0 = 20*H/((Np+1)*Nx)
    
    homo = Homo1D(complete_mesh, complete_mesh, rho1, G1, lamnda_0 = lamnda_0)
    
    homo.run_homo()
    homo.plot()
    
    rho_star = homo.get_rho_star()
    G_star   = homo.get_mu_star()
    SEM1D(mesh, rho_star, G_star, eps, Nt, Np, "data/homo2.nc").run()
    
    # Comparaison
    
    u1,x,t = read_netcdf("data/ref.nc")
    u2,_,_ = read_netcdf("data/homo1.nc")
    u3,_,_ = read_netcdf("data/homo2.nc")
    
    t_idx = t.shape[0] -1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,u1[t_idx,:],label="ref")
    ax.plot(x,u2[t_idx,:],label="moyenne arithmétique")
    ax.plot(x,u3[t_idx,:],label="2-scale homogeneisation")
    ax.legend()
    plt.show()


    
    
    
    
    
    
    
    
    
    
    

