#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 00:31:42 2021

@author: sylvain
"""

import numpy as np

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from SEM1D import SEM1D


if __name__ == "__main__":
    
    # Comparaison milieu periodique avec milieu homogenisé (moyenne arithmétique)
    
    Nx  = 150
    Np  = 4
    Nt  = 10000
    eps = 0.1
    rho0 = 3000
    G0  = 40e9
    
    H   = 10000
    mesh = np.linspace(0,H,Nx+1)
    
    f0 = 50
    
    rho = np.ones((Np+1)*Nx) * rho0
    G   = np.ones((Np+1)*Nx) * G0
    
    recep_list = [H//4, H//2, 3*H//4]
    
    path = os.path.dirname(os.path.realpath(__file__))
    
    simu = SEM1D(mesh, rho, G, eps, Nt, Np, "simple", path, f0=f0, src="ricker", recep_list=recep_list)
    print(simu)
    simu.run()
