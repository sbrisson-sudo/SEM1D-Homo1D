#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:02:19 2021
@author: sylvain
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from time import time,sleep
import sys

def read_netcdf(ncfile):
    '''read the ncfile and return u,t,x'''
    data = nc.Dataset(ncfile)
    x   = data["x"][:].data
    t   = data["time"][:].data
    u   = data["u"][:].data
    return u,x,t

def plot1D(U,x,t,filenames):
    '''plot the fields u=f(x,t)'''

    T = t.shape[0]
    dt = t[-1]/T

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Simulation 1D propagation ondes S (SEM)")
    
    umax = U.max()
    ax.set_ylim([-umax,umax])
    ax.set_xlabel("profondeur (m)")
    ax.set_ylabel("déplacement")
    lines = []
    for i,u in enumerate(U):   
        line, = ax.plot(x,u[0,:],label=filenames[i])
        lines.append(line)
    timer = ax.text(0.7*ax.get_xlim()[1],0.5*ax.get_ylim()[0],'temps: 0s')
    ax.grid()
    plt.legend()
    plt.show()
    
    t_update = 0.01     # frame lapse (s)
    i_update = 1      # number of time step between each frame
    t1 = time()
    t_idx = 0

    while plt.fignum_exists(fig.number):
        
        if time()-t1 < t_update: sleep(t_update-(time()-t1))
        
        for i,line in enumerate(lines):       
            line.set_ydata(U[i,t_idx,:])
        timer.set_text('temps: '+str(int(t_idx*dt*100)/100)+'s')
        fig.canvas.draw()
        fig.canvas.flush_events()
        t_idx = (t_idx + i_update) % T
        t1 = time()
        
        
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("usage: plot1Dnc.py file1.nc [file2.nc ...]")
        exit(1)
    
    filenames = sys.argv[1:]
    
    U = []
    for file in filenames:
        u,x,t = read_netcdf(file)
        U.append(u)
        
    plot1D(np.array(U),x,t,filenames)



    

