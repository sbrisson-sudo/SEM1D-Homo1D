#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:02:19 2021
@author: Sylvain Brisson, département de géosciences, ENS de Paris
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from time import time,sleep
import sys
from scipy import interpolate
import matplotlib.animation as animation

def read_netcdf(ncfile):
    '''read the ncfile and return u,t,x'''
    data = nc.Dataset(ncfile)
    x   = data["x"][:].data
    t   = data["time"][:].data
    u   = data["u"][:].data
    return u,x,t


def plot_field(U,x,t,filenames):
    '''plot the fields u=f(x,t)'''
    
    T   = t.shape[0]
    dt  = t[-1]/T

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Simulation 1D propagation ondes S (SEM)")
    
    umax = U.max()
    try: ax.set_ylim([-umax,umax])
    except: pass

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
    
    running = True
    
    def onClick(event):
        nonlocal running
        if running:
            running = False
            plt.waitforbuttonpress(timeout=- 1)
            running = True
    
    fig.canvas.mpl_connect('key_press_event', onClick)

    while plt.fignum_exists(fig.number):
        
        if time()-t1 < t_update: sleep(t_update-(time()-t1))
        
        for i,line in enumerate(lines):       
            line.set_ydata(U[i,t_idx,:])
        timer.set_text('temps: '+str(int(t_idx*dt*100)/100)+'s')
        fig.canvas.draw()
        fig.canvas.flush_events()
        t_idx = (t_idx + i_update) % T
        t1 = time()


def interpolate_field(U,X,T):
    '''interpolate the fields over the same time vector t and space vector x'''
    
    # Valeurs extrêmes
    xmin = max([x.min() for x in X])
    xmax = min([x.max() for x in X])
    tmin = max([t.min() for t in T])
    tmax = min([t.max() for t in T])  
        
    # Construction des interpolations
    F = []
    for i in range(len(U)):
        # print(U[i].shape, T[i].shape, X[i].shape)
        F.append(interpolate.interp2d(X[i], T[i], U[i]))
        
    # Points d'interpolation
    N_space = 1000
    N_time  = 1000
    
    space_vec   = np.linspace(xmin,xmax,N_space)
    time_vec    = np.linspace(tmin,tmax,N_time)

    # Calculs des champs au points d'interpolation
    U2 = []
    for i in range(len(U)):
        U2.append(F[i](space_vec,time_vec))
        
    return U2, space_vec, time_vec

def plot_field_anim(U,x,t,filenames):
    
    T   = t.shape[0]
    dt  = t[-1]/T
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Simulation 1D propagation ondes S (SEM)")
    
    umax = U.max()
    try: ax.set_ylim([-umax,umax])
    except: pass

    ax.set_xlabel("profondeur (m)")
    ax.set_ylabel("déplacement")
    lines = []
    t_idx = 0
    for i,u in enumerate(U):   
        line, = ax.plot(x,u[t_idx,:],label=filenames[i])
        lines.append(line)
    timer = ax.text(0.7*ax.get_xlim()[1],0.5*ax.get_ylim()[0],'temps: 0s')
    ax.grid()
    plt.legend()
    plt.show()
    
    # t_update = 0.01     # frame lapse (s)
    i_update = 1      # number of time step between each frame
    
    def updatefig(*args):
        global t_idx, T, i_update
        t_idx = (t_idx + i_update) % T
        print("updating")
        for i,line in enumerate(lines):       
            line.set_ydata(U[i,t_idx,:])
        timer.set_text('temps: '+str(int(t_idx*dt*100)/100)+'s')
        return lines,timer,
    
    animation.FuncAnimation(fig, updatefig,  blit=True)
    plt.show()
        
    
def plot_traces(trace_files):
    
    data = []
    headers = []
    for file in trace_files:
        data.append(np.loadtxt(file, delimiter='\t', skiprows=1))
        with open(file, 'r') as file: 
            headers.append(file.readline().split('\t'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, file in enumerate(trace_files):
        for i in range(1,data[k].shape[1]):   
            ax.plot(data[k][:,0],data[k][:,i], label=f"{headers[k][i]} in {trace_files[k]}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (m)")
    ax.legend()
    ax.grid()
    plt.show()

def plot_source(source_file):
    
    data = np.loadtxt(source_file, delimiter='\t')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[:,0],data[:,1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("source acceleration ($ms^{-2}$)")
    ax.grid()
    plt.show()
    
def plot_properties(properties_files):
    
    data = []
    headers = []
    for file in properties_files:
        data.append(np.loadtxt(file, delimiter='\t', skiprows=1))
        with open(file, 'r') as file: 
            headers.append(file.readline().split('\t'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, file in enumerate(properties_files):
        for i in range(1,data[k].shape[1]):   
            ax.plot(data[k][:,0],data[k][:,i], label=f"{headers[k][i]} in {properties_files[k]}")
    ax.set_xlabel("space (m)")
    ax.legend()
    ax.grid()
    plt.show()

        
if __name__ == "__main__":
    
    help_message = "Usage: plotSEM.py\n\t--field file1.nc [file2.nc ...]\n\t--trace file.asc[file2.asc ...]\n\t--source file.asc\n\t--propertie file.asc [file2.asc ...]\nOne option maximum"
    
    if len(sys.argv) < 3:
        print(help_message)
        exit(1)
        
    if sys.argv[1] == "--field":
        
        filenames = sys.argv[2:]
        
        U,X,T = [],[],[]
        for file in filenames:
            u,x,t = read_netcdf(file)
            U.append(u)
            X.append(x)
            T.append(t)
            
        if any([u.shape != U[0].shape for u in U]):
            U,x,t = interpolate_field(U, X, T)
            
        plot_field(np.array(U),x,t,filenames)
        # plot_field_anim(np.array(U),x,t,filenames)

    
    elif sys.argv[1] == "--trace":
    
        plot_traces(sys.argv[2:])
        
    elif sys.argv[1] == "--propertie":
    
        plot_traces(sys.argv[2:])
        
    elif sys.argv[1] == "--source":
        
        plot_source(sys.argv[2])
        
        
    else:
        print(f"Unknown option {sys.argv[1]}\n{help_message}")
        


    

