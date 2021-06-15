#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 15 11:35:07 2021
@author: sylvain
"""

import numpy as np
import netCDF4 as nc
import os

class SEM1D:
    
    ''' SOLVEUR SEM DE L'EQUATION D'ONDE ELASTIQUE EN CISAILLEMENT 1D 
    
    ** INPUTS **
    - mesh  : vecteur de taille Nx+1 (Nx éléments) des limites des éléments (sur [x1,x2])
    - Np    : degré polynomial utilisé au sein des éléments (Np+1 poinst par éléments)
    - rho   : densité, vecteur de taille (Np+1)*Nx
    - mu    : module de cisaillement vecteur de taille (Np+1)*Nx
    - eps   : critère de CFL pour calculer dt
    - Nt    : nombre de pas de temps
    - file_out   : fichier enregistrement
    
    '''
    
    frame_rate_save = 20
    
    def __init__(self, mesh, rho, mu, eps, Nt, Np, file_out):
        
        self.Nx = mesh.shape[0] - 1
        self.Np = Np
        self.Nt = Nt
        self.src_idx = int((self.Nx*self.Np+1)/4)
        self.mesh = mesh
        self.rho = rho
        self.mu = mu
        self.eps = eps
        self.file_out = file_out
        
        self.get_xwl()
        self.init_computation()
        
    def __repr__(self):
        
        out = ""
        out += f"Nombre d'éléments:\t{self.Nx}\n"
        out += f"Ordre polynomial:\t{self.Np}\n"
        out += f"Nombre de points distincts:\t{self.Np*self.Nx+1}\n"
        out += f"Nombre de pas de temps:\t{self.Nt}\n"
        out += f"Intervalle de temps:\t{self.dt}s\n"
        out += f"Durée simulation:\t{self.Nt*self.dt}s\n"
        out += f"Fréquence centrale de la source:\t{self.f0}Hz\n"
        out += f"Fichier d'écriture':\t{self.file_out}\n"
        return out
        
    
    def run(self):
        
        self.init_output_file()
        self.run_computation()
        self.close_output_file()
        
    def init_computation(self):

        # Compute the space vector (ie position of the Np*Nx+1 studied points)
        self.compute_space()
        
        # Compute the jacobian and its inverse
        self.compute_jacobian()
        
        # Compute dt
        self.compute_dt()
        self.compute_time()

        # Compute connectivity matrix
        self.compute_connectivity_matrix()

        # Compute mass matrix
        self.compute_inv_mass_matrix()
            
        # Compute stiffness matrix
        self.compute_stiffness_matrix()
        
        # Compute source
        self.compute_source()
        
    def run_computation(self):
        
        Nx,Np,Nt = self.Nx,self.Np,self.Nt
        U_old   = np.zeros(Nx*Np+1)
        U       = np.zeros(Nx*Np+1)
        U_new   = np.zeros(Nx*Np+1)
        d2Ut    = np.zeros(Nx*Np+1)
        F       = np.zeros(Nx*Np+1)
        tho     = np.zeros(Nx*Np+1)
        
        for t_idx in range(Nt):
            
            F[self.src_idx] = self.source[t_idx]
            tho[-1] = (self.rho[-1]*self.mu[-1])**0.5*(U[-1]-U_old[-1]) /self.dt
            d2Ut = self.Minv @ (F - self.K @ U - tho)
                    
            U_new = 2*U - U_old + self.dt**2 * d2Ut
        
            U_old,U = U,U_new  
            
            if (t_idx % SEM1D.frame_rate_save) == 0:
                self.u_nc[t_idx//SEM1D.frame_rate_save,:] = np.float32(U)
                self.t_w[t_idx//SEM1D.frame_rate_save] = self.time[t_idx]
       
    def get_xwl(self):
        
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        data = np.loadtxt(f"xwh_gll/gll_{self.Np+1:02d}.tab")
        self.x = data[0,:]
        self.w = data[1,:]
        self.l = data[2:,:]
    
    def compute_space(self):
        
        Nx,Np = self.Nx,self.Np
        mesh = self.mesh
        space = np.zeros(Np*Nx+1)
        for e in range(Nx):
            alpha_e = (mesh[e+1] - mesh[e])/2
            beta_e  = (mesh[e+1] + mesh[e])/2
            for i in range(Np):
                space[e*Np+i] = alpha_e* self.x[i] + beta_e
        space[Np*Nx] = mesh[-1] 
        self.space = space
        
    def compute_time(self):
        
        self.time = np.linspace(0,self.Nt*self.dt,self.Nt)
                
    def compute_dt(self):
                
        c = np.sqrt(self.mu / self.rho)      
        self.dt = self.eps * np.min(np.diff(self.space)) / np.max(c)
        
    def compute_jacobian(self):
        
        self.Je     = np.diff(self.mesh) / 2
        self.invJe  = 1/self.Je
        
    def compute_connectivity_matrix(self):
        
        Nx,Np = self.Nx,self.Np
        Q = np.zeros(( Nx*(Np+1) , Nx*Np+1 ))
        for i in range(Nx*(Np +1)):
            Q[i,i -i//(Np+1)] = 1
        self.Qt = Q.transpose()
        self.Q  = Q
        
    def compute_inv_mass_matrix(self):
        
        Nx,Np = self.Nx,self.Np
        ML = np.zeros(( Nx*(Np+1),Nx*(Np+1) ))
        for e in range(Nx):
            for i in range(Np+1):
                ML[e*(Np+1)+i,e*(Np+1)+i] = self.w[i]*self.rho[e*(Np+1)+i]*self.Je[e]
        M = self.Qt @ ML @ self.Q
        self.Minv = np.linalg.inv(M)
        
    def compute_stiffness_matrix(self):
        
        Nx,Np = self.Nx,self.Np
        KL = np.zeros(( Nx*(Np+1),Nx*(Np+1) ))
        for e in range(Nx):
            for i in range(Np+1):
                for j in range(Np+1):
                    K = 0
                    for k in range(Np+1):
                        K += self.w[k]* self.mu[e]* self.l[j,k]* self.l[i,k]* self.invJe[e]
                    KL[e*(Np+1)+i,e*(Np+1)+j] = K
        self.K = self.Qt @ KL @ self.Q
        
        
    def compute_source(self):
        
        time = self.time
        f0 = 100
        t0 = 4/f0 
        self.source = -2*(time-t0)*f0**2 * np.exp(-1*(time-t0)**2*f0**2)
        self.time = time
        self.f0 = f0
        
    def init_output_file(self):
        
        ncfile  = nc.Dataset(self.file_out ,mode='w', format='NETCDF4_CLASSIC')
    
        ncfile.createDimension('x', self.space.shape[0])
        ncfile.createDimension('time', None)
    
        x_w   = ncfile.createVariable('x', np.float32, ('x',))
        x_w[:] = np.float32(self.space)
        
        self.t_w   = ncfile.createVariable('time', np.float32, ('time',))
        
        self.u_nc = ncfile.createVariable('u',np.float32,('time', "x"))
        
        self.ncfile = ncfile
        
    def close_output_file(self):

        self.ncfile.close()
        
def compute_total_continuous_space(mesh, Np):
    '''compute the position of all points in a mesh for GLL quadrature points with order Np, collocation points are slighly differents'''
    Nx      = mesh.shape[0] - 1
    x,_,_   = get_xwl(Np)   
    space   = np.zeros((Np+1)*Nx)
    for e in range(Nx):
        for i in range(Np+1):
            space[e*(Np+1)+i] = (mesh[e+1]-mesh[e])/2* x[i] + (mesh[e+1]+mesh[e])/2
        eps = (mesh[e+1]-mesh[e])/100 # différence par rapport au collocation point
        space[e*(Np+1)]     += eps
        space[(e+1)*Np] -= eps
    return space

    
def get_xwl(Np):
    '''get the points, weights of the GLL quadrature, and th ederivative of the lagrangian interpolants over these points'''
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data = np.loadtxt(f"xwh_gll/gll_{Np+1:02d}.tab")
    x = data[0,:]
    w = data[1,:]
    l = data[2:,:]
    return x,w,l
        

if __name__ == "__main__":
    
    
    Nx  = 200
    Np  = 4
    Nt  = 10000
    eps = 0.1
    rho0 = 3000
    E0  = 40e9
    
    H   = 10000
    mesh = np.linspace(0,H,Nx+1)

    rho = np.ones((Np+1)*Nx) * rho0
    E   = np.ones((Np+1)*Nx) * E0
        
    simu = SEM1D(mesh, rho, E, eps, Nt, Np, "data/test.nc")
    print(simu)
    simu.run()
    
    
    