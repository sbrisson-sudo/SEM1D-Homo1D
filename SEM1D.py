#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 15 11:35:07 2021
@author: Sylvain Brisson, département de géosciences, ENS de Paris
"""

import numpy as np
import netCDF4 as nc
from time import time
import os

class SEM1D:
    
    ''' SOLVEUR SEM DE L'EQUATION D'ONDE ELASTIQUE EN CISAILLEMENT 1D 
    
    ** UTILISATION**
    1). Initialisation
        simu = SEM1D(mesh, rho, G, eps, Np, "test_newmark", ".", time = 1)
    2). Execution
        simu.run()
    
    ** INPUTS **
    - mesh  : vecteur de taille Nx+1 (Nx éléments) des limites des éléments (sur [x1,x2])
    - rho   : densité, vecteur de taille (Np+1)*Nx
    - mu    : module de cisaillement vecteur de taille (Np+1)*Nx
    - eps   : critère de CFL pour calculer dt
    - Np    : degré polynomial utilisé au sein des éléments (Np+1 poinst par éléments)
    - name  : nom de la simulation (utilisé pour les noms de fichiers produits)
    - path  : chemin du répertoire d'enregistrement des données
    
    ** ARGUMENTS OPTIONNELS **
    - time  : temps total simulation, en secondes
    - Nt    : nombre de pas de temps total (un des arguments Nt ou time doit être specifié)
    - f0    : fréquence centrale de la source (default : 100Hz)
    - src   : nom de la source (default : ricker, seule source implémentée pour l'instant)
    - tscheme : time marching scheme : classic (différence finie centrée d'ordre 2) ou Newmark gamma=0.5 beta=0 (default : newmark)
    - fr_save : frame frequencie to save the wave fields (dafault : 30)
                                                
    ** FICHIERS DE SORTIES **
    - field_[name].nc   : champ d'onde au cours du temps (format netCDF)
    - trace_[name].asc  : trace sur la liste des positions de reception specifiée (format ASCII)
    - source_[name].asc : source utilisée (format ASCII)
    
    ** METHODES DISPONIBLES **
    - simu.run()    : lance la simulation
    - simu.save_medium_properties(path)     : enregistre les propriétés du milieu 
    - print(simu)   : affiche les paramètres de la simulation
    
    ** FONCTIONS AUXILLIAIRES**
    - compute_total_space(mesh,Np) : retourne un maillage contenant l'ensemble des points distincts (différent du maillage ne contenant que les limites entre éléments)
    - compute_total_continuous_space(mesh,Np) : retourne un maillage contenant l'ensemble des points y compris non distincts (en les rendants distints d'une quantité epsilon)
    
    '''
    
    def __init__(self, mesh, rho, mu, eps, Np, name_simu, path, *args, **kwargs):
        
        self.Nx     = mesh.shape[0] - 1
        self.Np     = Np
        self.src_idx = int((self.Nx*self.Np+1)/4)
        self.mesh   = mesh
        self.rho    = rho
        self.mu     = mu
        self.eps    = eps
        self.name_simu = name_simu
        self.path   = path
        
        self.t          = kwargs.get('time',False)
        self.Nt         = kwargs.get('Nt',False)
        
        if (not self.t) and (not self.Nt):
            raise Exception("You have to specify 'time' (total time) or 'Nt' (number of time steps.")
            exit(1)

        
        self.f0         = kwargs.get('f0',100)
        self.src_name   = kwargs.get('src', 'ricker')
        self.recep_list = kwargs.get('recep_list', [])
        self.tScheme    = kwargs.get('tscheme', 'classic')
        self.frame_rate_save  = kwargs.get('fr_save', 30)
                
        self.get_xwl()
        self.init_computation()
        
    def __repr__(self):
        
        out = ""
        out += "\n===== RUN SEM1D =====\n"
        out += "{:<35}{}\n".format("Nom de la simulation",self.name_simu)
        out += "{:<35}{}\n".format("Nombre d'éléments",self.Nx)
        out += "{:<35}{}\n".format("Ordre polynomial",self.Np)
        out += "{:<35}{}\n".format("Nombre de points distincts",self.Np*self.Nx+1)
        out += "{:<35}{}m\n".format("Taille du domaine",self.mesh[-1]-self.mesh[0])        
        out += "{:<35}{}\n".format("Nombre de pas de temps",self.Nt)
        out += "{:<35}{}s\n".format("Pas de temps",self.dt)        
        out += "{:<35}{}s\n".format("Durée simulation",self.t)
        out += "{:<35}{}Hz\n".format("Fréquence centrale de la source",self.f0)
        out += "{:<35}{}\n".format("Fonction source",self.src_name)
        out += "{:<35}{}\n".format("Répertoire d'écriture",self.path)
        out += "{:<35}{}\n".format("Fichiers d'écriture",f"source_{self.name_simu}.asc")
        out += "{:<35}{}\n".format("",f"traces_{self.name_simu}.asc")
        out += "{:<35}{}\n".format("",f"field_{self.name_simu}.nc")

        return out
        
    
    def run(self):
        
        t_begin = time()
        
        self.init_output_files()
    
        if self.tScheme == "classic":
            self.run_classic()
        elif self.tScheme == "newmark":
            self.run_newmark()
        else :
            raise Exception(f"time marching scheme {self.tScheme} non reconnu")
            exit(1)
            
        self.close_output_files()
        
        print("{:<40}{}\n".format("Temps total (calcul + écriture)",time()-t_begin))
        
    def init_computation(self):

        # Compute the space vector (ie position of the Np*Nx+1 studied points)
        self.compute_space()
        
        # Compute the jacobian and its inverse
        self.compute_jacobian()
        
        # Compute dt
        self.compute_dt()
        
        # Compute total_time / number of time steps
        self.compute_time()

        # Compute connectivity matrix
        self.compute_connectivity_matrix()

        # Compute mass matrix
        self.compute_inv_mass_matrix()
            
        # Compute stiffness matrix
        self.compute_stiffness_matrix()
        
        # Compute source
        self.compute_source()
        
    def run_classic(self):
        
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
            
            if self.recep_list:
                self.save_traces(t_idx, [U[x_idx] for x_idx in self.recep_pos])
            
            if (t_idx % self.frame_rate_save) == 0:
                self.save_field(t_idx,U)
                
    def run_newmark(self):
        
        Nx,Np,Nt = self.Nx,self.Np,self.Nt
        U_old   = np.zeros(Nx*Np+1)
        U       = np.zeros(Nx*Np+1)
        dU_old  = np.zeros(Nx*Np+1)
        dU      = np.zeros(Nx*Np+1)
        d2U_old = np.zeros(Nx*Np+1)
        d2U     = np.zeros(Nx*Np+1)
        F       = np.zeros(Nx*Np+1)
        tho     = np.zeros(Nx*Np+1)
        dt      = self.dt
        
        for t_idx in range(Nt):
            
            
            F[self.src_idx] = self.source[t_idx]
            tho[-1] = (self.rho[-1]*self.mu[-1])**0.5 * dU[-1]
            U = U_old + dt*dU_old + dt*dt/2*d2U_old
            d2U = self.Minv @ (F - self.K @ U - tho)
            dU = dU_old + dt/2*(d2U + d2U_old) 
        
            U_old, dU_old, d2U_old = U, dU, d2U 
            
            if self.recep_list:
                self.save_traces(t_idx, [U[x_idx] for x_idx in self.recep_pos])
            
            if (t_idx % self.frame_rate_save) == 0:
                self.save_field(t_idx,U)
       
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
                    
        if self.t:
            if self.Nt:
                raise Exception("Since 'time' is specified, 'Nt' is overridden")
            self.Nt   = int(self.t / self.dt)
        elif self.Nt:
            self.t = self.dt*self.Nt
        
        self.time_vec = np.linspace(0,self.Nt*self.dt,self.Nt)
                
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
        
        if self.src_name == "ricker":
            time_vec = self.time_vec
            f0 = self.f0
            t0 = 4/f0 
            self.source = (4*(time_vec-t0)**2*f0**4 - 2*f0**2) * np.exp(-1*((time_vec-t0)*f0)**2)
        else :
            raise Exception(f"source {self.src_name} non reconnue")
            exit(1)
        
        np.savetxt(f"{self.path}/source_{self.name_simu}.asc", np.vstack((self.time_vec,self.source)).T, delimiter="\t")
            
    def init_output_files(self):
        
        self.init_save_field()
        self.init_save_traces()
        
    def close_output_files(self):

        self.ncfile.close()
        if self.recep_list:
            header = "\t".join(["time (s)" ] + [f"trace at x={x}m" for x in self.recep_list])
            np.savetxt(f"{self.path}/traces_{self.name_simu}.asc", np.vstack((self.time_vec,self.traces)).T, delimiter="\t", header=header)
            
    def init_save_field(self):
        
        filename = f"{self.path}/field_{self.name_simu}.nc"
        ncfile  = nc.Dataset(filename ,mode='w', format='NETCDF4_CLASSIC')
        ncfile.createDimension('x', self.space.shape[0])
        ncfile.createDimension('time', None)
        x_w   = ncfile.createVariable('x', np.float32, ('x',))
        x_w[:] = np.float32(self.space)
        self.t_w   = ncfile.createVariable('time', np.float32, ('time',))
        self.u_nc = ncfile.createVariable('u',np.float32,('time', "x"))
        self.ncfile = ncfile
        
    def save_field(self,t_idx,U):
        
        self.u_nc[t_idx//self.frame_rate_save,:] = np.float32(U)
        self.t_w[t_idx//self.frame_rate_save] = self.time_vec[t_idx]

    def init_save_traces(self):
        
        if not self.recep_list:
            return
        
        self.recep_pos = [(np.abs(self.space - x)).argmin() for x in self.recep_list]
        self.traces = np.zeros((len(self.recep_list), self.Nt))
        
    def save_traces(self, t_idx, U):
        
        self.traces[:,t_idx] = U
        
    def save_medium_properties(self):
        
        total_space = compute_total_space(self.mesh, self.Np)
        np.savetxt(f"{self.path}/density_{self.name_simu}.asc", np.vstack((total_space, self.rho)).T, delimiter="\t", header="position(m)\tdensity(kg/m3)\n")
        np.savetxt(f"{self.path}/shear-modulus_{self.name_simu}.asc", np.vstack((total_space, self.mu)).T, delimiter="\t", header="position(m)\tshear modulus()\n")
        
        
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


def compute_total_space(mesh, Np):
    '''compute the position of all the points inside a mesh with a Np polynomlial order inside the elements (for GLL quaadrature points)'''
    Nx = mesh.shape[0] - 1
    x,_,_   = get_xwl(Np)   
    space   = np.zeros((Np+1)*Nx)
    for e in range(Nx):
        for i in range(Np+1):
            space[e*(Np+1)+i] = (mesh[e+1]-mesh[e])/2* x[i] + (mesh[e+1]+mesh[e])/2
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
    
    Nx  = 300
    Np  = 4
    total_time = 1.5
    eps = 0.1
    rho0 = 3000
    G0  = 40e9
    
    H   = 5000
    mesh = np.linspace(0,H,Nx+1)

    rho = np.ones((Np+1)*Nx) * rho0
    G   = np.ones((Np+1)*Nx) * G0
    
    recep = [H/8, H/4] # position des récepteurs
    
    f0 = 100
        
    simu = SEM1D(mesh, rho, G, eps, Np, "test_newmark", ".", time = total_time, f0=f0, src="ricker", recep_list=recep, tsheme="newmark")
    print(simu)
    simu.run()

    
    