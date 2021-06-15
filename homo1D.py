#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 15 11:35:07 2021
@author: sylvain
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Homo1D:
    
    ''' SOLVEUR 1D PROBLEME D'HOMOGENEISATION MILIEU 1D SOUMIS A UN CISAILLEMENT
    
    **INPUTS**
        - mesh_out  : ensemble des points où calculer les propriétés effectives [x1,x2]
        - mesh_in   : ensemble des points où sont données les proprités mécaniques, doit comprendre mesh_out
        - rho       : densité, donnée sur mesh_in
        - mu        : module de cisaillement, donné sur mesh_in
        
    et : 
        - lamnda_0  : longueur d'onde caractéristique des petites échelles
    ou : 
        - fmax      : fréquence max de la source
        - eps_0     : paramètre adimentionel des petites échelles par rapport à lamnda_min
        
    **OUTPUTS**
        - rho_star  : densité effective, donnée sur mesh_out
        - mu_star   : mmodule de cisaillement effectif, donné sur mesh_out
    '''
    
    Nx = 100000 # discrétisation intermédiaire
        
    def __init__(self, mesh_out, mesh_in, rho, mu, *args, **kwargs):
        
        self.mesh_in    = mesh_in
        self.mesh_out   = mesh_out
        self.rho_in     = rho
        self.mu_in      = mu
        
        self.lamnda_0   = kwargs.get('lamnda_0',0)
        self.eps_0      = kwargs.get('eps_0',0)
        self.fmax       = kwargs.get('fmax',0)
        
        if (self.lamnda_0 == 0) and ((self.eps_0 == 0) or (self.fmax == 0)):
            print("Usage :\thomo = Homo1D(mesh_out, mesh_in, rho, mu, lamnda_0 = lamnda_0)\nor\t\thomo = Homo1D(mesh_out, mesh_in, rho, mu, eps_0 = eps_0, fmax = fmax)")
        
    def run_homo(self):
        
        if self.eps_0 != 0:
            self.compute_lamnda_0()
        self.define_filter()
        self.interp_in2inter()
        self.compute_effective_medium()
        self.interp_inter2out()

    def compute_lamnda_0(self):
        ''' compute such as eps_0 = lamnda_0/lamnda_min ; lamnda_min = beta_min / f_max'''
        c = np.sqrt(self.mu_in / self.rho_in)
        lamnda_min = c.min() / self.fmax
        self.lamnda_0 = self.eps_0 * lamnda_min        
        
    def define_filter(self):
        '''define the filter to use'''
        filt_name = "butter"
        
        if filt_name == "butter":
            order   = 5 # ordre polynomial du filtre
            cuttof  = abs(2*(self.mesh_out[-1]-self.mesh_out[0])/(self.lamnda_0*Homo1D.Nx)) # cuttof
            self.filt = signal.butter(order, cuttof)
            
        elif filt_name == "gaussian":
            print("Not implemented yet")
            exit(0)
        else:
            print("Unrecognized filter name")
            exit(0)
            
    def interp_in2inter(self):
        ''' interpolate the properties on a linear space'''
        mesh_inter      = np.linspace(self.mesh_out[1], self.mesh_out[-1], Homo1D.Nx)
        self.rho_inter  = np.interp(mesh_inter, self.mesh_in, self.rho_in)
        self.mu_inter   = np.interp(mesh_inter, self.mesh_in, self.mu_in)
        self.mesh_inter = mesh_inter
    
    def compute_effective_medium(self):

        b,a = self.filt # discrete filter representation
        self.rho_star_inter = signal.filtfilt(b,a,self.rho_inter)
        self.mu_star_inter  = 1 / signal.filtfilt(b,a,1/self.mu_inter)
        
        
    def interp_inter2out(self):
        '''Interpolat form the intermediate linear space to mesh_out'''
        self.rho_star = np.interp(self.mesh_out, self.mesh_inter, self.rho_star_inter)
        self.mu_star  = np.interp(self.mesh_out, self.mesh_inter, self.mu_star_inter)
        
    def get_rho_star(self):
        return self.rho_star
    def get_mu_star(self):
        return self.mu_star
    
    def plot(self):
        """Plot the fields before and after homogeneisation"""
        plt.subplot(211)
        plt.plot(self.mesh_inter, self.rho_inter,'g',label="density")
        plt.plot(self.mesh_inter, self.rho_star_inter,'b', label="effective density")
        plt.legend()
        plt.subplot(212)
        plt.plot(self.mesh_inter, self.mu_inter,'g', label="shear modulus")
        plt.plot(self.mesh_inter, self.mu_star_inter,'b', label="effective shear modulus")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":

    rho0 = 3000
    G0  = 40e9
    fmax = 10
    
    Nx  = 200       # nb d'éléments
    Np  = 4         # degré polynomial par élément
    H   = 10000     
    mesh = np.linspace(0,H,(Np+1)*Nx)
    
    eps_0 = 0.125
    
    # Milieu homogène avec pertubation aléatoire

    rho = np.ones((Np+1)*Nx) * rho0 * (np.random.random((Np+1)*Nx)/10 +1)
    G   = np.ones((Np+1)*Nx) * G0 * (np.random.random((Np+1)*Nx)/10 +1)
        
    homo = Homo1D(mesh, mesh, rho, G, eps_0, fmax)
    homo.run_homo()
    homo.plot()
    
    
    
    

