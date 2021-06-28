#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 15 11:35:07 2021
@author: Sylvain Brisson, département de géosciences, ENS de Paris
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Homo1D:
    
    ''' SOLVEUR 1D PROBLEME D'HOMOGENEISATION MILIEU 1D SOUMIS A UN CISAILLEMENT
    
    **INPUTS**
    - mesh      : ensemble des points où calculer les propriétés effectives [x1,x2]
    - rho       : densité, donnée sur mesh
    - mu        : module de cisaillement, donné sur mesh
    - k1        : nombre d'onde fin de gain égal à 1
    - k2        : nombde d'onde début de gain nul
    - fmax      : fréquence maximale de la source
    
    **ARGUMENST OPTIONNELS**
    - filter_name   : nom du filtre : gaussian ou heaviside (default : heaviside)
        rq : la marche du heaviside est un cosinus
    - filter_method : methode de filtrage : FFT ou convolution (dafault : FFT)
        
    **OUTPUTS**
        - propriétés effectives : données par les appels get_rho_star(mesh_out), get_mu_star(mesh_out) avec mesh_out le maillage de sortie désiré
    
    **METHODES AUTRES**
    - save_wavelet() : enregistre la wavelet utilisée (support spatial et spevctre fréquence)
    - save_properties() : enregistre les propriétés effectives calculées
    '''
    
    Nx = 1000000 # discrétisation intermédiaire
        
    def __init__(self, mesh, rho, mu, k1, k2, fmax, *args, **kwargs):
        
        self.mesh_in    = mesh
        self.rho_in     = rho
        self.mu_in      = mu
        self.k1         = k1
        self.k2         = k2
        self.fmax       = fmax
        
        self.filt_name      = kwargs.get('filter_name', 'heaviside')
        self.filt_method    = kwargs.get('filter_method', 'FFT')
        
        self.compute_eps_0()

        
    def __repr__(self):
        
        out = ""
        out += "\n===== RUN HOMO1D =====\n"
        out += "{:<35}{}\n".format("Nom du filtre",self.filt_name)
        out += "{:<35}{}\n".format("Méthode de filtrage",self.filt_method)
        out += "{:<35}{} et {} m-1\n".format("Nombre d'ondes coins",self.k1, self.k2)
        out += "{:<35}{}Hz\n".format("Fréquence maximale de la source",self.fmax)    
        out += "{:<35}{}\n".format("Epsilon_0 de l'homogeneisation",self.eps_0)
        
        if self.eps_0 > 1 :
            out += "-> ATTENTION: epsilon_0 élevé, l'homogeneisation peut ne pas être exacte"

        return out
        
    
    def run_homo(self):
        
        self.interp_in2inter()
        self.define_filter()
        self.compute_effective_medium()

    def compute_eps_0(self):
        ''' compute such as eps_0 = lamnda_0/lamnda_min ; lamnda_min = beta_min / f_max'''
        c = np.sqrt(self.mu_in / self.rho_in)
        lambda_min = c.min() / self.fmax
        lambda_0 = 2/(self.k1+self.k2)
        self.eps_0  = lambda_0 / lambda_min  
        
    def heaviside_wavelet(self):
        '''frequency response heavy side with cosine transition'''
        K = self.K
        k1 = self.k1
        k2 = self.k2
        wave_freq = np.array([0 if k > k2 else 1 if k < k1 else 0.5*(1+ np.cos((k-k1)*np.pi/(k2-k1))) for k in K])
        self.wave_freq = wave_freq
        self.wave_freq_all = np.concatenate([wave_freq, wave_freq[::-1]]) # numpy ordering
    
    def gaussian_wavelet(self):
        '''frequency response gaussian'''
        K = self.K
        k0 = (self.k1 + self.k2)/2
        n = 1.5 # paramètre arbitraire
        wave_freq = np.exp(-(K/(k0*n))**2)
        self.wave_freq = wave_freq
        self.wave_freq_all = np.concatenate([wave_freq, wave_freq[::-1]])
        
    def get_spatial_wavelet(self):
        '''Get impulse response from frequency response of the wavelet'''
        wave_fft = np.real(np.fft.ifft(self.wave_freq_all))
        self.wavelet = np.concatenate([wave_fft[self.idx_0:], wave_fft[:self.idx_0]])
        
    def define_filter(self):
        '''define the wavelet filter'''
        if self.filt_name == "heaviside":
            self.heaviside_wavelet()
            
        elif self.filt_name == "gaussian":
            self.gaussian_wavelet()
            
        else: raise Exception(f"unknown filter name : {self.filt_name}")
        
        self.get_spatial_wavelet()
            
    def apply_filt(self, X):
        '''apply the filter'''
        if self.filt_method == "convolution":
            return np.convolve(X, self.wavelet, mode='same')
            
        elif self.filt_method == "FFT":
            X_b = X - X.sum()
            X_fft = np.fft.fft(X_b)
            X_filt = np.real(np.fft.ifft(np.multiply(self.wave_freq_all, X_fft)))
            return X_filt + X.sum()
            
        else: raise Exception("Filter_method has to be 'FFT' or 'convolution'")

    def interp_in2inter(self):
        ''' interpolate the properties on a linear space'''
        mesh_inter      = np.linspace(self.mesh_in[1], self.mesh_in[-1], Homo1D.Nx)
        self.rho_inter  = np.interp(mesh_inter, self.mesh_in, self.rho_in)
        self.mu_inter   = np.interp(mesh_inter, self.mesh_in, self.mu_in)
        self.mesh_inter = mesh_inter
        K_complet   = np.fft.fftfreq(Homo1D.Nx, d=abs(self.mesh_inter[-1]-self.mesh_inter[0])/Homo1D.Nx) * 2*np.pi # espace des nombres d'onde
        self.idx_0  = np.argmin(-K_complet)
        self.K      = K_complet[:self.idx_0+1]
    
    def compute_effective_medium(self):
        '''methode issue de la solution analytique 1D'''
        self.rho_star_inter = self.apply_filt(self.rho_inter)
        self.mu_star_inter  = 1 / self.apply_filt(1/self.mu_inter)
        

    def get_rho_star(self, mesh_out):
        return np.interp(mesh_out, self.mesh_inter, self.rho_star_inter)
    def get_mu_star(self, mesh_out):
        return np.interp(mesh_out, self.mesh_inter, self.mu_star_inter)
    
    def plot_rho(self):
        plt.plot(self.mesh_inter, self.rho_inter,'g',label="density")
        plt.plot(self.mesh_inter, self.rho_star_inter,'b', label="effective density")
        plt.legend()
     
    def plot_mu(self):
        plt.plot(self.mesh_inter, self.mu_inter,'g', label="shear modulus")
        plt.plot(self.mesh_inter, self.mu_star_inter,'b', label="effective shear modulus")
        plt.legend()
        
    def plot_wavelet(self):
        plt.plot(self.mesh_inter, self.wavelet)
        
    def plot_wavelet_freq(self):
        plt.plot(self.K, self.wave_freq)
        axes = plt.gca()
        axes.set_xlim([0,5*self.k2])
        plt.axvline(self.k1,color='g')
        plt.axvline(self.k2,color='g')
        
    def save_properties(self, path):
        '''save the effective properties in the directory path'''
        rho_file = "density-eff.asc"
        mu_file = "shear-modulus-eff.asc"
        np.savetxt(f"{path}/{mu_file}", np.vstack((self.mesh_inter, self.mu_star_inter)).T, delimiter="\t", header="position(m)\teffective shear modulus()\n")
        np.savetxt(f"{path}/{rho_file}", np.vstack((self.mesh_inter, self.rho_star_inter)).T, delimiter="\t", header="position(m)\teffective density (kg/m3)\n")
        print(f"Enregistrement des fichiers {rho_file} et {mu_file} dans le répertoire {path}")
        
    def save_wavelet(self,path):
        '''save the wavelet in the frequencie and spatial spaces'''
        wv_sp_file = "wavelet_space.asc"
        wv_fq_file = "wavelet_freq.asc"
        np.savetxt(f"{path}/{wv_sp_file}", np.vstack((self.mesh_inter, self.wavelet)).T, delimiter="\t", header="position(m)\twavelet\n")
        np.savetxt(f"{path}/{wv_fq_file}", np.vstack((self.K, self.wave_freq)).T, delimiter="\t", header="nombre d'onde (m-1)\twavelet frequencie amplitude \n")
        print(f"Enregistrement des fichiers {wv_sp_file} et {wv_fq_file} dans le répertoire {path}")
        
        
if __name__ == "__main__":
    
    # EXEMPLE D'UTILISATION

    rho0 = 3000
    G0  = 40e9
    fmax = 10
    
    Nx  = 200       # nb d'éléments
    Np  = 4         # degré polynomial par élément
    H   = 10000     
    mesh = np.linspace(0,H,(Np+1)*Nx)
    
    k1,k2 = 0.1,0.5
    
    # Milieu homogène avec pertubation aléatoire

    rho = np.ones((Np+1)*Nx) * rho0 * (np.random.random((Np+1)*Nx)/10 +1)
    G   = np.ones((Np+1)*Nx) * G0 * (np.random.random((Np+1)*Nx)/10 +1)
        
    # homo = Homo1D(mesh, mesh, rho, G, k1, k2, fmax, filter_method='convolution')
    homo = Homo1D(mesh, rho, G, k1, k2, fmax, filter_name="gaussian")
    homo.run_homo()
    
    
    homo.plot_rho()
    
    # homo.plot_mu()
    
    # homo.plot_wavelet()
    
    # homo.plot_wavelet_freq()
    
    
    
    

