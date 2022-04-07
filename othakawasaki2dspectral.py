#!/opt/homebrew/anaconda3/bin/python3
#-*- coding: utf-8 -*-
# **************************************************************************** #
#                                                                              #
#                              ch2dspectral.py                                 #
#                                                                              #
# **************************************************************************** #
#                                                                              #
# Semi-implicit pseudo-spectral scheme for solving the Cahn-Hilliard (CH)      #
# equation in two dimensions using Periodic boundary conditions (PBC).         # 
# The equation is:                                                             #
#                                                                              #
#                   u_t = \nabla^2 (f(u) + K \nabla^2 u)                       #
#                                                                              #
# where f(u) = u^3 - u, and all constants, variables and parameters are dimen- #
# sionless.                                                                    #                                                    
#                                                                              #
# **************************************************************************** #
__version__ = 1.0
__author__ = "Daniel Coelho (daniel.coelho@mail.mcgill.ca)"

import os
import sys
import time
from datetime import datetime
import numpy as np
import scipy as sp
from scipy.fft import fft, ifft
from scipy.fftpack import fft2,ifft2,fftn,ifftn,fftfreq
import matplotlib.pyplot as plt
import customplots


sim = {'DT'       : 0.5,    # Time step [a.u.]
       'STARTTIME': 0,       # Initial time [a.u.]
       'ENDTIME'  : 1000,    # Final time [a.u.]
       }

Nsteps = int((sim['ENDTIME'] - sim['STARTTIME']) / sim['DT'])
sim.update(NSTEPS = Nsteps)   # Update simulation dict

par = {'K'    : 1.0,  # "Surface tension"
       'PHI0' : 0.0,  # Average order parameter \phi_0
       'N'    : 256,  # Number of nodes in x,y
       'GR'   : 8,    # Grid resolution in x,y
       }

# Nw = par['N'] / par['GR']                      # Number of wavelengths in x,y
# q0 = np.sqrt(1/2)                              # Critical wavenumber
# L  = Nw * (2*np.pi / q0)                       # Domain side length [a.u.]
# dx = L / par['N']                              # Grid spacing (dx, dy) [a.u.]
h = 2*np.pi / par['GR']
L  = par['N'] * h
par.update(L = L, DX = h)   # Update parameters dict


def spectral_modes(par):
    Nx = par['N']
    Ny = par['N']
    dx = par['DX']
    dy = par['DX']
    fx = (2*np.pi) * fftfreq(Nx, dx)
    fy = (2*np.pi) * fftfreq(Ny, dy)
    KX = np.outer(fx, np.ones(Nx))
    KY = np.outer(np.ones(Nx), fy)
    return KX, KY

def ch2solver(phi, phiHn, k2, kx2 , k4, dt):
    K = par['K']
    phi0 = par['PHI0']
    return (phiHn - dt * k2 * fft2(phi**3)) / (1 + dt * (K * k4 - k2))

def plot_now(A,t):
    N = par['N'] 
    L = par['L'] 
    dx = par['DX'] 
    plt.clf()
    plt.imshow(A, cmap = plt.get_cmap('binary'),
               interpolation='gaussian', extent=[0,L,0,L])#,
               # vmin=-.1,vmax=.1)
    plt.axhline(y=8.25,xmin=0.815,xmax=0.935,color='white',lw=12.0)
    plt.axhline(y=11,xmin=0.8,xmax=0.95,color='black',lw=2.0)
    plt.text(0.795*L,0.015*L,r'500 nm',color='black',fontsize=10,weight='bold')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('t = '+str(round(t,2)))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.pause(1e-10)
    return

def plot_png(A,t,Dir):
    fig, ax = plt.subplots()
    N = par['N'] 
    L = par['L'] 
    dx = par['DX']
    # Slicing
    # L1 = int(N/4)  H1 = int(N/4)
    # NC = int(N/4)
    # L2 = L1 + NC  H2 = H1 + NC
    # A = A[L1:L2,H1:H2]
    plt.clf()
    plt.imshow(A, cmap = plt.get_cmap('binary'),
               interpolation='gaussian', extent=[0,L,0,L])#,
               # vmin=-1,vmax=1)
    plt.title('t = %2.1f'%t)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.xticks([])
    plt.yticks([])
    plt.axis('on')
    fig.savefig(Dir + '/png/' + 'phi_t_%2.1f.pdf'%t,
                format='pdf', dpi=300)
    return

def plot_pdf(A,t,Dir):
    fig, ax = plt.subplots()
    N = par['N']
    L = par['L']
    dx = par['DX']
    plt.clf()
    plt.imshow(A, cmap = plt.get_cmap('binary'),
               interpolation='gaussian', extent=[0,L,0,L])#,
               # vmin=-1,vmax=1)
    plt.title('t = %2.1f'%t)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    clb = plt.colorbar()
    # plt.clim(-1,1)
    clb.ax.set_title(r'$\phi$', y = 1.01, rotation=0)
    fig.savefig(Dir + '/pdf/' + 'phi_t_%2.1f.pdf'%t)
    return

def create_folder(saveDir):
    if not os.path.isdir(saveDir):
        print("%s directory does not exist, creating..."%saveDir)
        os.mkdir(saveDir)
        print("...Done.")
    return

def main():

    SimStartsAt=datetime.now()
    start = time.perf_counter()

    # Parameters
    K = par['K']
    phi0 = par['PHI0']
    N = par['N']

    t = sim['STARTTIME']
    tf = sim['ENDTIME']
    dt = sim['DT']
    dt2 = 0.5 * dt
    Nsteps = sim['NSTEPS']

    # Wavenumbers
    kx, ky = spectral_modes(par)
    kx2 = kx**2
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Create folders
    create_folder("./simulations/")
    create_folder("./simulations/phi_%s/"%phi0)
    saveDir = "./simulations/phi_%s/"%phi0+"K_%s/"%K
    create_folder(saveDir)
    create_folder(saveDir + 'csv/')
    create_folder(saveDir + 'png/')
    create_folder(saveDir + 'pdf/')

    # From preexisting IC
    # saveDir = saveDir + "PRE/"
    # create_folder(saveDir)
    # create_folder(saveDir + 'png/')
    # create_folder(saveDir + 'pdf/')

    # Initial condition and matrices

    if t > 0:
        phin = np.loadtxt(saveDir + 'csv/phi_t_%2.1f.csv'%t, delimiter=',')
    else:
        # uncomment to start from preexisting initial condition:
        # phin = np.loadtxt(saveDir + '/csv/phi_t_%2.1f.csv'%t, delimiter=',')

        # uncomment to start from random initial condition:
        # np.random.seed(0)
        phin = phi0 + \
        sp.random.uniform(-1e0,1e0,(N, N))          
        np.savetxt('./phi_IC.csv',phin,delimiter=',')
    phiOld  = phin.copy()                              # (n-1) in real space
    phiHn = fft2(phin)                                 # (n) in Fourier space
    phiHold  = phiHn.copy()                            # (n-1) in Fourier space
    phiHnew  = phiHn.copy()                            # (n+1) in Fourier space
    phiNew  = phin.copy()                              # (n+1) in real space

    # if t == 0:
    #     np.savetxt(saveDir + '/csv/phi_t_%2.1f.csv'%t,phiNew,delimiter=',')
    #     plot_pdf(phin.T,t,saveDir)
    #     plot_png(phin.T,t,saveDir)

    # Time-stepping loop
    print("Now running code from t = %s"%t+" to t = %s"%tf+"...")

    # Main loop:
    for o in range(Nsteps):
        # Update phi in Fourier space
        phiHnew = ch2solver(phin, phiHn, k2, kx2, k4, dt)

        # Inverse transform new value (n+1)
        phiNew  = ifft2(phiHnew)

        # Save (n) and (n-1) for the next time step
        phin = phiNew.copy()
        phiHn = phiHnew.copy()

        # Re(\varphi)
        phiNew = phiNew.real

        # Plotting
        t += dt
        plot_now(phiNew.T,t)

    # Fixing axis (.T)
    np.savetxt(saveDir + 'csv/phi_t_%2.1f.csv'%t,phiNew,delimiter=',')
    # plot_png(phiNew.T,t,saveDir)
    plot_pdf(phiNew.T,t,saveDir)

    finish = time.perf_counter()
    print("...Done! Finished in "+str(datetime.now()-SimStartsAt))
        

if __name__ == "__main__":
    main()