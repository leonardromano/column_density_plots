#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Analysis code by Kazunori Okamoto, translated into python by Leonard Romano
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from time import time
#################################
#global variables
#proton mass
mp = 1.672621637*10**(-24)
#################################


#This function can be replaced with your SPH Kernel function of choice
def W(r, h):
    "cubic B4-spline"
    a = r/h
    if 0. <=a<=0.5:
        return (1.-6.*a**2+6.*a**3)/(np.pi*h**3)
    elif 0.5 <= a <= 1.:
        return (2.*(1.-a)**3)* 8/(np.pi*h**3)
    elif 1.<a:
        return 0.

def readable(label):
    return r"$\mathrm{\mathbf{" + label + "}}$"

def smoothColumnDensityPlot(positions, smoothingLengths, densities, masses, \
                            Nbins = 100, boxlength = 15., \
                            cmap = "inferno", output_dir = "./"):
    "Makes a Column Density plot including smoothing length"
    #first compute column densities
    density_xz, density_xy = densityDistribution(positions, smoothingLengths, \
                                                 densities, masses, \
                                                 Nbins, boxlength)
    #create grid
    XX = np.linspace(-boxlength, boxlength, Nbins)
    YY = np.linspace(-boxlength, boxlength, Nbins)
    X,Y = np.meshgrid(XX,YY)
    #finally make the plots
    t0 = time()
    makePlot(X,Y, density_xy, cmap, "face on", output_dir)
    makePlot(X,Y, density_xz, cmap, "edge on", output_dir)
    t1 = time()
    print("finished making column density plots! Took %g seconds." %(t1-t0))

def densityDistribution(pos, hsml, density, mass, binNumber = 100, boxsize = 15.):
    """calculates the column density including the smoothing length
    for gas particles"""
    #initialize arrays for data storage
    rho_3d = np.zeros((binNumber,binNumber,binNumber))
    w_3d = np.zeros((binNumber,binNumber,binNumber))
    rho_xz = np.zeros((binNumber,binNumber))
    rho_xy = np.zeros((binNumber,binNumber))
    
    t0 = time()
    print("calculating spatial density...\n")
    
    dr = 2.0*boxsize/binNumber
    #first loop over all particles to compute densities and weights
    l = 0
    for pt in pos:
        #initialize bounds of particle position in grid
        boundsi = [int((boxsize + pt[0] - hsml[l])/dr -0.5 - 2.), \
                   int((boxsize + pt[0] + hsml[l])/dr -0.5 + 2.)]
        boundsj = [int((boxsize + pt[1] - hsml[l])/dr -0.5 - 2.), \
                   int((boxsize + pt[1] + hsml[l])/dr -0.5 + 2.)]
        boundsk = [int((boxsize + pt[2] - hsml[l])/dr -0.5 - 2.), \
                   int((boxsize + pt[2] + hsml[l])/dr -0.5 + 2.)]
        #if particle reaches out of box: cut off
        if boundsi[0] < 0:
            boundsi[0] = 0
        if boundsj[0] < 0:
            boundsj[0] = 0
        if boundsk[0] < 0:
            boundsk[0] = 0
        #if lower bound out of box: discard particle
        if boundsi[0] >= binNumber:
            l+=1
            continue
        if boundsj[0] >= binNumber:
            l+=1
            continue
        if boundsk[0] >= binNumber:
            l+=1
            continue
        #if particle reaches out of box: cut off
        if boundsi[1] > binNumber:
            boundsi[1] = binNumber
        if boundsj[1] > binNumber:
            boundsj[1] = binNumber
        if boundsk[1] > binNumber:
            boundsk[1] = binNumber
        #if upper bound out of box: discard particle
        if boundsi[1] <= 0:
            l+=1
            continue
        if boundsj[1] <= 0:
            l+=1
            continue
        if boundsk[1] <= 0:
            l+=1
            continue
        #loop over all bins the particle is occupying
        for i in range(boundsi[0], boundsi[1]):
            for j in range(boundsj[0], boundsj[1]):
                for k in range(boundsk[0], boundsk[1]):
                    ds = np.linalg.norm(np.array([-boxsize + dr*(i+0.5) - pt[0], \
                                                  -boxsize + dr*(j+0.5) - pt[1], \
                                                  -boxsize + dr*(k+0.5) - pt[2]]))
                    #add particles contribution to SPH density estimate 
                    #in cell (i,j,k) in units of mp/hsml³
                    rho_3d[i,j,k] += mass[l]*W(ds, hsml[l])/mp
                    #add particles contribution to volume element weight in cell
                    w_3d[i,j,k] += (mass[l]/density[l])*W(ds, hsml[l])
        l+=1
    #now loop over all cells weighting densities
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(binNumber):
                if w_3d[i,j,k] != 0:
                    rho_3d[i,j,k] /= w_3d[i,j,k]
    #computing LOS integrals
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(binNumber):
                rho_xz[k,i] += rho_3d[i,j,k]
                rho_xy[j,i] += rho_3d[i,j,k]
    rho_xz *= dr
    rho_xy *= dr
    
    t1 = time()
    print("finished calculating the spatial density!\nTook %g seconds" %(t1-t0))
    return rho_xz, rho_xy

def makePlot(X, Y, data, cmap, mode, output_dir):
    "Makes a colormesh plot and optionally saves it"
    plt.rcParams["font.size"] = 14
    plt.pcolormesh(X, Y, data, cmap = cmap, \
                   norm = LogNorm(), shading = 'gouraud')
    plt.colorbar(label='column gas mass density')
    plt.title("Density distribution of gas (" +mode+")")
    plt.xlabel(readable("X [kpc]"))
    if mode == "edge on":
        plt.ylabel(readable("Z [kpc]"))
    else:
        plt.ylabel(readable("Y [kpc]"))
    plt.savefig(output_dir + "column_density_"+ mode +".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    plt.rcParams["font.size"] = 10


    
