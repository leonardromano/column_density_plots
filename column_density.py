#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:20:35 2021

@author: leonard
"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from numpy import array, zeros, sqrt, pi, linspace, meshgrid
from  os import environ
from psutil import cpu_count
import ray
from time import time


#################################
#global variables
#proton mass
mp = 1.672621637*10**(-24)
#depending on the used scheduling system we have a
#different environment variable for the number of available cores
Scheduler = "SGE"
if Scheduler == "SGE":
    NCPU = int(environ['NSLOTS'])
elif Scheduler == "SLURM":
    NCPU = int(environ['SLURM_CPUS_PER_TASK'])
elif Scheduler == "PBS":
    NCPU = int(environ['PBS_NP'])
else:
    NCPU = cpu_count(logical=False)
TINY_NUMBER = 1e-20
#################################

#call this function to make a column density plot
def smoothColumnDensityPlot(positions, smoothingLengths, densities, masses, \
                            Nbins = 100, boxlength = 15, \
                            cmap = "inferno", output_dir = "./"):
    "Makes a Column Density plot including smoothing length"
    #first compute column densities
    ray.shutdown()
    ray.init(num_cpus = NCPU)
    print("Making column density plot using %d processor cores."%NCPU)
    particles = array([Particle(positions[i], smoothingLengths[i], \
                          densities[i], masses[i]) \
                 for i in range(positions.shape[0])])
    density_xz, density_xy = densityDistribution(particles, Nbins, boxlength)
    ray.shutdown()
    #create grid
    XX = linspace(-boxlength, boxlength, Nbins)
    YY = linspace(-boxlength, boxlength, Nbins)
    X,Y = meshgrid(XX,YY)
    #finally make the plots
    t0 = time()
    makePlot(X,Y, density_xy, cmap, "face on", output_dir)
    makePlot(X,Y, density_xz, cmap, "edge on", output_dir)
    t1 = time()
    print("finished making column density plots! Took %g seconds." %(t1-t0))

#a class for handling the parallel part of the density computation
@ray.remote
class Density_data():
    def __init__(self, binNumber, boxsize):
        self.binNumber = binNumber
        self.boxsize = boxsize
        self.dr = 2 * boxsize / binNumber
        self.rho_3d = zeros((binNumber,binNumber, binNumber))
        self.w_3d   = zeros((binNumber,binNumber, binNumber))
        
    def update(self, p):
        #initialize the bounds of the region occupied by the particle
        #now compute the bounds of our computation
        h  = p.Hsml
        pt = p.Position
        
        boundsi = [int((self.boxsize + pt[0] - h)/self.dr - 0.5 - 2), \
                   int((self.boxsize + pt[0] + h)/self.dr - 0.5 + 2)]
        boundsj = [int((self.boxsize + pt[1] - h)/self.dr - 0.5 - 2), \
                   int((self.boxsize + pt[1] + h)/self.dr - 0.5 + 2)]
        boundsk = [int((self.boxsize + pt[2] - h)/self.dr - 0.5 - 2), \
                   int((self.boxsize + pt[2] + h)/self.dr - 0.5 + 2)]
        #if particle reaches out of box: cut off
        boundsi[0] = max(0, boundsi[0])
        boundsj[0] = max(0, boundsj[0])
        boundsk[0] = max(0, boundsk[0])
        #if lower bound out of box: return empty
        if boundsi[0] >= self.binNumber:
            return
        if boundsj[0] >= self.binNumber:
            return
        if boundsk[0] >= self.binNumber:
            return
        #if particle reaches out of box: cut off
        boundsi[1] = min(self.binNumber, boundsi[1])
        boundsj[1] = min(self.binNumber, boundsj[1])
        boundsk[1] = min(self.binNumber, boundsk[1])
        #if upper bound out of box: return empty
        if boundsi[1] <= 0:
            return
        if boundsj[1] <= 0:
            return
        if boundsk[1] <= 0:
            return
        #loop over all bins the particle is occupying
        for i in range(boundsi[0], boundsi[1]):
            for j in range(boundsj[0], boundsj[1]):
                for k in range(boundsk[0], boundsk[1]):
                    dx = self.dr * (i + 0.5) - pt[0] - self.boxsize
                    dy = self.dr * (j + 0.5) - pt[1] - self.boxsize
                    dz = self.dr * (k + 0.5) - pt[2] - self.boxsize
                    q = sqrt(dx**2 + dy**2 + dz**2)/h
                    if q > 1:
                        continue
                    #add particles contribution to SPH density estimate 
                    #in cell (i,j,k) in units of mp/hsml³
                    mlwk = p.Mass * W(q, h)
                    self.rho_3d[i,j,k] += mlwk/mp
                    #add particles contribution to volume element weight in cell
                    self.w_3d[i,j,k] += mlwk/p.Density
                    
    def process_particles(self, particles):
        for particle in particles:
            self.update(particle)
        return self.rho_3d, self.w_3d
        

#this class stores the particle data
class Particle():
    def __init__(self, position, hsml, density, mass):
        self.Position = position
        self.Hsml = hsml
        self.Density = density
        self.Mass = mass

#This function can be replaced with your SPH Kernel function of choice
def W(q, h):
    "cubic B4-spline"
    if 0 <= q <=0.5:
        return (1 - 6 * q**2 + 6 * q**3) / (pi * h**3)
    elif 0.5 <= q <= 1:
        return (2 * (1 - q)**3) * 8 / (pi * h**3)
    elif 1 < q:
        return 0

def readable(label):
    return r"$\mathrm{\mathbf{" + label + "}}$"

def densityDistribution(particles, binNumber = 100, boxsize = 15):
    """calculates the column density including the smoothing length
    for gas particles"""
    #initialize arrays for data storage
    actors = [Density_data.remote(binNumber, boxsize) for _ in range(NCPU)]
    rho_3d = zeros((binNumber,binNumber,binNumber))
    w_3d   = zeros((binNumber,binNumber,binNumber))
    rho_xz = zeros((binNumber,binNumber))
    rho_xy = zeros((binNumber,binNumber))
    
    t0 = time()
    print("calculating spatial density...\n")
    #first loop over all particles to compute densities and weights
    #spread the work evenly among all processors
    n = len(particles)//NCPU
    result_ids = [actors[i].process_particles.remote(particles[i * n : (i + 1) * n]) \
                  for i in range(NCPU-1)]
        
    result_ids.append(actors[NCPU-1].process_particles.\
                      remote(particles[(NCPU-1) * n :]))
    #now combine the contributions from all processors
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        drho_3d, dw_3d = ray.get(done_id[0])
        rho_3d += drho_3d
        w_3d   += dw_3d
        
    t1 = time()
    print("Particle loop took %g seconds"%(t1 - t0))
    
    #weight the densities with the SPH-Volume
    rho_3d /= (w_3d + TINY_NUMBER)
        
    #now integrate over the line of sights
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(binNumber):
                rho_xz[k,i] += rho_3d[i,j,k]
                rho_xy[j,i] += rho_3d[i,j,k]
    dr = 2 * boxsize / binNumber
    rho_xz *= dr
    rho_xy *= dr
    t3 = time()
    print("LoS integral took %g seconds"%(t3 - t1))
    t4 = time()
    print("finished calculating the spatial density! Took %g seconds" %(t4-t0))
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
