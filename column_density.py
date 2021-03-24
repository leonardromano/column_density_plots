#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:20:35 2021

@author: leonard
"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from numpy import array, zeros, sqrt, pi, linspace, meshgrid, arcsinh
from os import environ
from psutil import cpu_count
import ray
from time import time

###############################################################################
#global variables
#proton mass
mp = 1.672621637*10**(-24)
TINY_NUMBER = 1e-20

#depending on the used scheduling system we have a
#different environment variable for the number of available cores
Scheduler = ""
if Scheduler == "SGE":
    NCPU = int(environ['NSLOTS'])
elif Scheduler == "SLURM":
    NCPU = int(environ['SLURM_CPUS_PER_TASK'])
elif Scheduler == "PBS":
    NCPU = int(environ['PBS_NP'])
else:
    NCPU = cpu_count(logical=False)
    
###############################################################################
#Line convoluted SPH Kernel: Adjust to your liking

NORM_FAC = 7/(4*pi)
def Y(q, h):
    "Line weight kernel for Wendland-C2 Kernel"
    if q == 0:
        return 4 / h**2
    elif q < 1:
        return (sqrt(1 - q**2) * (4 - 28 * q**2 - 81 * q**4) \
                + 15 * q**4 * (6 + q**2) * arcsinh(sqrt(1 - q**2)/q))/h**2
    else:
        return 0

###############################################################################
#worker class

@ray.remote
class worker():
    def __init__(self, binNumber, boxsize):
        self.binNumber = binNumber
        self.boxsize = boxsize
        self.dr = 2 * boxsize / binNumber
        self.rho_2d = zeros((binNumber,binNumber))
        
    def update(self, p):
        #initialize the bounds of the region occupied by the particle
        #now compute the bounds of our computation
        h  = p.Hsml
        pt = p.Position
        
        boundsi = [int((self.boxsize + pt[0] - h)/self.dr - 0.5 - 2), \
                   int((self.boxsize + pt[0] + h)/self.dr - 0.5 + 2)]
        boundsj = [int((self.boxsize + pt[1] - h)/self.dr - 0.5 - 2), \
                   int((self.boxsize + pt[1] + h)/self.dr - 0.5 + 2)]
        #if particle reaches out of box: cut off
        boundsi[0] = max(0, boundsi[0])
        boundsj[0] = max(0, boundsj[0])
        #if lower bound out of box: return empty
        if boundsi[0] >= self.binNumber:
            return
        if boundsj[0] >= self.binNumber:
            return
        #if particle reaches out of box: cut off
        boundsi[1] = min(self.binNumber, boundsi[1])
        boundsj[1] = min(self.binNumber, boundsj[1])
        #if upper bound out of box: return empty
        if boundsi[1] <= 0:
            return
        if boundsj[1] <= 0:
            return
        #loop over all bins the particle is occupying
        for i in range(boundsi[0], boundsi[1]):
            for j in range(boundsj[0], boundsj[1]):
                dx = self.dr * (i + 0.5) - pt[0] - self.boxsize
                dy = self.dr * (j + 0.5) - pt[1] - self.boxsize
                q = sqrt(dx**2 + dy**2)/h
                if q > 1:
                    continue
                #add particles contribution to SPH density estimate 
                #in cell (i,j,k) in units of mp/hsml³
                mlwk = p.Mass * Y(q, h)
                self.rho_2d[i,j] += mlwk/mp
                    
    def process(self, particles):
        for particle in particles:
            self.update(particle)
        return self.rho_2d

#this class stores the particle data
class Particle():
    def __init__(self, position, hsml, mass):
        self.Position = position
        self.Hsml = hsml
        self.Mass = mass

###############################################################################
# Functions for plotting

def readable(label):
    "Emboldens labels"
    return r"$\mathrm{\mathbf{" + label + "}}$"

def makePlot(X, Y, data, cmap, output_dir):
    "Makes a colormesh plot and saves it to output_dir"
    plt.rcParams["font.size"] = 14
    plt.pcolormesh(X, Y, data, cmap = cmap, norm = LogNorm(), shading = 'gouraud')
    plt.colorbar(label='column gas mass density')
    plt.savefig(output_dir + "rendered_plot.pdf", bbox_inches='tight')
    plt.close()
    plt.rcParams["font.size"] = 10
    
###############################################################################

#call this function to make a column density plot
def smoothColumnDensityPlot(x, y, Hsml, mass, Nbins = 100, boxlength = 15, \
                            cmap = "inferno", output_dir = "./"):
    "Makes a Column Density plot including smoothing length"
    #first compute column densities
    ray.init(num_cpus = NCPU)
    print("Making column density plot using %d processor cores."%NCPU)
    Particles = array([Particle(array([x[i], y[i]]), Hsml[i], mass[i]) \
                       for i in range(x.shape[0])])
    density_2d = densityDistribution(Particles, Nbins, boxlength)
    ray.shutdown()
    #create grid
    XX = linspace(-boxlength, boxlength, Nbins)
    YY = linspace(-boxlength, boxlength, Nbins)
    X,Y = meshgrid(XX,YY)
    #finally make the plots
    t0 = time()
    makePlot(X,Y, density_2d, cmap, output_dir)
    t1 = time()
    print("finished making column rendered plots! Took %g seconds." %(t1-t0))
        
###############################################################################
# Function for density computation

def densityDistribution(Particles, binNumber = 100, boxsize = 15):
    """calculates the column density including the smoothing length
    for gas particles"""
    
    t0 = time()
    print("calculating spatial density...\n")
    
    #first loop over all particles computing densities and weights
    #spread the work evenly among all processors
    load = Particles.shape[0]//NCPU
    
    actors = [worker.remote(binNumber, boxsize) for _ in range(NCPU)]
    
    result_ids = [actors[i].process.remote(Particles[i * load:(i+1) * load]) \
                  for i in range(NCPU-1)]
    result_ids.append(actors[NCPU-1].process.remote(Particles[(NCPU-1) * load:]))
    
    #now reduce the individual results
    rho_2d = zeros((binNumber,binNumber))
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        rho_2d += ray.get(done_id[0])
        
    t1 = time()
    print("Particle loop took %g seconds"%(t1 - t0))
    
    #multiply with the normalisation factor
    rho_2d *= NORM_FAC

    return rho_2d