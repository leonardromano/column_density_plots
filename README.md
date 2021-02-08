# column_density_plots
A parallel python script for making column density plots

# non-standard dependencies
The code makes use of the following two non-standard libraries:
- psutil
- ray

using pip they can easily be installed:

	pip install psutil ray
 

# usage
The latest version of the code uses ray for parallelization. If this code is embedded in a larger framework it is probably advisable to comment out the lines ray.init() and ray.shutdown() from the code and put them at the beginning and end of your code in order to avoid multiple (cosly) reinitializations of ray or possible even crashes with already existing instances of ray!
Further you need to specify which scheduler you are using if you are working on a cluster, such that ray can assign the correct number of CPUs. If nothing is specified, it will simply take all physical CPUs on the machine.

Other than the above mentioned configuration the script is self contained and meant to be imported in your analysis code. 
The main routine for making column density plots is smoothColumnDensityPlot so probably an import like

"from column_density import smoothColumnDensityPlot"

in the right spot will be sufficient.

smoothColumnDensityPlot takes as arguments: 
(numpy_array (N,3) positions, numpy_array (N) smoothingLengths, numpy_array (N) densities, numpy_array (N) masses, 
 int NumberOfBins (default=100), float boxlength (default=15.(code units)), string colormap (default="inferno"), string output_directory (default="./")) 

By default the code assumes that you are plotting a disk rotating around the z-axis. It then constructs and plots the 2D-density maps for a box enclosing the region (-boxlength, boxlength)^3, so in order to make use of it, it will be necessary to align the coordinates accordingly. The code makes use of the "cubic B-spline" for its computation of the SPH densities. If you are using a different Kernel function, feel free to modify W(x,h) to your Kernel function -- the modifications should be straightforward.
If you have any suggestions for extra functionalities feel free to modify the code and contact me, if you think your modifications should be shared with the world!
