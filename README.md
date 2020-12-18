# column_density_plots
A script for making column density plots

# usage
The script is self contained and meant to be imported in your analysis code. 
The main routine for making column density plots is smoothColumnDensityPlot so probably an import like

"from column_density_plots import smoothColumnDensityPlot"

in the right spot will be sufficient.

smoothColumnDensityPlot takes as arguments: 
(numpy_array (N,3) positions, numpy_array (N) smoothingLengths, numpy_array (N) densities, numpy_array (N) masses, 
 int NumberOfBins (default=100), float boxlength (default=15.(code units)), string colormap (default="inferno"), string output_directory (default="./")) 

By default the code assumes that you are plotting a disk rotating around the z-axis. You need to align your data in such a way first before calling this code, in order to get a useful result. The code makes use of the "cubic B-spline" for its computation of the SPH densities. If you are using a different Kernel function, feel free to modify W(x,h) to your Kernel function -- the modifications should be straightforward.
If you have any suggestions for extra functionalities feel free to modify the code and contact me, if you think your modifications should be shared with the world!
