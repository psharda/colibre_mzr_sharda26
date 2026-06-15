This repository stores the code and data needed to reproduce the figures shown in the paper on the gas-phase mass-metallicity relation (MZR) of galaxies in the COLIBRE cosmological simulations.

The observational compilation is available in the directory `observed_data`. The functions `plot_obsv_data` and `plot_obsv_data_z` in `my_funcs.py` can be used to retrieve and plot the compiled observations on the mass-metallicity plane and the metallicity-redshift plane, respectively.

The compilation of simulated data used from COLIBRE, as well as from EAGLE, TNG and SIMBA is available in the directory `simulated_data`. The SIMBA data was kindly provided by Alex Garcia.

To produce the figures from the paper, simply clone this repository and run the associated jupyter notebook `plotting_scripts`, where each cell produces a given figure in the paper.

For any queries about this repository or the associated paper, please contact Piyush Sharda, Leiden University (sharda@strw.leidenuniv.nl).

For queries about the [COLIBRE Project](https://colibre.strw.leidenuniv.nl/), please contact Joop Schaye, Leiden University (schaye@strw.leidenuniv.nl)

```Dependencies

This repository needs the following python packages:
numpy
scipy
astropy
cmasher
h5py
matplotlib


```
