#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma # ma = masked arrays
import bindfit as bf

if __name__ == '__main__':
    
    # define supramolecular systems:
    ssHHG = bf.SupraSystem('HHG')
    ssHHG.add_equilibrium(['H', 'G'], ['HG'], 2.0)
    ssHHG.add_equilibrium(['H', 'HG'], ['HHG'], 0.5)
    
    ssHGG = bf.SupraSystem('HGG')
    ssHGG.add_equilibrium(['H', 'G'], ['HG'], 2.0)
    ssHGG.add_equilibrium(['G', 'HG'], ['HGG'], 0.5)
    
    # species are ordered alphabetically, e.g. G, H, HG, HHG
    # observables are ordered alphabetically, e.g. 550 nm, 562 nm, 601 nm
    epsilons = ma.array([
                [0.0, 23364., 1.5e4, 1.1e4], # ig @ obs[0]
                [0.0, 9153.8, 1.7e4, 2.2e4], # ig @ obs[1]
                [0.0, 1286.1, 0.7e4, 1.0e4]  # ig @ obs[2]
            ], mask=[
                [True, True, False, False], # ig @ obs[0]
                [True, True, False, False], # ig @ obs[1]
                [True, True, False, False]  # ig @ obs[2]
            ])
    # equilibria are ordered as they are added to the supramolecular system
    assconst = ma.array([7.5e3, 7.5e3], mask=False)
    
    # run the fitter:
    bcf = bf.BindingCurveFitter(ssHGG)
    bcf.set_data_files('initial-concentrations.csv', 'binding-curves.csv')
    bcf.set_initial_guess(epsilons, assconst) # masked arrays
    bcf.optimize()
    bcf.corr_resid_specconc()
    
    # print optimization results:
    print(bcf.epsilons)
    print(bcf.assconst)
    
    # variables for plots:
    x_values = bcf.initial_conc('G')/bcf.initial_conc('H')
    
    # create figure and color maps:
    fig, ax = plt.subplots(2, 2, figsize=(10,8))
    cmap_obs = mpl.cm.get_cmap('gnuplot')
    cmap_conc = mpl.cm.get_cmap('nipy_spectral')
    cmap_corr = mpl.cm.get_cmap('bwr')
    
    # generate results plot:
    for index, obs in enumerate(bcf.observed_at):
        cval = float(index / bcf.num_observables)
        ax[0][0].semilogx(x_values, bcf.bindcurv_calc[index,:], color=cmap_obs(cval), linestyle='dashed')
        ax[0][0].semilogx(x_values, bcf.bindcurv_exp[index,:], color=cmap_obs(cval), linestyle='none',
                    marker='o', markerfacecolor=cmap_obs(cval), markersize=5, label=obs)
    ax[0][0].set_title('Observed vs. calculated binding curves')
    ax[0][0].set_xlabel(r'$[G]_{0}/[H]_{0}$')
    ax[0][0].set_ylabel('Observable')
    ax[0][0].legend()
    
    # generate concentrations plot:
    for index, spec in enumerate(bcf.species_names):
        cval = float(index / bcf.num_species)
        ax[0][1].semilogx(x_values, bcf.equilibrium_conc[index,:] / bcf.equilibrium_conc[index,:].max(),
                          color=cmap_conc(cval), linestyle='solid', label=f'normalized [{spec}]')
    ax[0][1].set_title('Calculated species concentrations')
    ax[0][1].set_xlabel(r'$[G]_{0}/[H]_{0}$')
    ax[0][1].set_ylabel('Concentration')
    ax[0][1].legend()
    
    # generate error plot:
    for index, obs in enumerate(bcf.observed_at):
        cval = float(index / bcf.num_observables)
        ax[1][0].semilogx(x_values, bcf.residuals[index,:], color=cmap_obs(cval), linestyle='none',
                    marker='o', markerfacecolor=cmap_obs(cval), markersize=5, label=obs)
    ax[1][0].set_title('Residuals plot')
    ax[1][0].set_xlabel(r'$[G]_{0}/[H]_{0}$')
    ax[1][0].set_ylabel('Observable')
    ax[1][0].legend()
    
    # plot correlation heat map:
    heatmap = ax[1][1].imshow( bcf.correlation_coeff('float'), cmap=cmap_corr, vmin=-1., vmax=1. )
    fig.colorbar(heatmap)
    ax[1][1].set_xticks(np.arange(bcf.num_species))
    ax[1][1].set_yticks(np.arange(bcf.num_observables))
    ax[1][1].set_xticklabels(bcf.species_names)
    ax[1][1].set_yticklabels(bcf.observed_at)
    ax[1][1].set_title('Correlation of residuals\nwith species concentrations')
    
    # remove overlaps between title and labels:
    plt.tight_layout()
    