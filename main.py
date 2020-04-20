#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma # ma = masked arrays
#import bindfit as bf
import bindfit as bf

if __name__ == '__main__':
    
    # define supramolecular systems
    # (all )
    ssHHG = bf.SupraSystem('HHG')
    ssHHG.add_equilibrium(['H', 'G'], ['HG'], 2.0)
    ssHHG.add_equilibrium(['H', 'HG'], ['HHG'], 0.5)
    ssHHG.finalize_system([[1, 1],  # [[HG contains 1 G, HHG contains 1 G],
                           [1, 2]]) #  [HG contains 1 H, HHG contains 2 H]]
    
    ssHGG = bf.SupraSystem('HGG')
    ssHGG.add_equilibrium(['H', 'G'], ['HG'], 2.0)
    ssHGG.add_equilibrium(['G', 'HG'], ['HGG'], 0.5)
    ssHGG.finalize_system([[1, 2],  # [[HG contains 1 G, HGG contains 2 G],
                           [1, 1]]) #  [HG contains 1 H, HGG contains 1 H]]
    
    ssHG = bf.SupraSystem('HG')
    ssHG.add_equilibrium(['H', 'G'], ['HG'], 1.0)
    ssHG.finalize_system([[1],  # [[HG contains 1 G],
                          [1]]) #  [HG contains 1 H]]
    
    # species are ordered alphabetically, e.g. G, H, HG, HHG
    # observables are ordered alphabetically, e.g. 550 nm, 562 nm, 601 nm
    epsilons = ma.array([
                [10.0, 23364., 1.0e4],#, 1.0e4], # ig @ obs[0]
                [10.0, 9153.8, 1.0e4],#, 1.0e4], # ig @ obs[1]
                [10.0, 1286.1, 1.0e4],#, 1.0e4], # ig @ obs[2]
            ], mask=[
                [False, True, False],#, False], # ig @ obs[0]
                [False, True, False],#, False], # ig @ obs[1]
                [False, True, False],#, False]  # ig @ obs[2]
            ])
    # equilibria are ordered as they are added to the supramolecular system
    #assconst = ma.array([7.5e3, 1.0], mask=[False, False])
    assconst = ma.array([7.5e3], mask=[False])
    
    # run the fitter:
    bcf = bf.BindingCurveFitter(ssHG)
    bcf.set_data_files('initial-concentrations.csv', 'binding-curves.csv')
    bcf.set_initial_guess(epsilons, assconst) # masked arrays
    bcf.optimize()
    #bcf.corr_resid_specconc()
    
    # print optimization results:
    print(bcf.epsilons)
    print(bcf.assconst)
    
    """ testing if the equilibrium concentrations are calculated correctly
    # for each point on the titration curve, check consistency:
    for p in range(bcf.num_points):
        conc_eq = np.power(bcf._conc_equilibrated[:,p], -bcf._equilibria)
        conc_eq[conc_eq==np.Infinity] = 1. # to correct for 0.^-1. (dirty, but fine here)
        K_calc = np.prod( conc_eq, axis=1 ) / bcf._scopicity
        K_err = (K_calc - bcf.assconst) / bcf.assconst
        print( f'Error in calc_equil_conc @ point {p:2}: {K_err[0]:5.4} {K_err[1]:5.4}')
    """
    
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
        ax[0][0].semilogx(x_values, bcf.bindcurv_calc(obs), color=cmap_obs(cval), linestyle='dashed')
        ax[0][0].semilogx(x_values, bcf.bindcurv_exp(obs), color=cmap_obs(cval), linestyle='none',
                    marker='o', markerfacecolor=cmap_obs(cval), markersize=5, label=obs)
    ax[0][0].set_title('Observed vs. calculated binding curves')
    ax[0][0].set_xlabel(r'$[G]_{0}/[H]_{0}$')
    ax[0][0].set_ylabel('Observable')
    ax[0][0].legend()
    
    # generate concentrations plot:
    for index, spec in enumerate(bcf.species_names):
        cval = float(index / bcf.num_species)
        #ax[0][1].semilogx(x_values, bcf.equilibrium_conc[index,:] / bcf.equilibrium_conc[index,:].max(),
        #                  color=cmap_conc(cval), linestyle='solid', label=f'normalized [{spec}]')
        ax[0][1].semilogx(x_values, bcf.equilibrium_conc(spec) / bcf.equilibrium_conc(spec).max(),
                          color=cmap_conc(cval), linestyle='solid', label=f'normalized [{spec}]')
    ax[0][1].set_title('Calculated species concentrations')
    ax[0][1].set_xlabel(r'$[G]_{0}/[H]_{0}$')
    ax[0][1].set_ylabel('Concentration')
    ax[0][1].legend()
    
    # generate error plot:
    for index, obs in enumerate(bcf.observed_at):
        cval = float(index / bcf.num_observables)
        ax[1][0].semilogx(x_values, bcf.bindcurv_residuals(obs), color=cmap_obs(cval), linestyle='none',
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
    