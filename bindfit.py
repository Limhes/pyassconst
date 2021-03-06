#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, fsolve



class SupraSystem:
    
    def __init__(self, name:str):
        self._name = name
        self.equilibria = pd.DataFrame()
        self._boundary_matrix = np.array([])
        
    def add_equilibrium(self, _LHS:list, _RHS:list, _scopicity:float=1.0):
        """
        Adds an equilibrium to the supramolecular system. Example: A + A <-> A2
        
        params:
            LHS:list(str)    list of species in the left-hand side, e.g. ['A', 'A']
            RHS:list(str)    list of species in the left-hand side, e.g. ['A2']
            scopicity:float  ratio between macroscopic and microscopic equilibrium constant
        """
        # first, transform e.g. ['H', 'H', 'G'] to {'H': 2, 'G': 1} for LHS and RHS
        LHS_summed = dict(Counter(_LHS))
        RHS_summed = dict(Counter(_RHS))
        # then take union of LHS and the negative of RHS:
        eq = {species: [int(LHS_summed.get(species, 0) - RHS_summed.get(species, 0))]
            for species in set(LHS_summed) | set(RHS_summed)}
        eq['scopicity'] = [float(_scopicity)]
        
        # add eq to self.equilibria (outer join) and replace NaN with 0.0:
        self.equilibria = pd.concat([self.equilibria, pd.DataFrame(eq)],
                                    axis=0, join='outer', ignore_index=True).fillna(0)
        # sort columns alphabetically on species name:
        self.equilibria = self.equilibria.sort_index(axis=1)
        
    def finalize_system(self, species_content:list):
        self._boundary_matrix = np.hstack([np.identity(len(species_content)), species_content])
    
    @property
    def name(self):
        return self._name
    
    @property
    def species_names(self):
        return [column_name for column_name in self.equilibria.columns if column_name != 'scopicity']
    
    @property
    def num_species(self):
        return self.equilibria.shape[1]-1
    
    @property
    def num_equil(self):
        return self.equilibria.shape[0]
    
    @property
    def boundary_matrix(self):
        return self._boundary_matrix



class ConcentrationProfileSimulator:
    
    def __init__(self, ss:SupraSystem):
        # highest level data:
        self._ss = ss
        self._conc0 = np.array([]) # initial concentrations, only starting species
        self._conc_initial = np.array([]) # initial concentrations, all species
        self._conc_equilibrated = np.array([]) # concentrations at chemical equilibrium
        
        # concentration matrices:
        self._scopicity = np.array(self._ss.equilibria['scopicity'])
        self._equilibria = np.array(self._ss.equilibria[self._ss.species_names])
        self._eq_pos = np.array([])
        self._eq_neg = np.array([])

    @property
    def num_equil(self):
        return self._ss.num_equil
    
    @property
    def num_species(self):
        return self._ss.num_species
    
    @property
    def species_names(self):
        return self._ss.species_names
    
    @property
    def num_points(self):
        return self._conc_initial.shape[1]
    
    def equilibrium_conc(self, species_name:str) -> np.ndarray:
        return self._conc_equilibrated[self.species_names.index(species_name), :]
    
    def initial_conc(self, species_name:str) -> np.ndarray:
        return self._conc_initial[self.species_names.index(species_name), :]

    def set_initial_concentrations(self, conc_initial:pd.DataFrame):
        # set initial concentration arrays from dataframe:
        self._conc_initial = np.array(
                pd.concat( [ pd.DataFrame( {key: [] for key in self._ss.species_names} ), conc_initial],
                            axis=0, join='outer', ignore_index=True ).fillna(0.0).sort_index(axis=1)
            ).transpose()
        self._conc0 = np.array(conc_initial.sort_index(axis=1)).transpose()
        
        self._conc_equilibrated = np.zeros((self.num_species, self.num_points))
        self._eq_pos = np.clip(self._equilibria, 0, np.inf)
        self._eq_neg = np.clip(-self._equilibria, 0, np.inf)
        
    # the next two functions can be rewritten without the for-loop, but I cannot think (yet) in 3D
    def __fsolve_target_fnc(self, conc, conc0, assconst):
        return [
                *(np.prod( np.power(conc, self._eq_pos), axis=1) * assconst * self._scopicity - \
                  np.prod( np.power(conc, self._eq_neg), axis=1)),
                *(np.matmul(self._ss.boundary_matrix, conc) - conc0)
            ]
    
    def calc_equil_conc(self, ass_const:np.ndarray):
        # ass_const is still in the form [K_a, alpha1, alpha2, ...], so transform it:
        ass_const_copy = np.copy(ass_const) # if not copied, original array will be overwritten --> bad
        ass_const_copy[1:] = ass_const_copy[1:]*ass_const_copy[0] # change alpha_N into K_(N+1)
        # for each titration point, calculate concentrations:
        for p in range(self.num_points):
            self._conc_equilibrated[:,p] = fsolve(self.__fsolve_target_fnc,
                       x0=self._conc_initial[:,p],
                       args=(self._conc0[:,p], ass_const_copy))



class BindingCurveSimulator(ConcentrationProfileSimulator):
    
    def __init__(self, ss:SupraSystem):
        ConcentrationProfileSimulator.__init__(self, ss)
        
        self._bindcurv_exp = np.array([])
        self._bindcurv_calc = np.array([])
        self._observed_at = []

    def bindcurv_exp(self, obs_name:str) -> np.ndarray:
        return self._bindcurv_exp[self._observed_at.index(obs_name),:]
    
    def bindcurv_calc(self, obs_name:str) -> np.ndarray:
        return self._bindcurv_calc[self._observed_at.index(obs_name),:]
    
    @property
    def num_observables(self):
        return len(self._observed_at)
    
    @property
    def observed_at(self):
        return self._observed_at

    def set_data_files(self, fn_initconc:str, fn_bindingdata:str=None):
        self.set_initial_concentrations( pd.read_csv(fn_initconc) )
        
        if fn_bindingdata is not None:
            df_bindcurv_exp = pd.read_csv(fn_bindingdata).sort_index(axis=1)
            self._observed_at = df_bindcurv_exp.columns.tolist()
            self._bindcurv_exp = df_bindcurv_exp.to_numpy().transpose()
            print(f'> {self.num_observables} experimental binding curves loaded at {self.observed_at}')
        
    def calc_binding_curves(self, epsilons:np.ndarray, assconst:np.ndarray):
        self.calc_equil_conc(assconst)
        self._bindcurv_calc = np.matmul(epsilons, self._conc_equilibrated)



class BindingCurveFitter(BindingCurveSimulator):
    
    def __init__(self, ss:SupraSystem):
        BindingCurveSimulator.__init__(self, ss)
        
        self._epsilons = np.array([])
        self._epsilons_flat = np.array([])
        self._epsilons_mask_flat = np.array([])
        self._max_epsilons = 0.
        self._num_epsilons_used_in_fit = 0
        
        self._assconst = np.array([])
        self._assconst_flat = np.array([])
        self._assconst_mask_flat = np.array([])
        self._max_assconst = 0.
        self._num_assconst_used_in_fit = 0
        
        self._rel_imp = 1. # relative importance of assconst over epsilon in fit
        self._residuals = np.array([])
        self._df_corr_coef = pd.DataFrame()
        
    @property
    def epsilons(self):
        return self._epsilons
    
    @property
    def assconst(self):
        return self._assconst
    
    def bindcurv_residuals(self, obs_name:str) -> np.ndarray:
        return self._residuals[self._observed_at.index(obs_name),:]

    def correlation_coeff(self, dtype:str='df'):
        if dtype == 'df':
            return self._df_corr_coef
        else:
            return self._df_corr_coef.to_numpy(dtype=dtype)
    
    def set_initial_guess(self, ig_epsilons:np.ndarray, ig_assconst:np.ndarray):
        self._epsilons = ig_epsilons
        self._epsilons_flat = self._epsilons.data.flatten()
        self._max_epsilons = self._epsilons_flat.max()
        self._epsilons_mask_flat = self._epsilons.mask.flatten()
        self._num_epsilons_used_in_fit = self._epsilons.compressed().shape[0]
        
        self._assconst = ig_assconst
        self._assconst_flat = self._assconst.data.flatten()
        self._max_assconst = self._assconst_flat.max()
        self._assconst_mask_flat = self._assconst.mask.flatten()
        self._num_assconst_used_in_fit = self._assconst.compressed().shape[0]
        
    def unpack_params(self, params):
        # unmasking!
        np.place( self._epsilons_flat, ~self._epsilons_mask_flat, params[ :self._num_epsilons_used_in_fit ] * self._max_epsilons )
        np.place( self._assconst_flat, ~self._assconst_mask_flat, params[ -self._num_assconst_used_in_fit: ] * self._max_assconst )
        return self._epsilons_flat.reshape(self.num_observables, self.num_species), self._assconst_flat
        
    def minimization_target(self, params) -> float:
        self.calc_binding_curves( *self.unpack_params(params) )
        return (self._bindcurv_exp - self._bindcurv_calc).flatten()
    
    def optimize(self, verbose=2):
        initial_guess = np.hstack([self._epsilons.compressed() / self._max_epsilons,
                                   self._assconst.compressed() / self._max_assconst])
        # compressed() returns the unmasked part of the masked array
        relative_importance = np.hstack( [np.ones((self._num_epsilons_used_in_fit)),
                                         np.ones((self._num_assconst_used_in_fit)) * self._rel_imp] )
        bounds = np.zeros((2, initial_guess.shape[0]))
        bounds[1,:] = np.inf # upper bound
        res = least_squares(self.minimization_target, x0=initial_guess,
                            bounds=bounds, x_scale=relative_importance,
                            method='trf', gtol=1e-6, verbose=verbose)
        
        self._epsilons, self._assconst = self.unpack_params(res.x)
        self._residuals = self._bindcurv_exp - self._bindcurv_calc
        
        # after optimization, run correlation on residuals with species concentrations
        self.corr_resid_specconc()
        
    def corr_resid_specconc(self):
        
        self._df_corr_coef = pd.DataFrame([], columns=self.species_names, index=self.observed_at)
        
        for obs_idx, obs_name in enumerate(self.observed_at):
            for spec_idx, spec_name in enumerate(self.species_names):
                corr_coef_mtx = np.corrcoef( self._residuals[obs_idx],
                                             self._conc_equilibrated[spec_idx] )
                self._df_corr_coef.loc[obs_name, spec_name] = (corr_coef_mtx[0][1] + corr_coef_mtx[1][0])/2.


