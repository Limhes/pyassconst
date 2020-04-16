#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares



class SupraSystem:
    
    def __init__(self, name:str):
        self._name = name
        self.equilibria = pd.DataFrame()
        
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
        eq = {species: [float(LHS_summed.get(species, 0) - RHS_summed.get(species, 0))]
            for species in set(LHS_summed) | set(RHS_summed)}
        eq['scopicity'] = [float(_scopicity)]
        
        # add eq to self.equilibria (outer join) and replace NaN with 0.0:
        self.equilibria = pd.concat([self.equilibria, pd.DataFrame(eq)],
                                    axis=0, join='outer', ignore_index=True).fillna(0.0)
        # sort columns alphabetically on species name:
        self.equilibria = self.equilibria.sort_index(axis=1)
        
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



class ConcentrationProfileSimulator:
    
    def __init__(self, ss:SupraSystem):
        # highest level data:
        self._ss = ss
        self._df_conc_initial = pd.DataFrame()
        self._conc_initial = np.array([])
        self._conc_equilibrated = np.array([])
    
        # concentration matrices:
        self._scopicity = np.array(self._ss.equilibria['scopicity'])
        self._equilibria = np.array(self._ss.equilibria[self._ss.species_names])
        self._Del = np.array([])
        self._Del0 = np.array([])
        self._conc_min = np.array([])
        self._conc = np.array([])

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
        return self._df_conc_initial[species_name].to_numpy()

    def set_initial_concentrations(self, conc_initial:pd.DataFrame):
        self._df_conc_initial = pd.DataFrame( {key: [] for key in self._ss.species_names} )
        self._df_conc_initial = pd.concat([self._df_conc_initial, conc_initial],
                                      axis=0, join='outer', ignore_index=True).fillna(0.0)
        
        self._conc_min = np.amin( np.array(conc_initial), axis=1 )
        self._conc_initial = np.array(self._df_conc_initial).transpose()
        self._Del0 = np.zeros((self.num_species, self.num_points))

    def calc_equil_conc(self, ass_const:np.ndarray):
        # create empty matrices to store concentrations:
        self._Del = self._Del0
        self._conc = self._conc_initial
        
		# set equilibrium constants for each equilibrium:
        ass_const = ass_const * self._scopicity
        
        # set simulation variables:
        MAX_ITER = 1e6
        V_STABLE = 1e-8
        V_SHRINK = 0.3
        V_GROW = 1.1
        
        # for each titration point, calculate concentrations:
        for p in range(self.num_points):
            # set initial time increment:
            #dt = np.min(self.conc0[p]) / np.max(ass_const)
            dt = self._conc_min[p] / np.max(ass_const)
            
            current_iteration = 0
            sim_has_converged = False
            while not sim_has_converged:
                
                # calculate next concentration:
                #self._Del[p,:] = 0.0
                self._Del[:,p] = 0.0
                for e in range(self.num_equil):
                    k_f = 1.0e6
                    k_b = k_f / ass_const[e]
                    f = np.prod(( np.power(self._conc[:,p], np.abs(self._equilibria[e])) )[ self._equilibria[e] > 0 ])
                    b = np.prod(( np.power(self._conc[:,p], np.abs(self._equilibria[e])) )[ self._equilibria[e] < 0 ])
                    self._Del[:,p] += -self._equilibria[e] * (f*k_f - b*k_b) * dt
                self._conc[:,p] += self._Del[:,p]
                
                # has concentration gone negative?
                if np.count_nonzero( self._conc[:,p] < 0.0 ) > 0:
                    index_of_minimum = np.argmin(self._conc[:,p])
                    minConc2 = self._conc[:,p][index_of_minimum]
                    self._conc[:,p] -= self._Del[:,p] # restore original concentrations
                    minConc1 = self._conc[:,p][index_of_minimum]
                    dt *= V_SHRINK * minConc1/(minConc1-minConc2)
                else:
                    dt *= V_GROW
                
                # has concentration converged?
                sim_has_converged = not (np.count_nonzero( self._Del[:,p] > V_STABLE * dt ) > 0)
                
                # if too many iterations, take corrective action
                if current_iteration > MAX_ITER:
                    raise Exception('Simulation has reached maximum number of iterations.')
                else:
                    current_iteration += 1
        
        self._conc_equilibrated = self._conc



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


