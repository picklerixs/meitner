import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import warnings

from vamas import Vamas
from math import ceil
from scipy.special import wofz
from scipy.integrate import trapezoid
from lmfit import minimize, Parameters
from lmfit.models import LinearModel
from operator import itemgetter
from matplotlib import rc, rcParams
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Fit:
    
    # class variables for plot styling
    colors = [[98/255,146/255,190/255], # mid-blue
            [64/255,105/255,149/255], # dark blue
            [133/255,127/255,188/255], # sour purple
            [66/255,69/255,133/255], # dark purple
            [250/255,172/255,116/255], # light orange
            [194/255,104/255,47/255], # burnt orange
            [247/255,168/255,170/255], # strawberry pink
            [240/255,79/255,82/255]] # red
        
    data_color = 'black'
    background_color = 'gray'
    envelope_color = 'black'
    
    font_family = 'Arial'
    label_font_size = 12
    tick_font_size = 12
    usetex = False
    
    envelope_linewidth = 1.7
    component_linewidth = 1.6
    background_linewidth = 1.6
    axes_linewidth = 1.5
    
    marker = '+'
    residual_marker = marker
    marker_size = 6
    marker_alpha = 2/3
    marker_edge_width = 0.75
    
    data_zorder = 999
    background_zorder = 100
    envelope_zorder = background_zorder+1
    
    rc('font',**{'family':'sans-serif','sans-serif':[font_family]})
    rc('text', usetex=usetex)
    rcParams['axes.linewidth'] = axes_linewidth
    
    def __init__(self, 
                 xps,
                 n_peaks=1,
                 lineshape='voigt',
                 first_peak_index=0,
                 dict_keys=None,
                 be_guess=None,
                 fit=True,
                 expr_constraints=None,
                 params=None,
                 method='leastsq',
                 peak_ids=None,
                 shift=None):
        '''
        Args:
            xps: Xps instance or list or dict of Xps instances
            n_peaks: single specification for number of peaks or list of len(xps) with entries
                corresponding to each Xps instance in xps.
            first_peak_index: starting index for peak IDs, either single specification
                or list of len(xps) with entries corresponding to each Xps instance in xps.
            dict_keys: list of keys corresponding to elements in xps_list
            expr_constraints: expression constraints passed to Parameters instance.
                Overwrites existing parameter specifications in params.
        '''
        self.shift = shift
        self.expr_constraints = expr_constraints
        # ensure xps is iterable
        if (not Aux.is_list_or_tuple(xps)) and (not isinstance(xps,dict)):
            xps = [xps]
        self.xps = xps
        len_xps = len(self.xps)
        # ensure n_peaks is iterable
        if Aux.is_float_or_int(n_peaks):
            n_peaks = [n_peaks for _ in range(len_xps)]
        self.n_peaks = n_peaks
        if Aux.is_float_or_int(first_peak_index):
            first_peak_index = [first_peak_index for _ in range(len_xps)]
        self.first_peak_index = first_peak_index
        if Aux.is_float_or_int(shift):
            self.shift = [shift for _ in range(len_xps)]

        # read or create dict_keys
        if Aux.is_list_or_tuple(dict_keys):
            self.dict_keys = dict_keys
        # if dict_keys not specified, try reading from xps
        elif isinstance(self.xps, dict):
            self.dict_keys = list(self.xps.keys())
        # default fallback: di where i = Xps object index
        else:
            self.dict_keys = ['d{}'.format(i) for i in range(len_xps)]

        # ensure xps is dict
        if Aux.is_list_or_tuple(self.xps):
            self.xps = dict(zip(self.dict_keys, self.xps))
            
        for dk in self.dict_keys:
            self.xps[dk].ds['index'] = dk
            
        if Aux.is_list_or_tuple(self.shift):
            self.shift = dict(zip(self.dict_keys, self.shift))
            
        if isinstance(shift, dict):
            for dk in self.dict_keys:
                self.xps[dk].shift(shift[dk])
            
        # ! ensure that this is kept updated if new peaks are added!
        if peak_ids is None:
            self.peak_ids = {}
            for i in range(len(self.dict_keys)):
                dk = self.dict_keys[i]
                self.peak_ids.update({dk: [j for j in range(self.first_peak_index[i], self.n_peaks[i], 1)]})
            
        # initialize lmfit parameters if params not specified
        if params is None:
            self.params = Parameters()
            self.init_peaks(params=self.params, 
                            n_peaks=self.n_peaks,
                            lineshape=lineshape,
                            first_peak_index=self.first_peak_index,
                            dict_keys=self.dict_keys)
            if be_guess is None:
                be_guess = np.finfo(float).eps
            self.guess_multi_component(params=self.params,
                                    param_id='center',
                                    # dict_keys=self.dict_keys,
                                    peak_ids=self.peak_ids,
                                    value=be_guess)
            self.guess_multi_component(params=self.params,
                                    param_id='amplitude',
                                    # dict_keys=self.dict_keys,
                                    peak_ids=self.peak_ids,
                                    value=1/max(self.n_peaks))
        else:
            self.params = params
            
        # apply expression constraints
        self.enforce_expr_constraints(params=self.params,
                                      expr_constraints=self.expr_constraints)
        
        # self.xps_concat_indices = [len(x.ds['be']) for x in list(self.xps.values)]
        # # self.xps_concat_indices = dict(zip(self.dict_keys, self.xps_concat_indices))
        # self.xps_concat = xr.concat([x.ds for x in list(self.xps.values())], dim='be')
        
        xps_concat_list = []
        self.xps_concat_indices = {}
        
        for dk in self.dict_keys:
            xps_dk_ds = self.xps[dk].ds
            xps_concat_list.append(xps_dk_ds)
            self.xps_concat_indices.update({dk: len(xps_dk_ds['be'])})
            
        self.xps_concat = xr.concat(xps_concat_list, dim='be')
        self.xps_concat['model_no_bg'] = self.xps_concat['cps_no_bg_norm']*0
        
        # TODO create empty columns for model_no_bg, residual, std_residual, and model if not already present
        
        if fit:
            self.fit(method=method)

    
    def plot_separate(self, 
                    subtract_bg=True, normalize=True,
                    display_bg=False, display_envelope=True, display_components=True, display_residuals=True,
                    text=None,
                    residual_ylim=False,
                    envelope_zorder=False,
                    colors=None, component_z_spec=False, xdim=None, ydim=3.25, energy_axis='be',
                    xlim=False,
                    yticks=False,
                    height_ratio=7.5, # ratio of main plot to residual plot
                    save_fig=False, ypad=0, ylabel=None, ax_kwargs=None, **kwargs):
        '''
        Args:
            display_residuals: If True or 'residuals', adds a plot of unscaled residuals. If 'std' or 'std_residuals', adds
                a plot of standardized residuals.
        '''
        if colors is None:
            colors = self.colors
        else:
            self.colors = colors
            
        if subtract_bg:
            bg_suffix = '_no_bg'
        else:
            bg_suffix = ''
            
        if normalize:
            norm_suffix = '_norm'
        else:
            norm_suffix = ''
        if envelope_zorder == 'top':
            self.envelope_zorder = 9999
        elif (envelope_zorder == 'bottom') or (envelope_zorder == 'bot'):
            self.envelope_zorder = 0
            
        if Aux.is_list_or_tuple(text):
            text = dict(zip(self.dict_keys, text))
            
        # includes leading underscore
        y_suffix = bg_suffix + norm_suffix
            
        if xdim is None:
            xdim = ydim*4/3
            
        if (ylabel is None) or (not isinstance(ylabel, str)):
            ylabel = 'Intensity'
        
        # generate separate subplot objects for each xarray dataset
        self.fig_dict = {}
        self.ax_dict = {}
        self.residual_ax_dict = {}
        for dk in self.dict_keys:
            # use gridspec to combine main and residual plots
            fig = plt.figure(layout='constrained')
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[height_ratio,1])
            ax, residual_ax = gs.subplots(sharex=True,sharey=False)
            ds_dk = self.xps[dk].ds
            
            xmin = float(ds_dk[energy_axis].min().data)
            xmax = float(ds_dk[energy_axis].max().data)
            
            ds_dk['cps'+y_suffix].plot.line(ax=ax, mec=self.data_color, marker=self.marker, ls='None',
                            ms=self.marker_size, mew=self.marker_edge_width, zorder=self.data_zorder)
            
            if display_bg and ('bg' in list(ds_dk.data_vars)):
                if subtract_bg:
                    ax.hlines(0, xmin, xmax,
                               colors=self.background_color,
                               linewidths=self.background_linewidth,
                               zorder=self.background_zorder)
                else:
                    ds_dk['bg'+norm_suffix].plot.line(ax=ax, color=self.background_color, linewidth=self.background_linewidth, zorder=self.background_zorder)
            
            if ('model' in list(ds_dk.data_vars)):
                if display_envelope:
                    ds_dk['model'+bg_suffix].plot.line(ax=ax, color=self.envelope_color, linewidth=self.envelope_linewidth, zorder=self.envelope_zorder)
                
                if display_components:
                    j = 0
                    for i in self.peak_ids[dk]:
                        ds_dk['p{}{}'.format(i, bg_suffix)].plot.line(ax=ax, color=colors[j], linewidth=self.component_linewidth)
                        j += 1

            # if not isinstance(ax_kwargs, dict):
            #     ax_kwargs = {}
            # self.ax_opts(ax, **ax_kwargs)
            # # ! overwrites manual specification of ylim
            # if ypad != False or (ypad != 0):
            #     ymin = min(df_key['cps'+y_suffix])
            #     ymax = max(df_key['cps'+y_suffix])
            #     ax.set_ylim([ymin-(ymax-ymin)*0.05,ymax*(1+ypad)])
            
            ax.set_ylabel(ylabel, fontsize=self.label_font_size)
            ax.tick_params(axis='both', which='major', labelsize=self.tick_font_size)
            ax.set_xlabel('')
            
            ylim = ax.get_ylim()
            dy = max(ylim) - min(ylim)
            if isinstance(text,str):
                ax.text(0.05,0.9,text,fontsize=self.label_font_size,transform = ax.transAxes, horizontalalignment='left', verticalalignment='center')
            if isinstance(text,dict):
                ax.text(0.05,0.9,text[dk],fontsize=self.label_font_size,transform = ax.transAxes, horizontalalignment='left', verticalalignment='center')
            # # if tight_layout:
            # #     plt.tight_layout()
            # j += 1
            
            if (display_residuals == True) or (display_residuals == 'residuals'):
                residual_prefix = ''
            elif (display_residuals == 'std') or (display_residuals == 'std_residuals'):
                residual_prefix = 'std_'
            if residual_ylim == False:
                if (display_residuals == True) or (display_residuals == 'residuals'):
                    residual_ylim = [-dy/2/height_ratio, dy/2/height_ratio]
                elif (display_residuals == 'std') or (display_residuals == 'std_residuals'):
                    residual_ylim = [-3, 3]
            
            residual_ax.axhline(y=0, color=self.envelope_color, linestyle='--', linewidth=self.axes_linewidth, alpha=0.5)
            if 'residual' in list(ds_dk.data_vars):
                ds_dk[residual_prefix+'residual'].plot.line(
                                ax=residual_ax, mec=self.data_color, marker=self.residual_marker, ls='None',
                                ms=self.marker_size, mew=self.marker_edge_width)
            residual_ax.set_ylabel('R', style='italic', fontsize=self.label_font_size)
            if Aux.is_list_or_tuple(xlim):
                residual_ax.set_xlim([min(xlim), max(xlim)])
            residual_ax.set_ylim(residual_ylim)
            residual_ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,2))
            # self.ax_opts(residual_ax, ylim=residual_lim, **ax_kwargs)
            if energy_axis == 'be':
                residual_ax.set_xlabel("Binding Energy (eV)", fontsize=self.label_font_size)
                residual_ax.invert_xaxis() # only need to invert one axis
            elif energy_axis == 'ke':
                residual_ax.set_xlabel("Kinetic Energy (eV)", fontsize=self.label_font_size)
            residual_ax.tick_params(axis='both', which='major', labelsize=self.tick_font_size)
            
            # fig.set_size_inches(xdim,ydim)
            # # if tight_layout:
            # #     plt.tight_layout()

            # if isinstance(save_fig,dict):
            #     fig.savefig(save_fig[key]+".svg")
            # if isinstance(save_fig,str):
            #     fig.savefig(save_fig+".svg")
                
            self.fig_dict[dk] = fig
            self.ax_dict[dk] = ax
            self.residual_ax_dict[dk] = residual_ax
    
    def fit(self, method='leastsq', **kwargs):
        self.result = minimize(self.residual, 
                                self.params, 
                                method=method,
                                kws={
                                    'dict_keys': self.dict_keys,
                                    'n_peaks': self.n_peaks[0]},
                                **kwargs)
        for dk in self.dict_keys:
            self.calculate_model_single_spectrum(self.xps[dk].ds, 
                                                 self.result.params, 
                                                 dk=dk, 
                                                 model=0,
                                                 n_peaks=self.n_peaks[0])
        # save component y-values
        for dk in self.dict_keys:
            ds_dk = self.xps[dk].ds
            for i in self.peak_ids[dk]:
                fn_params_ids = ['amplitude', 'center', 'sigma', 'gamma']
                fn_params = [self.result.params['{}_p{}_{}'.format(dk, i, j)] for j in fn_params_ids]
                fn = Fn.voigt(ds_dk['be'], *fn_params)
                ds_dk['p{}_no_bg'.format(i)] = fn
                ds_dk['p{}'.format(i)] = fn + ds_dk['bg_norm']
                
            
        
    def enforce_expr_constraints(self, params=None, expr_constraints=None):
        '''
        Args:
            params: Parameters instance. Defaults to Xps.params if None.
            expr_constraints: Expression constraints to be applied to Parameters instance.
                Syntax: {'parameter_1': {'value': value, ...}, 'parameter_2': {...}, ...}
        '''
        if params is None:
            params = self.params
        if expr_constraints is None:
            expr_constraints = {}
        expr_constraints_keys = list(expr_constraints.keys())
        for ek in expr_constraints_keys:
            params.add(ek, **expr_constraints[ek])
        
        
    def residual(self,
                 params,
                 xps=None,
                 dict_keys=None,
                 **kwargs):
        '''Objective function for lmfit.minimize. Generates a concatenated Xarray Dataset (xps_concat).'''
        if dict_keys is None:
            dict_keys = self.dict_keys
        if xps is None:
            xps = self.xps
        # for dk in dict_keys:
        #     self.calculate_model_single_spectrum(xps[dk].ds, 
        #                                          params, 
        #                                          dk=dk, 
        #                                          **kwargs)
        # self.xps_concat = xr.concat([x.ds for x in list(self.xps.values())], dim='be')
        self.calculate_model_concat_spectra(params, **kwargs)
        return self.xps_concat['residual']
        
    def calculate_model_concat_spectra(self,
                                       params,
                                        model=0,
                                        n_peaks=0):
        '''
        Wrapper for model_single_spectrum(). Creates new columns in ds for model and residuals.
        Always fits to normalized data.
        '''
        i = 0
        j = 0
        for dk in self.dict_keys:
            j += self.xps_concat_indices[dk]
            self.xps_concat['model_no_bg'][i:j] = self.model_single_spectrum(self.xps_concat['be'][i:j],
                                                        params,
                                                        dk=dk,
                                                        model=model,
                                                        n_peaks=n_peaks)
            i = j
        self.xps_concat['residual'] = self.xps_concat['cps_no_bg_norm'] - self.xps_concat['model_no_bg']
        # self.xps_concat['std_residual'] = self.xps_concat['residual']/self.xps_concat['residual'].std()
        self.xps_concat['model'] = self.xps_concat['model_no_bg'] + self.xps_concat['bg_norm']
        
    def calculate_model_single_spectrum(self,
                              ds,
                              params,
                              dk=None,
                              model=0,
                              n_peaks=0):
        '''
        Wrapper for model_single_spectrum(). Creates new columns in ds for model and residuals.
        Always fits to normalized data.
        '''
        ds['model_no_bg'] = self.model_single_spectrum(ds['be'],
                                                       params,
                                                       dk=dk,
                                                       model=model,
                                                       n_peaks=n_peaks)
        ds['residual'] = ds['cps_no_bg_norm'] - ds['model_no_bg']
        ds['std_residual'] = ds['residual']/ds['residual'].std()
        ds['model'] = ds['model_no_bg'] + ds['bg_norm']
        
    # TODO move to Fn class?
    # TODO add support for nonzero starting index
    def model_single_spectrum(self, 
                              be, 
                              params, 
                              dk=None, 
                              model=0, 
                              n_peaks=0,
                              first_peak_index=0):
        '''
        Models a single core-level region with a linear combination of Voigt components.
        
        Args:
            be: binding energy
            params: lmfit Parameters instance
            dk: key corresponding to xps entry in Fit instance
            model: constant shift applied to modelled data
            n_peaks: number of Voigt components in model
            first_peak_index: starting index of components in Parameters instance
        '''
        if dk is None:
            prefix_dk = ''
        else:
            prefix_dk = '{}_'.format(dk)
        for i in range(first_peak_index,n_peaks,1):
            prefix_dk_i = '{}p{}_'.format(prefix_dk, i)
            model += Fn.voigt(be, 
                              params[prefix_dk_i+'amplitude'],
                              params[prefix_dk_i+'center'],
                              params[prefix_dk_i+'sigma'],
                              params[prefix_dk_i+'gamma'])
        return model
        
    def guess_multi_component(self, 
                              params=None, 
                              param_id='center', 
                              peak_ids=None, 
                              value=np.finfo(float).eps, 
                              min=0, 
                              max=np.inf, 
                              vary=True,
                              dict_keys=None):
        '''
        Sets initial value and bounds [min, max] for a single fit parameter in an lmfit Parameters instace for 
        one or more core-level regions given in dict_keys.
        
        Args:
            params: lmfit Parameters instance.
            param_id: Parameter to be initialized.
            peak_ids: List of peak IDs. Possible inputs:
                Int or float: Specification of a single peak ID to be guessed across all regions in dict_keys.
                List of int or float: Specification of multiple peak IDs to be guessed across all regions in dict_keys.
                    Assumes specified peak IDs exist for all regions in dict_keys.
                Dict of list of int or float: Dictionary with entries corresponding to dict_keys. Each entry must be a list
                    containing one or more peak IDs to be guessed.
            value: Parameter value. Form analogous to peak_ids.
            min: Minimum parameter bound. Form analogous to peak_ids.
            max: Maximum parameter bound. Form analogous to peak_ids.
            vary: Fix or vary parameter value(s) from initial value.
        '''
        if params is None:
            params = self.params
        if peak_ids is None:
            peak_ids = self.peak_ids
            
        if dict_keys is None:
            dict_keys = self.dict_keys
        # if dict_keys is None:
        #     dict_keys = ['']
        #     dict_keys = self.dict_keys
        # elif not Aux.is_list_or_tuple(dict_keys):
        #     dict_keys = [dict_keys]
        
        # for handling multiple datasets
        peak_ids_dict = Aux.initialize_dict(peak_ids, dict_keys)
        value_dict = Aux.initialize_dict(value, dict_keys, ref_dict=peak_ids_dict)
        min_dict = Aux.initialize_dict(min, dict_keys, ref_dict=peak_ids_dict)
        max_dict = Aux.initialize_dict(max, dict_keys, ref_dict=peak_ids_dict)
        vary_dict = Aux.initialize_dict(vary, dict_keys, ref_dict=peak_ids_dict)

        # loop over all datasets
        for dk in dict_keys:
            peak_ids_dk = peak_ids_dict[dk]
            len_peak_ids = len(peak_ids_dk)
            value_dk = value_dict[dk]
            min_dk = min_dict[dk]
            max_dk = max_dict[dk]
            vary_dk = vary_dict[dk]
            # loop over all peak IDs
            for i in range(len_peak_ids):
                prefix = '{}_p{}'.format(dk,peak_ids_dk[i])
                self.guess_component_parameter(params=self.params, 
                                              param_id=param_id,
                                              value=value_dk[i],
                                              min=min_dk[i],
                                              max=max_dk[i],
                                              vary=vary_dk[i],
                                              prefix=prefix)
                
    
        
    def guess_component_parameter(self,
                                  params=None, 
                                  param_id='center', 
                                  value=np.finfo(float).eps, 
                                  min=0, 
                                  max=np.inf,
                                  vary=True,
                                  prefix=None, 
                                  **kwargs):
        '''
        Wrapper to set initial value and properties of a parameter in an lmfit Parameters instance.
        
        Args:
            params: lmfit Parameters instance.
            param_id: Parameter name.
            value: Parameter value.
            min: Minimum bound for parameter value.
            max: Maximum bound for parameter value.
            vary: Vary (True) or fix (False) parameter value during minimization.
            prefix: Prepended to parameter name with syntax prefix_param_id.
                Example: prefix='d0_p0' with param_id='center' specifies d0_p0_center.
        '''
        if params is None:
            params = self.params
        if prefix is None:
            prefix_dk = ''
        else:
            prefix_dk = '{}_'.format(prefix)
        if value is None:
            value = np.finfo(float).eps
        params[prefix_dk+param_id].set(value=value, min=min, max=max, vary=vary, **kwargs)
    
    
    # TODO options for 'all', align (peak-by-peak), and within (each spectrum)
    def constrain_all_gaussian_width(self,
                                    params=None,
                                    peak_ids='all',
                                    reference_peak_id=0,
                                    n_peaks=None,
                                    dict_keys=None,
                                    **kwargs):
        '''
        Wrapper for constrain_parameter_to_reference() with spec='sigma'.
        
        Args:
            peak_ids: List of component IDs, 'all' to constrain all, 'align' to constrain
                component-by-component across multiple spectra, and 'within' to constrain all
                components in each spectrum
        '''
        if params is None:
            params = self.params
        if dict_keys is None:
            dict_keys = self.dict_keys
        len_dict_keys = len(dict_keys)
        if n_peaks is None:
            n_peaks = self.n_peaks
        if Aux.is_float_or_int(n_peaks):
            n_peaks = [n_peaks for _ in range(len_dict_keys)]
        if Aux.is_float_or_int(peak_ids):
            peak_ids = [peak_ids]
        # TODO support n_peaks with different spec for each entry in dict_keys
        elif peak_ids == 'all':
            peak_ids = [i for i in range(reference_peak_id+1, n_peaks[0], 1)]
        self.constrain_parameter_to_reference(params=params,
                                              peak_ids=peak_ids,
                                              reference_peak_id=reference_peak_id,
                                              **kwargs)
        return
    
    
    def constrain_parameter_to_reference(self,
                                         params=None,
                                         param_id='sigma',
                                         peak_ids=None,
                                         reference_peak_id=0,
                                         reference_peak_key=None,
                                         dict_keys=None,
                                         vary=True,
                                         value=None,
                                         min=0,
                                         max=np.inf):
        if params is None:
            params = self.params
        if dict_keys is None:
            dict_keys = self.dict_keys
        # len_dict_keys = len(dict_keys)
        if Aux.is_float_or_int(peak_ids):
            peak_ids = [peak_ids]
        for dk in dict_keys:
            prefix_dk = '{}_'.format(dk)
            if reference_peak_key is None:
                reference_prefix = prefix_dk
            else:
                reference_prefix = '{}_'.format(reference_peak_key)
            if value is not None:
                params.add('{}p{}_{}'.format(reference_prefix,reference_peak_id,param_id),
                           value=value,
                           min=min,
                           max=max,
                           vary=vary)
            for peak_id in peak_ids:
                if (prefix_dk == reference_prefix) and (peak_id == reference_peak_id):
                    pass
                else:
                    params.add('{}p{}_{}'.format(prefix_dk,peak_id,param_id),
                            expr='{}p{}_{}'.format(reference_prefix,reference_peak_id,param_id))
        return
    
    def link_parameters(self,
                        full_param_ids,
                        reference_param=None):
        if reference_param is None:
            reference_param = full_param_ids[0]
        if isinstance(full_param_ids, str):
            full_param_ids = [full_param_ids]
        for id in full_param_ids:
            if id != reference_param:
                self.params.add(id, expr=reference_param)

    
    def constrain_parameter_pair(self,
                                 params=None,
                                 spec='ratio',
                                 param_id='amplitude',
                                 peak_ids=[1,0],
                                 value=1,
                                 dict_keys=None,
                                 vary=False,
                                 min=0,
                                 max=np.inf,
                                 param_min=0,
                                 param_max=np.inf):
        '''
        For each pair of peaks in peak_ids, constrain the ratio or spacing of the specified parameter (param_id).
        For each pair [a,b], generates a new parameter with name dk_pa_pb_[param_id]_[type].
        For peak a, the parameter expression is either:
            dk_pa_[param_id] = pa_pb_[param_id]_ratio * pb_[param_id]
            dk_pa_[param_id] = pa_pb_[param_id]_spacing + pb_[param_id]
        where dk is an entry in dict_keys.
        
        Args:
            params: Parameters object
            spec: 'ratio' or 'spacing'
            param_id: id of parameter to be constrained
            peak_ids: list of pairs of peaks to be constrained.
            value: single constraint value for all pairs or list of values for each pair
        '''
        if params is None:
            params = self.params
        if spec == 'ratio':
            op = '*'
        elif spec == 'spacing':
            op = '+'
        peak_ids = Aux.encapsulate(peak_ids)
        # TODO ensure this will work for constraining peaks across multiple Xps instances
        if dict_keys is None:
            dict_keys = self.dict_keys
        elif not Aux.is_list_or_tuple(dict_keys):
            dict_keys = [None]
        # else:
        if not Aux.is_list_or_tuple(value):
            value = [value for _ in range(len(peak_ids))]
        for dk in dict_keys:
            # prefix to append to parameter names
            # ignored if dict_keys not specified
            if dk == None:
                prefix_dk = ''
            else:
                prefix_dk = '{}_'.format(dk)
            # add constraint
            for i in range(len(peak_ids)):
                peak_ids_i = peak_ids[i]
                value_i = value[i]
                # generate ratio parameter
                params.add('{}p{}_p{}_{}_{}'.format(prefix_dk,peak_ids_i[0],peak_ids_i[1],param_id,spec),
                           value=value_i, 
                           vary=vary,
                           min=min,
                           max=max)
                # redefine parameter expression
                params.add('{}p{}_{}'.format(prefix_dk,peak_ids_i[0],param_id), 
                           expr='{}p{}_p{}_{}_{} {} {}p{}_{}'.format(prefix_dk,peak_ids_i[0],peak_ids_i[1],param_id,spec,
                                                                     op,
                                                                     prefix_dk,peak_ids_i[1],param_id),
                           min=param_min,
                           max=param_max)
                
    def constrain_ratio(self, params=None, **kwargs):
        '''Wrapper for constrain_parameter_pair() to constrain peak ratios.'''
        self.constrain_parameter_pair(params=params, spec='ratio', param_id='amplitude', **kwargs)
        
    def constrain_spacing(self, params=None, **kwargs):
        '''Wrapper for constrain_parameter_pair() to constrain peak spacings.'''
        self.constrain_parameter_pair(params=params, spec='spacing', param_id='center', **kwargs)
    
    def constrain_doublet(self, 
                          params=None,
                          peak_ids=[1,0],
                          ratio=0.5,
                          splitting=1,
                          dict_keys=None,
                          gamma_spacing_value=0,
                          gamma_spacing_min=0,
                          gamma_spacing_max=1,
                          constrain_gamma=True,
                          constrain_sigma=True,
                          sigma_value=0.5,
                          sigma_min=0,
                          sigma_max=2,
                          sigma_vary=True):
        '''
        Constrains one pair of doublet peaks. Wrapper for constrain_parameter_pair().
        
        Args:
            params: Parameters object
            peak_ids: pair of peaks to be constrained
            ratio: doublet type ('p','d','f') or manual specification (float or int)
            splitting: doublet splitting
            constrain_gamma: if True, gamma of the higher-binding energy peak will be 
                greater than that of the lower-energy peak. If 'match', gamma will be
                the same for both peaks.
        '''
        if params is None:
            params = self.params
        if ratio == 'p':
            ratio = 0.5
        elif ratio == 'd':
            ratio = 2/3
        elif ratio == 'f':
            ratio = 3/4
            
        if dict_keys is None:
            dict_keys = self.dict_keys
            
        # constrain peak ratio and spacing
        if ratio:
            self.constrain_ratio(params=params, peak_ids=peak_ids, value=ratio, dict_keys=dict_keys)
        if splitting:
            self.constrain_spacing(params=params, peak_ids=peak_ids, value=splitting, dict_keys=dict_keys)
        
        if not constrain_gamma:
            pass
        else:
            if constrain_gamma == 'match':
                vary_gamma = False
            else:
                vary_gamma = True
            self.constrain_parameter_pair(params=params, 
                                            spec='spacing',
                                            param_id='gamma',
                                            peak_ids=peak_ids,
                                            value=gamma_spacing_value,
                                            dict_keys=dict_keys,
                                            vary=vary_gamma,
                                            min=gamma_spacing_min,
                                            max=gamma_spacing_max)
        if constrain_sigma:
            self.constrain_all_gaussian_width(params=params,
                                                peak_ids=peak_ids,
                                                reference_peak_id=peak_ids[-1],
                                                n_peaks=2,
                                                dict_keys=dict_keys,
                                                vary=sigma_vary,
                                                value=sigma_value,
                                                min=sigma_min,
                                                max=sigma_max)
            
    def au_4f(self, value=84, min=70, max=100, display=False, **kwargs):
        self.constrain_doublet(peak_ids=[1,0],
                               ratio='f',
                               splitting=3.67,
                               **kwargs)
        self.guess_multi_component(param_id='center',
                                   peak_ids=0,
                                   value=value,
                                   min=min,
                                   max=max)
        self.fit()
        if display:
            self.plot_separate()


    def init_peak(self, 
                  params=None,
                  peak_id=0,
                  lineshape='voigt', 
                  prefix=None,
                  center=10**-8,
                  amplitude=None,
                  gamma=0.5,
                  sigma=0.5):
        '''Initializes parameters for a single peak.'''
        if params is None:
            params = self.params
        param_ids = ['center', 'amplitude']
            
        if lineshape == 'voigt':
            voigt_param_ids = ['sigma', 'gamma']
            param_ids += voigt_param_ids
            params._asteval.symtable['calculate_voigt_fwhm'] = Fn.calculate_voigt_fwhm
            
        if prefix == None:
            prefix_dk = ''
        else:
            prefix_dk = '{}_'.format(prefix)
            
        prefix_dk = '{}p{}_'.format(prefix_dk, peak_id)
        # TODO consolidate loops and logic...
        # guess amplitude based on initial Gaussian width
        # TODO guess amplitude based on integral of data
        if amplitude is None:
            amplitude = sigma*np.sqrt(2*np.pi)
        param_values = [center,amplitude,sigma,gamma]
        param_values = dict(zip(param_ids, param_values))
        for param_ids_i in param_ids:
            # default value > 0 to prevent division by zero
            if param_ids_i in voigt_param_ids:
                param_max = 2
            else:
                param_max = np.inf
            params.add(prefix_dk+param_ids_i, value=param_values[param_ids_i], min=0, max=param_max)
        for param_ids_i in param_ids:
            if lineshape == 'voigt':
                params.add(prefix_dk + 'fwhm',
                            expr="calculate_voigt_fwhm("+prefix_dk+"sigma,"+prefix_dk+"gamma)")
                
            
                
    def init_peaks(self, 
                   params=None, 
                   n_peaks=1,
                   lineshape='voigt',
                   first_peak_index=0,
                   dict_keys=None):
        '''
        Wrapper for init_peak to initialize multiple peaks.
        
        Args:
            params: lmfit Parameters instance.
            n_peaks: single specification for number of peaks or list of len(dict_keys).
            first_peak_index: starting index for peak IDs, either single specification
                or list of len(dict_keys).
            dict_keys: list of keys corresponding to list of Xps instances to be fitted.
        '''
        # if no spec for dict_keys, assume only one dataset
        if params is None:
            params = self.params
        if dict_keys is None:
            dict_keys = ['']
        len_dict_keys = len(dict_keys)
        # ensure n_peaks is iterable
        if Aux.is_float_or_int(n_peaks):
            n_peaks = [n_peaks for _ in range(len_dict_keys)]
        # ensure first_peak_index is iterable
        if Aux.is_float_or_int(first_peak_index):
            first_peak_index = [first_peak_index for _ in range(len_dict_keys)]
        # initialize parameters
        for i in range(len_dict_keys):
            dk = dict_keys[i]
            for peak_id in range(first_peak_index[i],n_peaks[i],1):
                self.init_peak(params=params, peak_id=peak_id, lineshape=lineshape, prefix=dk)
            
    
class Fn:
    
    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        return (amplitude/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-center)/sigma)**2)

    @staticmethod
    def lorentzian(x, amplitude, center, sigma):
        return amplitude*sigma/(np.pi*((x-center)**2+sigma**2))

    @classmethod
    def pseudo_voigt(cls, x, amplitude, center, sigma, fraction):
        return (1-fraction)*cls.gaussian(x,amplitude,center,sigma/np.sqrt(2*np.log(2))) + fraction*cls.lorentzian(x,amplitude,center,sigma)

    @staticmethod
    # borrowed from Andreas Seibert
    def voigt(x, amplitude, center, sigma, gamma):
        voigtfunction = np.sqrt(np.log(2))/(sigma*np.sqrt(np.pi)) * wofz((x-center)/sigma * np.sqrt(np.log(2)) + 1j * gamma/sigma * np.sqrt(np.log(2))).real
        return voigtfunction*amplitude
    
    @staticmethod
    # borrowed from Andreas Seibert
    def calculate_voigt_fwhm(sigma, gamma):
        fl = 2*gamma
        fg = 2*sigma
        return fl/2 + np.sqrt(np.power(fl,2)/4 + np.power(fg,2))
    

class Xps:
    
    def __init__(self, 
                 ds, 
                 be_range=None, 
                 method='area', 
                 shift=False,
                 **kwargs):
        self.ds = ds
        # get delta y before normalization (for shirley offset)
        ymin = self.ds.min(dim='be')['cps']
        ymax = self.ds.max(dim='be')['cps']
        self.delta_y_approx = float(ymax - ymin)
        # preprocessing
        if shift:
            self.shift(shift)
        if Aux.is_list_or_tuple(be_range):
            self.ds = self.ds.sel(be=slice(*be_range))
        # automatically fits background (bg) and stores background-subtracted data (cps_no_bg)
        if 'bg' not in list(self.ds.data_vars):
            self.fit_background(delta_y=self.delta_y_approx, **kwargs)
        # automatically computes normalized data:
        # cps_norm, bg_norm, and cps_no_bg_norm
        if 'cps_no_bg_norm' not in list(ds.data_vars):
            [self.delta_y, self.total_area] = Processing.normalize(self.ds, method=method)
            # if method == 'minmax':
            #     norm_constant = self.delta_y
            # elif method == 'area':
            #     norm_constant = self.total_area
            # print(ymin/norm_constant)
            # self.ds['cps_no_bg'] = self.ds['cps_no_bg'] - ymin/norm_constant
        
    @classmethod
    def from_vamas(cls, path=None, region_id=None, vamas_kwargs=None, **kwargs):
        if vamas_kwargs is None:
            vamas_kwargs = {}
        return cls(Vms.import_single_vamas(path=path, region_id=region_id, **vamas_kwargs), **kwargs)
    
    def shift(self, shift):
        self.ds['be'] = self.ds['be'] + shift
        self.ds['ke'] = self.ds['ke'] - shift
    
    def fit_background(self,
                       background='shirley',
                       break_point=None,
                       break_point_search_interval=None,
                       break_point_offset=0,
                       n_samples=[10, 10],
                       delta_y=None,
                       offset_by_delta_y=False,
                       **kwargs):
        '''
        Wrapper for Bg.shirley()
        
        Args:
            background: Background type ('shirley' or 'linear').
            break_point: Split data into 2 sub-regions at the specified binding energy.
                If 'search', will attempt to find a minimum on cps in break_point_search_interval
                and use the index of the minimum to define break_point.
            break_point_search_interval: Search interval if break_point is 'search'.
        '''
        if Aux.is_list_or_tuple(break_point_search_interval) and (break_point == 'search'):
            break_point_search_interval = [max(break_point_search_interval),
                                           min(break_point_search_interval)]
            break_point = float(self.ds.sel(be=slice(*break_point_search_interval)).cps.idxmin().data)
            print()
            print('Found break point at {} eV.'.format(break_point))
            
        if Aux.is_float_or_int(delta_y) and offset_by_delta_y:
            break_point_offset = break_point_offset*delta_y
            
        if Aux.is_float_or_int(break_point):
            ds_upper = self.ds.sel(be=slice(np.inf, break_point))
            ds_lower = self.ds.sel(be=slice(break_point, -np.inf))
            ds_list = [ds_upper, ds_lower]
            if (background == 'shirley') or (background == 's'):
                i = 0
                for ds in ds_list:
                    if i == 0:
                        y_offset = (0, break_point_offset)
                        n_samples_offset = (n_samples[0], 1)
                    else:
                        y_offset = (break_point_offset, 0)
                        n_samples_offset = (1, n_samples[1])
                    ds['bg'] = ('be', Bg.shirley(ds['cps'], y_offset=y_offset, n_samples=n_samples_offset, **kwargs))
                    if i > 0:
                        ds_list[i] = ds.drop_isel(be=0)
                    i += 1
                self.ds = xr.concat(ds_list, dim='be')
        else:
            if (background == 'shirley') or (background == 's'):
                self.ds['bg'] = ('be', Bg.shirley(self.ds['cps'], n_samples=n_samples, **kwargs))
        self.ds['cps_no_bg'] = ('be', (self.ds['cps'] - self.ds['bg']).data)

class Aux:
    '''
    Aux methods.
    '''
    @classmethod
    def return_entry(cls, x, k):
        if cls.is_list_or_tuple(x):
            return x[k]
        else:
            return x
        
    @staticmethod
    def is_list_or_tuple(x):
        return isinstance(x, list) or isinstance(x, tuple)
    
    @staticmethod
    def is_float_or_int(x):
        return isinstance(x, float) or isinstance(x, int)
    
    @classmethod
    def encapsulate(cls, x):
        if cls.is_float_or_int(x[0]):
            return [x]
        else:
            return x
        
    @classmethod
    def initialize_dict(cls, x, dict_keys, len_list=1, ref_dict=None):         
        if (cls.is_float_or_int(x) or (x is None)) and isinstance(ref_dict, dict):
            x_dict = {}
            for dk in dict_keys:
                x_dict.update({dk: [x for _ in range(len(ref_dict[dk]))]})
            return x_dict
        
        if (cls.is_float_or_int(x) or (isinstance(x, bool)) or (x is None)):
            x = [x for _ in range(len_list)]
        
        if cls.is_list_or_tuple(x):
            x_dict = {}
            x_dict.update({dk: x for dk in dict_keys})
        elif isinstance(x, dict):
            x_dict = x
        return x_dict
     
class Processing:
    @staticmethod
    def normalize(ds, method='area', norm_range=None, y='cps_no_bg'):
        ymin = ds.min(dim='be')[y]
        ymax = ds.max(dim='be')[y]
        
        delta_y = float(ymax - ymin)
        total_area = abs(trapezoid((ds[y] - ymin), x=ds['be']))
        
        if method == 'minmax':
            norm_constant = delta_y
        if method == 'area':
            norm_constant = total_area
            
        for y in ['bg','cps','cps_no_bg']:
            if y in list(ds.data_vars):
                ds['{}_norm'.format(y)] = ds[y]/norm_constant
        return delta_y, total_area

class Bg:
    '''
    Methods to fit backgrounds to XPS (and other) data.
    '''
    @staticmethod
    # stolen from pyARPES
    def shirley(
        xps: np.ndarray, eps=1e-7, max_iters=500, n_samples=(5,5), y_offset=(0,0),
    ) -> np.ndarray:
        """Core routine for calculating a Shirley background on np.ndarray data."""
        background = np.copy(xps)
        cumulative_xps = np.cumsum(xps, axis=0)
        total_xps = np.sum(xps, axis=0)

        rel_error = np.inf

        i_left = np.mean(xps[:n_samples[0]], axis=0) + y_offset[0]
        i_right = np.mean(xps[-n_samples[1]:], axis=0) + y_offset[1]

        iter_count = 0

        k = i_left - i_right
        for iter_count in range(max_iters):
            cumulative_background = np.cumsum(background, axis=0)
            total_background = np.sum(background, axis=0)

            new_bkg = np.copy(background)

            for i in range(len(new_bkg)):
                new_bkg[i] = i_right + k * (
                    (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                    / (total_xps - total_background + 1e-5)
                )

            rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

            background = new_bkg

            if np.any(rel_error < eps):
                break

        if (iter_count + 1) == max_iters:
            warnings.warn(
                "Shirley background calculation did not converge "
                + "after {} steps with relative error {}!".format(max_iters, rel_error)
            )
            
        return background


class Vms:
    '''
    Methods to import and parse VAMAS data.
    '''

    @staticmethod
    def check_vamas_input(path=None, vms=False):
        '''
        Check whether a path to VAMAS data is specified, a VAMAS object is passed, or an invalid input is given.
        
        Args:
            path: Path to VAMAS data, including extension.
            vms: Vamas object.
            
        Returns:
            Vamas object.
        '''
        if isinstance(path,str) and (path[-4:] == ".vms"):
            vms = Vamas(path)
        elif vms:
            pass
        else:
            warnings.warn(
                'Data format is not VAMAS.'
            )
            return
        return vms

    @classmethod
    def read_vamas_blocks(cls, path=None, vms=False):
        '''
        Wrapper for Vamas() to read VAMAS block IDs.
        If path is specified, VAMAS data are loaded.
        Alternatively, a Vamas object (vms) can be passed directly.
        
        Args:
            path: Path to VAMAS data, including extension.
            vms: Vamas object.
            
        Returns:
            List of VAMAS block IDs.
        '''
        vms = cls.check_vamas_input(path=path, vms=vms)
        return [vms.blocks[k].block_identifier for k in range(len(vms.blocks))]

    @classmethod
    def import_single_vamas(cls,
                            path=None,
                            vms=False,
                            region_id=None,
                            read_phi=False,
                            verbose=True):
            '''
            Wrapper for Vamas() to import XPS data into an Xarray Dataset.
            If path is specified, VAMAS data are loaded.
            Alternatively, a Vamas object (vms) can be passed directly.
            
            Args:
                path: Path to VAMAS data, including extension.
                region_id: Identifier for VAMAS block. Can be string (e.g., 'Au 4f 690') or integer index of block (more robust).
                    # TODO Identify when multiple VAMAS blocks share the same name and prompt user to select.
                read_phi: If True, reads the analyzer work function from VAMAS data and factors it into the binding energy calculation.
                
            Returns:
                Xarray dataset with variables ke (kinetic energy) and cps (raw counts) and coordinate be (binding energy).
            
            # ! Data processing options (normalize, shift) have been moved downstream.
            '''

            vms = cls.check_vamas_input(path=path, vms=vms)

            # check spectra contained in VAMAS by pulling block_identifier for each block
            ids = cls.read_vamas_blocks(vms=vms)
            n = len(ids)
            
            if verbose:
                print('Found ' + str(len(ids)) + ' blocks.')
                print()
            
            # check for and log duplicate block IDs
            duplicate_flag = n != len(set(ids))
            duplicate_ids = []
            if duplicate_flag:
                if verbose:
                    print('Detected multiple VAMAS blocks with the same name.')
                    print()
                duplicate_ids = [id for id in ids if ids.count(id) > 1]
            
            # if region_id not specified, get user input
            if (region_id == None) or (not region_id):
                print(pd.DataFrame(ids, columns=['id']))
                region_id = input('Specify block ID or index to access...')
                try:
                    region_id = int(region_id)
                except ValueError:
                    pass
            
            # block index specified directly
            if isinstance(region_id, int):
                idx = region_id
            # get block index from block ID
            elif isinstance(region_id, str):
                if region_id in duplicate_ids:
                    print('String specification is ambiguous.')
                    print('Defaulting to first instance.')
                idx = ids.index(region_id)
                
            # access spectrum, pull counts (c), generate KE range, and calculate BE from KE
            dataBlock = vms.blocks[idx]
            cps = dataBlock.corresponding_variables[0].y_values # ! assumes counts are always the first corresponding variable...
            n = len(cps)
            ke = np.linspace(dataBlock.x_start, dataBlock.x_start + dataBlock.x_step*(n-1), n)
            
            # analyzer work function
            if read_phi:
                phi = dataBlock.analyzer_work_function_or_acceptance_energy
            else:
                phi = 0
            # excitation energy
            hv = dataBlock.analysis_source_characteristic_energy
            if verbose:
                print("Excitation energy:")
                print(str(hv) + " eV")

            
            # compute binding energy
            be = hv - ke - phi
            
            return xr.Dataset(
                data_vars=dict(
                    cps=('be', cps),
                    ke=('be', ke)
                ),
                coords=dict(
                    be=('be', be)
                )
            )
        
        
'''
Deprecated functions and classes
'''

def get_data(file, ids, be_range, dict_keys, region, shift=0, suffix=None):
    path = [file for k in range(len(ids))]
    if isinstance(suffix,str):
        region = region+suffix
    return {region: Pes.from_vamas(path, region_id=ids, be_range=be_range, dict_keys=dict_keys, shift=shift)}

def process_data(file, region, blocks, dict_keys, data, n_peaks, expr_constraints=None, shifts=None, flag=None, be_range=None,
                 plot_survey=False, fit_data=False, be_guess=None, n_samples=[5,5],
                 peak_spacings=None, peak_ratios=None, bg_midpoint=None, ids_subset=None, suffix=None,
                 sigma_max=1.5, gamma_max=1.5, plot_kwargs=None, **kwargs):
    if be_range == None:
        be_range = [0,9999]
    ids = [k for k, s in enumerate(blocks) if region in s]
    if Pes.is_list_or_tuple(ids_subset):
        ids = itemgetter(*ids_subset)(ids)
    if isinstance(shifts, dict):
        shifts = [shifts[key] for key in dict_keys]
    data_dict = get_data(file, ids, be_range, dict_keys, region, shift=shifts, suffix=suffix)
    if isinstance(suffix,str):
        region = region + suffix
    data.update(data_dict)
    if plot_survey:
        if not isinstance(plot_kwargs, dict):
            plot_kwargs = {}
        data[region].plot_survey(**plot_kwargs)
    if fit_data:
        data[region].set_n_peaks(n_peaks)
        data[region].generate_params(be_guess=be_guess, expr_constraints=expr_constraints, shirley_kwargs={'n_samples': n_samples},
                                        sigma_max=sigma_max, gamma_max=gamma_max, match_sigma=True,
                                        peak_spacings=peak_spacings,
                                        peak_ratios=peak_ratios, **kwargs)
        data[region].fit_data(bg_midpoint=bg_midpoint)
        
def match_spacings(peak_ids, dict_keys, expr_constraints, spacing_guess, spacing_min=None, spacing_max=None):
    if not Pes.is_list_or_tuple(spacing_min):
       spacing_min = [0 for k in range(len(peak_ids))]
    if not Pes.is_list_or_tuple(spacing_max):
       spacing_max = [np.inf for k in range(len(peak_ids))] 
    for key in dict_keys:
        k = 0
        for id in peak_ids:
            expr_constraints.update({'data_{}_p{}_center'.format(key,id): {'expr': 'data_{}_p{}_p0_spacing+data_{}_p0_center'.format(key,id,key)}})
            if id != 0:
                if key != dict_keys[0]:
                    expr_constraints.update({'data_{}_p{}_p0_spacing'.format(key,id): {'expr': 'data_{}_p{}_p0_spacing'.format(dict_keys[0],id)}})
                else:
                    expr_constraints.update({'data_{}_p{}_p0_spacing'.format(key,id): {'value': spacing_guess[k],
                                                                                       'min': spacing_min[k], 'max': spacing_max[k]}})
                    k += 1
                    
def constrain_doublet_gamma(peak_ids, dict_keys, expr_constraints, spacing_value=None, spacing_min=None, spacing_max=None):
    '''
    peak_ids: list only id of first peak in doublet!
    '''
    if not Pes.is_list_or_tuple(spacing_min):
       spacing_min = [0 for k in range(len(peak_ids))]
    if not Pes.is_list_or_tuple(spacing_max):
       spacing_max = [np.inf for k in range(len(peak_ids))] 
    for key in dict_keys:
        k = 0
        for id in peak_ids:
            expr_constraints.update({'data_{}_p{}_p{}_gamma_spacing'.format(key,id+1,id): {'value': np.average([spacing_min[k],spacing_max[k]]),
                                                                                           'min': spacing_min[k], 'max': spacing_max[k]}})
            expr_constraints.update({'data_{}_p{}_gamma'.format(key,id+1): {'expr': 'data_{}_p{}_p{}_gamma_spacing+data_{}_p{}_gamma'.format(key,id+1,id,key,id),
                                                                            'min': 0}})
            k += 1
                    
def gaussian_only(peak_ids, dict_keys, expr_constraints):
    for key in dict_keys:
        for id in peak_ids:
            expr_constraints.update({'data_{}_p{}_gamma'.format(key,id): {'value': 0, 'vary': False}})
            
def align_gamma(peak_ids, dict_keys, expr_constraints, gamma_min=0, gamma_max=2):
    for key in dict_keys[1:]:
        for id in peak_ids:
            expr_constraints.update({'data_{}_p{}_gamma'.format(key,id): {'expr': 'data_{}_p{}_gamma'.format(dict_keys[0],id), 'min': gamma_min, 'max': gamma_max}})

# TODO improve default params
# TODO clean up parameter specifications
# TODO add support for kwargs
# TODO take dict of PES dataframes or Pes object as input instead?
def get_au4f_shift(path, region_id, be_range=None, be_guess=[83,0], background='shirley', shirley_kwargs={'n_samples': [5,5]}, plot_result=True, verbose=False):
    au4f = Pes.from_vamas(path, region_id=region_id, be_range=be_range)
    au4f.set_n_peaks(2)
    au4f.background = background
    if background != 'shirley':
        shirley_kwargs = None
    au4f.generate_params(be_guess=be_guess, peak_spacings=[1,0,3.67], peak_ratios=[1,0,0.75])
    au4f.fit_data(shirley_kwargs=shirley_kwargs)
    if plot_result:
        au4f.plot_result()
    if verbose:
        param_list = ['lfwhm','gfwhm','fwhm','glmix']
        for param in param_list:
            print(au4f.result.params['data_0_p0_'+param])
    return 84 - au4f.result.params['data_0_p0_center'].value

def get_au4f_gfwhm(path, region_id, be_range=None, be_guess=[83,0], background='shirley', shirley_kwargs={'n_samples': [5,5]}, plot_result=True):
    au4f = Pes.from_vamas(path, region_id=region_id, be_range=be_range)
    au4f.set_n_peaks(2)
    au4f.background = background
    if background != 'shirley':
        shirley_kwargs = None
    au4f.generate_params(be_guess=be_guess, peak_spacings=[1,0,3.67], peak_ratios=[1,0,0.75])
    au4f.fit_data(shirley_kwargs=shirley_kwargs)
    if plot_result:
        au4f.plot_result()
    return au4f.result.params['data_0_p0_gfwhm'].value

# TODO employ iteration if dataframes have same range but different eV step sizes
def average_all_dataframes(df_dict, weights=None):
    '''Averages the cps columns of multiple equal-length PES dataframes'''
    keys_list = list(df_dict.keys())
    df_cols = ['ke', 'be', 'cps']
    # initialize empty dataframe
    df = pd.DataFrame(dict(zip(df_cols, [[] for _ in range(len(df_cols))])))
    cps_arr = []
    for key in keys_list:
        cps_arr.append(df_dict[key]['cps'])
        
    cps_arr = np.array(cps_arr)
    cps_avg = np.average(cps_arr, axis=0, weights=weights)

    # output average
    for i in range(2):
        df[df_cols[i]] = df_dict[keys_list[0]][df_cols[i]]
    df['cps'] = cps_avg
    return df

def average_dataframes(df_dict, step, start=None, stop=None, keys_list=None, weights=None):
    '''Averages the cps columns of equal-length PES dataframes in groups of size step in the range from start to stop, inclusive'''
    if start == None:
        start = 0
    if stop == None:
        stop = len(df_dict)
    if keys_list == None:
        keys_list = [str(i) for i in range(ceil((stop-start)/step))]
    avg_df_dict = {}
    df_dict_keys = list(df_dict.keys())
    i0 = 0
    k = 0
    for i in range(start+step, stop, step):
        avg_df_dict.update({keys_list[k]: average_all_dataframes({df_dict_keys[j]: df_dict[df_dict_keys[j]] for j in range(i0, i, 1)}, weights=weights)})
        i0 = np.copy(i)
        k += 1
    return avg_df_dict

# TODO clean up passing of kwargs to different methods
# TODO add support for filetypes other than VAMAS

class Pes:
    
    colors = [[98/255,146/255,190/255], # mid-blue
        [64/255,105/255,149/255], # dark blue
        [133/255,127/255,188/255], # sour purple
        [66/255,69/255,133/255], # dark purple
        [250/255,172/255,116/255], # light orange
        [194/255,104/255,47/255], # burnt orange
        [247/255,168/255,170/255], # strawberry pink
        [240/255,79/255,82/255]] # red
    envelope_color = "#000000"
    data_color = "#000000"
    background_color = "gray"
    
    # fitting options
    background = 'shirley'
    params = {}
    
    def __init__(self, df_dict, n_peaks=1):
        '''Generate PES dataframe(s).'''
        if isinstance(df_dict, dict):
            self.df_dict = df_dict
            self.keys_list = list(df_dict.keys())
        elif self.is_list_or_tuple(df_dict):
            self.keys_list = [str(i) for i in range(len(df_dict))]
            self.df_dict = dict(zip([self.keys_list, df_dict]))
        self.set_n_peaks(n_peaks)
        self.set_class_plot_config()
        
    def plot_survey(self, keys_list=None, xdim=3.25*4/3, ydim=3.25, ax_kwargs=None, save_fig=False, stack_spectra=False, offset=0, normalize=False, energy='be',
                    norm_sample_range=None, fit_bg=False, shirley_kwargs=None, n_samples=None, bg_midpoint=None, subtract_bg=False, norm_kwargs=None, colors=colors,
                    y_label='Intensity', label_font_size=12, tick_font_size=12, **kwargs):
        '''Plot survey spectrum for each PES dataframe.'''
        if not isinstance(ax_kwargs, dict):
            ax_kwargs = {}
        if keys_list == None:
            keys_list = self.keys_list
        if stack_spectra:
            self.survey_fig, self.survey_ax = plt.subplots(layout="constrained")
        if fit_bg:
            self.fit_background(shirley_kwargs=shirley_kwargs, n_samples=n_samples, bg_midpoint=bg_midpoint)
        if normalize:
            if not isinstance(norm_kwargs,dict):
                norm_kwargs = {}
            self.normalize(**norm_kwargs)
        if subtract_bg and fit_bg:
            if normalize:
                y = 'cps_no_bg_norm'
            else:
                y = 'cps_no_bg'
        else:
            y = 'cps'
        i = 0
        for key in keys_list:
            df = self.df_dict[key]
            if not stack_spectra:
                self.survey_fig, self.survey_ax = plt.subplots(layout="constrained")
                color = 'k'
            else:
                color = colors[i]
            # sns.lineplot(data=self.df_dict[key], x='be', y=y, 
            #     ax=self.survey_ax, color=color, **kwargs)
            self.survey_ax.plot(df[energy], df[y]+offset*i, color=color)
            self.ax_opts(self.survey_ax, **ax_kwargs)
            self.survey_fig.set_size_inches(xdim,ydim)
            self.survey_ax.set_ylabel(y_label, fontsize=label_font_size)
            self.survey_ax.set_xlabel("Binding Energy (eV)", fontsize=label_font_size)
            self.survey_ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
            self.survey_ax.invert_xaxis()
            
            if isinstance(save_fig,dict):
                self.survey_fig.savefig(save_fig[key]+".svg")
            if isinstance(save_fig,str):
                self.survey_fig.savefig(save_fig+".svg")
                
            i += 1
            
    # TODO implement sample_range
    def normalize(self, mode='minmax', sample_range=None, y='cps_no_bg'):
        for key in self.keys_list:
            df = self.df_dict[key]
            ymin = min(df[y])
            ymax = max(df[y])
            if self.is_list_or_tuple(sample_range):
                df_trimmed = df.loc[(df['be'] <= max(sample_range)) & (df['be'] >= min(sample_range))]
                ymin = min(df_trimmed[y])
                ymax = max(df_trimmed[y])
                print(ymax)
            if mode == 'minmax':
                self.df_dict[key]['{}_norm'.format(y)] = (df[y]-ymin)/(ymax-ymin)
            if mode == 'area':
                # need abs() due to ordering of be values
                self.df_dict[key]['{}_norm'.format(y)] = (df[y]-ymin)/abs(trapezoid(df[y]-ymin, x=df['be']))
    
    # stopgap solution for cases when vamas readout breaks
    @classmethod
    def from_casa_ascii(cls, path, be_range=None, shift=None, n_peaks=1):
        df = pd.read_table(path, skiprows=7, usecols=[0,1], names=['be','cps'])
        if cls.is_float_or_int(shift):
            df['be'] = df['be'] + shift
        if cls.is_list_or_tuple(be_range):
            be_range_idx = df[(df['be'] <= min(be_range)) | (df['be'] >= max(be_range))].index
            df.drop(be_range_idx, inplace=True)
        
        return cls({'0': df}, n_peaks=n_peaks)
    
    # TODO: add automatic matching of partial region_id (e.g., C = C1 1 or S 2p = S2p/Se3p)
    # TODO optimize read times if the same vamas file is referenced multiple times
    # ! if reading multiple spectra from one file, path must be same length as region_id
    @classmethod
    def from_vamas(cls, path, region_id=None, be_range=None, read_phi=False, shift=None, 
                     dict_keys=None, n_peaks=1, **kwargs):
        '''Alternate constructor to import PES dataframes from VAMAS data.'''
        df_dict = {}
        
        # if single path is specified, repack into list
        if not cls.is_list_or_tuple(path):
            path = [path]
        
        for k in range(len(path)):
            
            region_id_k = cls.return_entry(region_id, k)
                
            if be_range == None or (be_range == False):
                be_range_k = (-np.inf,np.inf)
            elif cls.is_list_or_tuple(be_range[0]):
                be_range_k = be_range[k]
            elif cls.is_float_or_int(be_range[0]):
                be_range_k = be_range
            else:
                be_range_k = (-np.inf,np.inf)
            
            shift_k = cls.return_entry(shift, k)
            if shift_k == None or (shift_k == False):
                shift_k = 0
                
            read_phi_k = cls.return_entry(read_phi, k)
            
            key = cls.return_entry(dict_keys, k)
            if key == None:
                key = str(k)
                
            df = cls.import_single_vamas(path[k], region_id=region_id_k, shift=shift_k,
                                    be_range=be_range_k, read_phi=read_phi_k, **kwargs)
            df_dict.update({key: df})
        return cls(df_dict, n_peaks=n_peaks)
    
    @staticmethod
    def import_single_vamas(path, region_id=None,
                # data processing options
                shift=None,
                be_range=None,
                read_phi=False,
                normalize=False):
        '''Wrapper for Vamas() to extract a single PES dataframe from VAMAS data.'''


        if path[-4:] == ".vms":
            # pull VAMAS data
            data = Vamas(path)
            
            # check spectra contained in VAMAS by pulling block_identifier for each block
            ids = [data.blocks[k].block_identifier for k in range(len(data.blocks))]
            
            print('Found ' + str(len(ids)) + ' blocks')
            
            # if spectrum was not specified, prompt user to select
            if isinstance(region_id, int):
                idx = region_id
            elif region_id == None or (region_id == False):
                print()
                print(ids)
                region_id = input('Specify spectrum ID to access...')
                print(region_id)

            # get block index of desired spectrum
            if region_id in ids:
                idx = ids.index(region_id)
                
            # access spectrum, pull counts (c), generate KE range, and calculate BE from KE
            dataBlock = data.blocks[idx]
            cps = dataBlock.corresponding_variables[0].y_values # assumes counts are always the first corresponding variable...
            n = len(cps)
            ke = np.linspace(dataBlock.x_start, dataBlock.x_start + dataBlock.x_step*(n-1), n)
            
            # analyzer work function
            phi = dataBlock.analyzer_work_function_or_acceptance_energy
            # excitation energy
            hv = dataBlock.analysis_source_characteristic_energy
            print("Excitation energy:")
            print(str(hv) + " eV")
        else:
            warnings.warn(
                "Data format is not VAMAS! Aborting :C"
            )
            return

        # calculate binding energy
        if read_phi == None or (read_phi == False):
            phi = 0
            
        # shift kinetic energy to compensate for charging and/or hv error
        ke = ke - shift
        # compute binding energy after shift correction
        be = hv - ke - phi
        
        # note: trimming is done AFTER shift correction!
        trim = np.logical_and(be >= min(be_range), be <= max(be_range))
        ke = np.trim_zeros(ke*trim)
        be = np.trim_zeros(be*trim)
        cps = np.trim_zeros(cps*trim)
        
        if normalize:
            cps = cps/(max(cps)-min(cps))
        
        return pd.DataFrame(data={"ke": ke, "be": be, "cps": cps})
    
    @classmethod
    def return_entry(cls, x, k):
        if cls.is_list_or_tuple(x):
            return x[k]
        else:
            return x
        
    @staticmethod
    def is_list_or_tuple(x):
        return isinstance(x, list) or isinstance(x, tuple)
    
    @staticmethod
    def is_float_or_int(x):
        return isinstance(x, float) or isinstance(x, int)
    
    # lineshapes
    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        return (amplitude/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-center)/sigma)**2)

    @staticmethod
    def lorentzian(x, amplitude, center, sigma):
        return amplitude*sigma/(np.pi*((x-center)**2+sigma**2))

    @classmethod
    def pseudo_voigt(cls, x, amplitude, center, sigma, fraction):
        return (1-fraction)*cls.gaussian(x,amplitude,center,sigma/np.sqrt(2*np.log(2))) + fraction*cls.lorentzian(x,amplitude,center,sigma)

    @staticmethod
    # borrowed from Andreas Seibert
    def voigt(x, amplitude, center, sigma, gamma):
        voigtfunction = np.sqrt(np.log(2))/(sigma*np.sqrt(np.pi)) * wofz((x-center)/sigma * np.sqrt(np.log(2)) + 1j * gamma/sigma * np.sqrt(np.log(2))).real
        return voigtfunction*amplitude
    
    @staticmethod
    # borrowed from Andreas Seibert
    def calculate_voigt_fwhm(sigma, gamma):
        fl = 2*gamma
        fg = 2*sigma
        return fl/2 + np.sqrt(np.power(fl,2)/4 + np.power(fg,2))
    
    def fit_background(self, shirley_kwargs=None, n_samples=None, bg_midpoint=None):
        for key in self.keys_list:
            min_be = min(self.df_dict[key]['be'])
            max_be = max(self.df_dict[key]['be'])
            # print(self.df_dict[key]['be'])
            self.df_dict[key]['bg'] = 0.0
            if self.background == 'shirley':
                if not isinstance(shirley_kwargs, dict):
                    shirley_kwargs = {}
                if self.is_list_or_tuple(n_samples):
                    shirley_kwargs['n_samples'] = n_samples
                if self.is_float_or_int(bg_midpoint):
                    df_lower = self.df_dict[key].loc[(self.df_dict[key]['be'] <= bg_midpoint)]
                    df_upper = self.df_dict[key].loc[(self.df_dict[key]['be'] > bg_midpoint)]
                    # TODO at bg_midpoint force nsamples=1
                    bg_lower = self.calculate_shirley_background(np.array(df_lower['cps'],dtype='float64'), **shirley_kwargs)
                    bg_upper = self.calculate_shirley_background(np.array(df_upper['cps'],dtype='float64'), **shirley_kwargs)
                    self.df_dict[key].loc[(self.df_dict[key]['be'] <= bg_midpoint), 'bg'] = bg_lower
                    self.df_dict[key].loc[(self.df_dict[key]['be'] > bg_midpoint), 'bg'] = bg_upper
                else:
                    self.df_dict[key]['bg'] = self.calculate_shirley_background(self.df_dict[key]['cps'], **shirley_kwargs)
            elif self.background == 'linear':
                bgx = np.concatenate((self.df_dict[key]['be'][0:n_samples[0]],
                        self.df_dict[key]['be'][-n_samples[1]:]))
                bgy0 = np.concatenate((self.df_dict[key]['cps'][0:n_samples[0]],
                            self.df_dict[key]['cps'][-n_samples[1]:]))
                bgmodel = LinearModel(prefix="data_"+key+"_bg_")
                bgparams = bgmodel.make_params()
                bgresult = bgmodel.fit(bgy0, bgparams, x=bgx)
                self.df_dict[key]['bg'] = bgresult.params["data_"+key+"_bg_intercept"] + bgresult.params["data_"+key+"_bg_slope"]*self.df_dict[key]['be']
            self.df_dict[key]['cps_no_bg'] = self.df_dict[key]['cps'] - self.df_dict[key]['bg']
    
    def fit_data(self, lineshape="voigt", bgdata=None, fit_bg=True, shirley_kwargs=None, n_samples=None, bg_midpoint=None):
        if fit_bg:
            self.fit_background(shirley_kwargs=shirley_kwargs, n_samples=n_samples, bg_midpoint=bg_midpoint)
                
        self.result = minimize(self.residual, self.params)
        
        start_index = 0
        for key in self.keys_list:
            df_key = self.df_dict[key]
            x_key = np.array(df_key['be'])
            df_key['fit_no_bg'] = self.generate_model_single_spectrum_no_bg(self.result.params, key, x_key)
            df_key['fit'] = df_key['fit_no_bg'] + df_key['bg']
            # TODO implement lineshapes other than voigt
            for peak_number in range(self.n_peaks[key]):
                peak_id = "data_{}_p{}_".format(key, peak_number)
                df_key['p{}_no_bg'.format(peak_number)] = self.voigt(x_key, 
                                                                    self.result.params[peak_id+'amplitude'].value,
                                                                    self.result.params[peak_id+'center'].value,
                                                                    self.result.params[peak_id+'sigma'].value,
                                                                    self.result.params[peak_id+'gamma'].value)
                df_key['p{}'.format(peak_number)] = df_key['p{}_no_bg'.format(peak_number)] + df_key['bg']
            # slice and store residuals
            end_index = start_index + len(df_key)
            df_key['residuals'] = self.result.residual[start_index:end_index]
            df_key['std_residuals'] = df_key['residuals']/np.std(df_key['residuals'])
            start_index = np.copy(end_index)
        
        # generate normalized data
        amplitudes = []
        j = 0
        for key in self.keys_list:
            norm_params = ['cps_no_bg','fit_no_bg'] + ['p{}_no_bg'.format(i) for i in range(self.n_peaks[key])]
            amplitude = 0
            # calculate total area by summing component areas
            for i in range(self.n_peaks[key]):
                amplitude += self.result.params['data_{}_p{}_amplitude'.format(key,i)].value
            amplitudes.append(amplitude)
            for param in norm_params:
                self.df_dict[key][param+'_norm'] = self.df_dict[key][param]/amplitudes[j]
            j += 1
    
    def residual(self, params, keys_list=None, lineshape="voigt", bgdata=None, *args, **kwargs):
        residuals = np.array([])
        # if keys_list == None:
        #     keys_list = self.keys_list
        for key in self.keys_list:
            # print(keys_list)
            # TODO figure out why keys_list gets turned into Parameters object..
            x_key = np.array(self.df_dict[key]['be'])
            y_key = np.array(self.df_dict[key]['cps_no_bg'])

            resid = (y_key - self.generate_model_single_spectrum_no_bg(params, key, x_key, **kwargs))/(len(x_key)*np.linalg.norm(x_key))
            residuals = np.append(residuals, resid)
        return residuals
    
    def generate_params(self, be_guess, keys_list=None,
                        # general parameters
                        lineshape="voigt",
                        peak_ratios=False, peak_spacings=False,
                        expr_constraints=None,
                        # voigt/pseudo-voigt specific parameters (will not apply to pure gaussian or lorentzian peaks)
                        align_peaks=None, match_fwhm=None, fwhm_max=3, fwhm_min=0,
                        glmix_guess=0.5, glmix_max=1, glmix_min=0, glmix_vary=True, match_glmix=False,
                        sigma_guess=1, sigma_max=np.inf, sigma_min=0, sigma_vary=True, match_sigma='within',
                        gamma_guess=1, gamma_max=np.inf, gamma_min=0, gamma_vary=True, match_gamma=False,
                        *args, **kwargs):
        '''
        match:
            False: all varied independently
            'within': peaks within a single spectrum are matched to p0 in that spectrum
            True: all matched to p0 in first spectrum
        match_glmix only applies to pseudo-voigt peaks. Use match_sigma=True and match_gamma=True to contrain
        glmix for voigt peaks.
        '''
        
        # TODO if number of peaks is different for each spectrum, allow user to specify which peaks get aligned
        
        if keys_list == None:
            keys_list = self.keys_list
        
        self.params = Parameters()
        
        # each entry in be_guess_dict contains a list of guess positions for each region
        # preferred input
        if isinstance(be_guess, dict):
            self.be_guess = be_guess
            # ! no need to subset because of how dicts work
            # subset
            # self.be_guess = {key: be_guess[key] for key in keys_list}
            # TODO add check to ensure dict keys match
        # convert a single-valued input into expected dict format
        elif self.is_float_or_int(be_guess):
                self.be_guess = {key: [be_guess for _ in range(self.n_peaks[key])] for key in keys_list}
        elif self.is_list_or_tuple(be_guess):
            # convert a list of single-valued inputs into expected dict format
            if self.is_float_or_int(be_guess[0]):
                self.be_guess = {key: [be_guess[k] for k in range(self.n_peaks[key])] for key in keys_list}
            # convert a list of a list of guesses into dict format
            elif self.is_list_or_tuple(be_guess[0]):
                self.be_guess = dict(zip(keys_list, be_guess))
                
        if self.is_list_or_tuple(peak_spacings) and (not self.is_list_or_tuple(peak_spacings[0])):
            peak_spacings = [peak_spacings]
            
        if self.is_list_or_tuple(peak_ratios) and (not self.is_list_or_tuple(peak_ratios[0])):
            peak_ratios = [peak_ratios]
                
        for key in keys_list:
            x_key = np.array(self.df_dict[key]['be'])
            y_key = np.array(self.df_dict[key]['cps'])
                
            # set number of peaks for each dataframe
            n_peaks_key = self.n_peaks[key]
                
            for peak_number in range(n_peaks_key):
                peak_id = "data_{}_p{}_".format(key, peak_number)
                # basis parameters
                self.params.add(peak_id+"amplitude", value=(max(y_key)-min(y_key))*1.5, min=0)
                self.params.add(peak_id+"center", value=self.be_guess[key][peak_number], min=min(x_key), max=max(x_key))
                # for pseudo-Voigt, sigma is the width of both the Gaussian and the Lorentzian component
                # for Voigt, sigma is the width of the Gaussian component and gamma is of the Lorentzian component
                self.params.add(peak_id+"sigma", value=sigma_guess, min=sigma_min, max=sigma_max, vary=sigma_vary)
                # pseudo-Voigt-specific parameters
                if lineshape == "pseudo_voigt":
                    self.params.add(peak_id+"glmix", value=glmix_guess, min=glmix_min, max=glmix_max, vary=glmix_vary)
                    self.params.add(peak_id+"height", 
                            expr=peak_id+"amplitude/"+peak_id+"sigma*((1-"+peak_id+"glmix)/sqrt(pi/log(2))+1/(pi*"+peak_id+"sigma))")
                    self.params.add(peak_id+"fwhm", expr="2*"+peak_id+"sigma")
                # Voigt-specific parameters
                elif lineshape == "voigt":
                    self.params.add(peak_id+"gamma", value=gamma_guess, min=gamma_min, max=gamma_max)
                    self.params.add(peak_id+"gfwhm", expr="2*"+peak_id+"sigma")
                    self.params.add(peak_id+"lfwhm", expr="2*"+peak_id+"gamma")
                    self.params.add(peak_id+"glmix", expr=peak_id+"lfwhm/("+peak_id+"lfwhm+"+peak_id+"gfwhm)",
                            min=glmix_min, max=glmix_max)
                    self.params._asteval.symtable['calculate_voigt_fwhm'] = self.calculate_voigt_fwhm
                    self.params.add(peak_id+"fwhm", expr="calculate_voigt_fwhm("+peak_id+"sigma,"+peak_id+"gamma)", 
                            min=fwhm_min, max=fwhm_max)
                
                for i in range(peak_number):
                    self.params.add("data_"+key+"_p"+str(peak_number)+"_p"+str(i)+"_ratio", expr=peak_id+"amplitude/data_"+key+"_p"+str(i)+"_amplitude", min=0, max=np.inf)
                    # setting min=0 ensures subsequent peaks increase in binding energy
                    self.params.add("data_"+key+"_p"+str(peak_number)+"_p"+str(i)+"_spacing", expr=peak_id+"center-data_"+key+"_p"+str(i)+"_center", min=0)
                
                if peak_number > 0:
                    if match_glmix == "within" and (lineshape == "pseudo_voigt"):
                        self.params.add(peak_id+"glmix", expr="data_"+key+"_p0_glmix")
                    if match_sigma == "within":
                        self.params.add(peak_id+"sigma", expr="data_"+key+"_p0_sigma")
                    if match_gamma == "within":
                        self.params.add(peak_id+"gamma", expr="data_"+key+"_p0_gamma")
                
                if ((peak_number > 0) or (key != keys_list[0])):
                    if lineshape == "pseudo_voigt": 
                        if ((match_glmix == "all") or (match_glmix == True)):
                            self.params.add(peak_id+"glmix", expr="data_"+keys_list[0]+"_p0_glmix")
                        if (match_fwhm == "all" or match_fwhm == True):
                            self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p0_sigma")
                            
                    if (match_sigma == "all" or match_sigma == True):
                        self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p0_sigma")
                    if (match_gamma == "all" or match_gamma == True):
                        self.params.add(peak_id+"gamma", expr="data_"+keys_list[0]+"_p0_gamma")
                        
                if key != keys_list[0]:
                    # constrain peak positions across different spectra?
                    # constrain all
                    if align_peaks == "all" or (align_peaks == True):
                        self.params.add(peak_id+"center", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_center")
                    # constrain only peaks specified in list
                    elif isinstance(align_peaks, list) or isinstance(align_peaks, tuple):
                        if peak_number == align_peaks[peak_number]:
                            self.params.add(peak_id+"center", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_center")
                    if lineshape == "pseudo_voigt":
                        if match_fwhm == "align":
                            self.params.add(peak_id+"sigma", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_sigma")
                        if match_glmix == "align":
                            self.params.add(peak_id+"glmix", expr="data_"+keys_list[0]+"_p"+str(peak_number)+"_glmix")
                        
            # syntax: (i,j,k) where i = ID of peak 1, j = ID of peak 2, and k = value where i > j
            if peak_ratios != False and (isinstance(peak_ratios, list) or isinstance(peak_ratios, tuple)):
                dataKey = "data_"+key+"_"
                if isinstance(peak_ratios[0], int):
                    peakRatios = [peak_ratios]
                    
                for i in range(len(peak_ratios)):
                    peakRatios = peak_ratios[i]
                        
                    peak1 = "p"+str(max(peakRatios[0:2]))
                    peak2 = "p"+str(min(peakRatios[0:2]))
                    
                    if isinstance(self.n_peaks,dict):
                        if max(peakRatios[0:2]) > (self.n_peaks[key]):
                            break
                    # peak1 = "p"+str(peakRatios[0])
                    # peak2 = "p"+str(peakRatios[1])
                    
                    # print(peak1)
                    # print(peak2)
                    
                    # if peakRatios[2] == "align":
                    #     if k == 0:
                    #         params.add(dataKey+peak1+"_"+peak2+"_ratio")
                    #     else:
                    #         params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+keys_list[0]+"_"+peak1+"_"+peak2+"_ratio")
                    # else:
                    if peakRatios[2] == "align":
                        if key == keys_list[0]:
                            self.params.add(dataKey+peak1+"_"+peak2+"_ratio")
                        else:
                            self.params.add(dataKey+peak1+"_"+peak2+"_ratio", expr="data_"+keys_list[0]+"_"+peak1+"_"+peak2+"_ratio")
                    else:
                        self.params.add(dataKey+peak1+"_"+peak2+"_ratio", value=peakRatios[2])
                        self.params[dataKey+peak1+"_"+peak2+"_ratio"].vary = False
                    self.params.add(dataKey+peak1+"_amplitude", expr=dataKey+peak2+"_amplitude*"+dataKey+peak1+"_"+peak2+"_ratio")

            if peak_spacings != False and (isinstance(peak_spacings, list) or isinstance(peak_spacings, tuple)):
                dataKey = "data_"+key+"_"
                # for i in range(len(constrainPeakSpacings)):
                #     peakSpacings = constrainPeakSpacings[i]
                for peakSpacings in peak_spacings:
                    # print(peakSpacings)
                        
                    peak1 = "p"+str(max(peakSpacings[0:2]))
                    peak2 = "p"+str(min(peakSpacings[0:2]))
                    
                    if isinstance(self.n_peaks,dict):
                        if max(peakSpacings[0:2]) > self.n_peaks[key]:
                            print(key)
                            print(peak1,peak2)
                            break
                    
                    self.params.add(dataKey+peak1+"_"+peak2+"_spacing", value=peakSpacings[2])
                    if len(peakSpacings) > 3:
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].min = peakSpacings[3]
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].max = peakSpacings[4]
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].vary = True
                    elif len(peakSpacings) == 3:
                        self.params[dataKey+peak1+"_"+peak2+"_spacing"].vary = False
                    self.params.add(dataKey+peak1+"_center", expr=dataKey+peak2+"_center+"+dataKey+peak1+"_"+peak2+"_spacing")
                    # print(params[dataKey+peak1+"_center"])
                    
        # unpack user-specified expression-based constraints
        if isinstance(expr_constraints, dict):
            for key in list(expr_constraints.keys()):
                if isinstance(expr_constraints[key], str):
                    self.params.add(key, expr=expr_constraints[key])
                elif isinstance(expr_constraints[key], dict):
                    self.params.add(key, **expr_constraints[key])
            
        # print(params)
        return self.params
    
    def set_n_peaks(self, n_peaks):
        '''
        Ensures n_peaks is a dict.
        '''
        if isinstance(n_peaks, dict):
            self.n_peaks = n_peaks
            # TODO add check to ensure dict keys match
        # generate dict from list/tuple
        elif self.is_list_or_tuple(n_peaks):
            self.n_peaks = dict(zip(self.keys_list, n_peaks))
        # generate dict from float/int
        elif self.is_float_or_int(n_peaks):
            self.n_peaks = dict(zip(self.keys_list, [n_peaks for _ in range(len(self.keys_list))]))
    
    def generate_model_single_spectrum_no_bg(self, params, key, x, model=0, lineshape="voigt"):
        model = 0
        for k in range(self.n_peaks[key]):
            peak_id = "data_{}_p{}_".format(key, k)
            
            if isinstance(lineshape, str):
                lineshape_k = lineshape
            elif self.is_list_or_tuple(lineshape):
                lineshape_k = lineshape[k]
                
            if lineshape_k == "gaussian":
                model += self.gaussian(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
            elif lineshape_k == "lorentzian":
                model += self.lorentzian(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
            elif lineshape_k == "pseudo_voigt":
                model += self.pseudo_voigt(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"glmix"])
            elif lineshape_k == "voigt":
                model += self.voigt(x, params[peak_id+"amplitude"],
                                    params[peak_id+"center"],params[peak_id+"sigma"],params[peak_id+"gamma"])
        return model

    # TODO make lineshape spec a class variable
    # TODO make component plotting work if multiple lineshapes are specified
    # TODO add stacking functionality for multiple spectra
    # TODO make display_residuals flag functional
    # TODO implement normalization after bg subtraction (not necessary?)
    # TODO implement plotting of initial fit
    def plot_result(self, 
                    subtract_bg=True, normalize=True,
                    display_bg=False, display_envelope=True, display_components=True, display_residuals=True,
                    text=None,
                    tight_layout=True, residual_lim=[-3,3],
                    colors=colors, component_z_spec=False, xdim=3.25*4/3, ydim=3.25, energy_axis='be',
                    save_fig=False, ypad=0, ylabel=None, ax_kwargs=None, **kwargs):
        
        j = 0
        for key in self.keys_list:
            # use gridspec to combine main and residual plots
            fig = plt.figure(layout='constrained')
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[5,1])
            ax, residual_ax = gs.subplots(sharex=True,sharey=False)
            df_key = self.df_dict[key]            
            
            if subtract_bg:
                bg_suffix = '_no_bg'
            else:
                bg_suffix = ''
            
            sns.lineplot(data=df_key, x=energy_axis, y='cps'+bg_suffix, 
                            ax=ax, mec=self.data_color, marker=self.marker, ls='None',
                            ms=self.marker_size, mew=self.marker_edge_width, zorder=999)
            
            if display_bg and (not subtract_bg):
                sns.lineplot(data=df_key, x=energy_axis, y='bg', 
                             ax=ax, color=self.background_color, linewidth=self.background_linewidth)
                
            if display_envelope:
                sns.lineplot(data=df_key, x=energy_axis, y='fit'+bg_suffix, 
                             ax=ax, color=self.envelope_color, linewidth=self.envelope_linewidth, zorder=998)
            
            if display_components:
                i = 0
                for peak_number in range(self.n_peaks[key]):
                    sns.lineplot(data=df_key, x=energy_axis, y='p{}'.format(peak_number)+bg_suffix, 
                                 ax=ax, color=colors[i], linewidth=self.component_linewidth)
                    i += 1

            if not isinstance(ax_kwargs, dict):
                ax_kwargs = {}
            self.ax_opts(ax, **ax_kwargs)
            # ! overwrites manual specification of ylim
            if ypad != False or (ypad != 0):
                ymin = min(df_key['cps'+bg_suffix])
                ymax = max(df_key['cps'+bg_suffix])
                ax.set_ylim([ymin-(ymax-ymin)*0.05,ymax*(1+ypad)])
            
            if ylabel == None or (not isinstance(ylabel, str)):
                ylabel = 'Intensity'
            
            ax.set_ylabel(ylabel, fontsize=self.label_font_size)
            ax.tick_params(axis='both', which='major', labelsize=self.tick_font_size)
            if text != None:
                ax.text(0.85,0.85,text[j],fontsize=self.label_font_size,transform = ax.transAxes, horizontalalignment='center', verticalalignment='center')
            # if tight_layout:
            #     plt.tight_layout()
            j += 1
            
            residual_ax.axhline(y=0, color=self.envelope_color, linestyle='--', linewidth=self.axes_linewidth, alpha=0.5)
            sns.lineplot(data=df_key, x=energy_axis, y='std_residuals', 
                            ax=residual_ax, mec=self.data_color, marker=self.residual_marker, ls='None',
                            ms=self.marker_size, mew=self.marker_edge_width)
            residual_ax.set_ylabel('R', style='italic', fontsize=self.label_font_size)
            self.ax_opts(residual_ax, ylim=residual_lim, **ax_kwargs)
            if energy_axis == 'be':
                residual_ax.set_xlabel("Binding Energy (eV)", fontsize=self.label_font_size)
                residual_ax.invert_xaxis() # only need to invert one axis
            elif energy_axis == 'ke':
                residual_ax.set_xlabel("Kinetic Energy (eV)", fontsize=self.label_font_size)
            residual_ax.tick_params(axis='both', which='major', labelsize=self.tick_font_size)
            
            fig.set_size_inches(xdim,ydim)
            # if tight_layout:
            #     plt.tight_layout()

            if isinstance(save_fig,dict):
                fig.savefig(save_fig[key]+".svg")
            if isinstance(save_fig,str):
                fig.savefig(save_fig+".svg")

    @staticmethod
    def ax_opts(ax, xlim=None, ylim=None, xticks=False, yticks=False, tick_direction='out',
                major_tick_multiple=0, minor_tick_multiple=0):
        if xlim != None:
            ax.set_xlim([min(xlim),max(xlim)])
        if ylim != None:
            ax.set_ylim(ylim)
            
        if xticks != False:
            ax.set_xticks(xticks)
        if yticks == False:
            ax.set_yticks([])
        elif yticks == True:
            ax.set_yticks()
        else:
            ax.set_yticks(yticks)
            
        ax.tick_params(direction=tick_direction)
        # specifying major_tick_multiple overrides manual xticks spec
        if major_tick_multiple > 0:
            ax.xaxis.set_major_locator(MultipleLocator(major_tick_multiple))
        if minor_tick_multiple > 0:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_multiple))
                
    @staticmethod
    def figOpts(fig, fontsize, xdim, ydim):
        fig.set_size_inches(xdim,ydim)

    @classmethod
    def set_class_plot_config(cls, profile=None,
                            # font options
                            font_family='Arial', label_font_size=12, tick_font_size=12, usetex=False,
                            # line styling options
                            envelope_linewidth=1.7, component_linewidth=1.6, axes_linewidth=1.5, background_linewidth=1.6,
                            # marker styling options
                            marker='+', residual_marker='+', marker_size=5, marker_alpha=2/3, marker_edge_width=2/3):
        # TODO add presets that can be overwritten if any of the individual parameters are user-specified
        if profile == 'print':
            pass
        elif profile == 'slide':
            pass
        
        cls.font_family = font_family
        cls.label_font_size = label_font_size
        cls.tick_font_size = tick_font_size
        cls.usetex = usetex
        
        cls.envelope_linewidth = envelope_linewidth
        cls.component_linewidth = component_linewidth
        cls.background_linewidth = background_linewidth
        cls.axes_linewidth = axes_linewidth
        
        cls.marker = marker
        cls.residual_marker = residual_marker
        cls.marker_size = marker_size
        cls.marker_alpha = marker_alpha
        cls.marker_edge_width = marker_edge_width
        
        rc('font',**{'family':'sans-serif','sans-serif':[cls.font_family]})
        rc('text', usetex=cls.usetex)
        rcParams['axes.linewidth'] = cls.axes_linewidth

    # stolen from PyARPES
    ## TODO update to use scipy integration routine
    @staticmethod
    def calculate_shirley_background(
        xps: np.ndarray, eps=1e-7, max_iters=500, n_samples=(5,5)
    ) -> np.ndarray:
        """Core routine for calculating a Shirley background on np.ndarray data."""
        background = np.copy(xps)
        cumulative_xps = np.cumsum(xps, axis=0)
        total_xps = np.sum(xps, axis=0)

        rel_error = np.inf

        i_left = np.mean(xps[:n_samples[0]], axis=0)
        i_right = np.mean(xps[-n_samples[1]:], axis=0)

        iter_count = 0

        k = i_left - i_right
        for iter_count in range(max_iters):
            cumulative_background = np.cumsum(background, axis=0)
            total_background = np.sum(background, axis=0)

            new_bkg = np.copy(background)

            for i in range(len(new_bkg)):
                new_bkg[i] = i_right + k * (
                    (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                    / (total_xps - total_background + 1e-5)
                )

            rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

            background = new_bkg

            if np.any(rel_error < eps):
                break

        if (iter_count + 1) == max_iters:
            warnings.warn(
                "Shirley background calculation did not converge "
                + "after {} steps with relative error {}!".format(max_iters, rel_error)
            )
            
        return background
    

# def stack_spectra(data_list, offset=0.02, energy_range=None, y='norm_intensity', legend=None):
# 	fig, ax = plt.subplots()
# 	i = 0
# 	for xas in data_list:
# 		df = xas.df
# 		df[y] = df[y] + offset*i
# 		sns.lineplot(data=df, x='energy', y=y, ax=ax)
# 		if is_tuple_or_list(legend):
# 			ax.text(min(df['energy']), offset*i, legend[i])
# 		i += 1
# 	if is_tuple_or_list(energy_range):
# 		ax.set_xlim(energy_range)
# 	#if is_tuple_or_list(legend):
# 		#ax.legend(legend)

def is_tuple_or_list(x):
    return isinstance(x,tuple) or isinstance(x,list)

class Xas:

    def __init__(self, df, skiprows=None, flip=True):
        '''
        generate Xas object from pre-processed dataframe
        skiprows: number of leading rows to skip
        flip: multiply spectrum by -1 if True
        '''
        self.df = df
        if skiprows != None:
            self.df.drop(skiprows, inplace=True)
        # print(df)
        if 'norm_intensity' not in self.df.columns:
            if flip:
                flip = -1
            else:
                flip = 1
            y = self.df['intensity']*flip
            if flip:
                y = y - max(y)
            else:
                y = y - min(y)
            # normalize by area and shift so that min value is zero
            self.df['norm_intensity'] = y/abs(trapezoid(y, x=self.df['energy']))
            self.df['norm_intensity'] = self.df['norm_intensity'] - min(self.df['norm_intensity'])
        
    @classmethod
    def from_txt(cls, path, drop_zeros=True, energy_range=None, **kwargs):
        '''
        initialize Xas object from .fits-derived .txt
        drop_zeroes: drop columns in which energy=0
        energy_range: trim rows to lower and upper limit, exclusive. format: [lower, upper]
        '''
        df = pd.read_table(path, names=['time','energy','counts','i0'])
        if drop_zeros:
            idx = df[df['energy'] == 0].index
            df.drop(idx, inplace=True)
        if is_tuple_or_list(energy_range):
            idx = df[(df['energy'] < min(energy_range)) | (df['energy'] > max(energy_range))].index
            df.drop(idx, inplace=True)
        df['intensity'] = df['counts']/df['i0']
        return cls(df, **kwargs)
    
    def plot(self, energy_range=None, y='norm_intensity'):
        fig, ax = plt.subplots()
        sns.lineplot(data=self.df, x='energy', y=y, ax=ax)

    @staticmethod
    def generate_data_list(path_list, **kwargs):
        return [Xas.from_txt(path, **kwargs) for path in path_list]

    @staticmethod
    def stack_spectra(data_list, offset=0.02, energy_range=None, y='norm_intensity', 
        legend=None, ax_spec=None):
        if ax_spec:
            ax = ax_spec
        else:
            fig, ax = plt.subplots()
        i = 0
        for xas in data_list:
            df = xas.df
            df[y] = df[y] + offset*i
            series = sns.lineplot(data=df, x='energy', y=y, ax=ax)
            if is_tuple_or_list(legend):
                ax.text(min(df['energy']), offset*i, legend[i])
            i += 1
        if is_tuple_or_list(energy_range):
            ax.set_xlim(energy_range)
        #if is_tuple_or_list(legend):
            #ax.legend(legend)

    @staticmethod
    def average_spectra(data_list, energy_range=None, ax_spec=None):
        norm_intensity_arr = []
        for xas in data_list:
            df = xas.df
            if is_tuple_or_list(energy_range):
                energy_range_idx = df[(df['energy'] < min(energy_range)) | (df['energy'] > max(energy_range))].index
                df.drop(energy_range_idx, inplace=True)
            norm_intensity_arr.append(df['norm_intensity'])
        norm_intensity_arr = np.array(norm_intensity_arr)
        norm_intensity_avg = np.average(norm_intensity_arr, axis=0)
        if ax_spec:
            ax = ax_spec
        else:
            fig, ax = plt.subplots()
        series = ax.plot(df['energy'], norm_intensity_avg)
        # TODO: fix this atrocity
        df = pd.DataFrame(np.transpose(np.array([np.array(df['energy']), norm_intensity_avg])), columns=['energy', 'intensity'])
        return Xas(df, flip=False)
        
    @staticmethod
    def plot_all(dir, prefix, id_list, xlim=None, flip=True, **kwargs):
        path_list = []
        for k in id_list:
            if k < 10:
                id = '0{}'.format(k)
            else:
                id = str(k)
            path_list.append(dir+prefix+id+'.txt')
        # import data
        data_list = Xas.generate_data_list(path_list, flip=flip, **kwargs)
        # plot
        fig, ax = plt.subplots(2,1,sharex=True,sharey=False)
        Xas.stack_spectra(data_list, ax_spec=ax[0])
        xas = Xas.average_spectra(data_list, ax_spec=ax[1])
        if is_tuple_or_list(xlim):
            for axi in ax:
                axi.set_xlim(left=min(xlim),right=max(xlim))
        fig.set_size_inches(3.25,3.25*1.618)
        return xas
    
    
class Casa:
    
    fontsize=18
    labelsize=fontsize
    legendsize=fontsize
    linewidth=2.5

    font_family='Arial'
    axes_linewidth=2.25
    tick_linewidth=axes_linewidth*.9
    tick_length=tick_linewidth*5
    marker_size=7
    marker_edge_width=linewidth/2
    labelpad=5

    rc('font',**{'family':'sans-serif','sans-serif':[font_family]})
    rc('text', usetex=False)
    rcParams['axes.linewidth'] = axes_linewidth

    @staticmethod
    def load_csv(path, norm='area', skiprows=7, norm_target='Envelope'):
        di = pd.read_csv(path, skiprows=skiprows)
        di = di.dropna(axis=1)
        comps = []
        norm_suffix = ''
        n_cols = len(di.columns.values)
        if n_cols > 4:
            n_comps = int((len(di.columns.values) - 8)/2)
        else:
            n_comps = 0
        if n_comps > 0:
            for i in range(n_comps):
                comps += ['p{} CPS'.format(i)]
                di.columns.values[i+2] = 'p{}'.format(i)
                di.columns.values[i+6+n_comps] = 'p{} CPS'.format(i)
        comps += ['CPS']
        
        bg_flag = 'Background CPS' in di.columns.values
        comps_flag = n_comps > 0
        if bg_flag:
            comps += ['Background CPS']
            norm_suffix = '_no_bg'
        if comps_flag:
            comps += ['Envelope CPS']
            
        if bg_flag:
            for c in comps:
                di[c+'_no_bg'] = di[c] - di['Background CPS']
            min_cps = min(di['Background CPS'+norm_suffix])
        else:
            min_cps = min(di['CPS'+norm_suffix])
        if norm == 'area':
            denom = -trapezoid(di['CPS'+norm_suffix], x=di['B.E.'])
        elif norm == 'minmax':
            denom = max(di['CPS'+norm_suffix])-min_cps
        elif norm[0] == 'p':
            denom = -trapezoid(di[norm+' CPS'+norm_suffix], x=di['B.E.'])
        for c in comps:
            di[c+'_norm'] = (di[c] - min_cps)/denom
            if norm_suffix == '_no_bg':
                di[c+norm_suffix+'_norm'] = (di[c+norm_suffix] - min_cps)/denom
                
        if n_comps > 0:
            di['residual'] = di['CPS'] - di['Envelope CPS']
            di['residual_norm'] = di['CPS_norm'] - di['Envelope CPS_norm']
        return di

    @classmethod
    def plot_single(cls, di, color, xlim, ylim, savefig=False, dim=[4,3], ylabel='Intensity (a.u.)', subtract_bg=False):
            fig, ax = plt.subplots(layout='constrained')

            if subtract_bg:
                bg_suffix = '_no_bg'
            else:
                bg_suffix = ''
            
            shift = 0
            ax.plot(di['B.E.'], di['CPS{}_norm'.format(bg_suffix)]+shift, linewidth=cls.linewidth, color=color, zorder=999)
            ax.hlines(shift, 0, 999, color='gray', linewidth=cls.linewidth*2/3, zorder=100+1)

            Pes.ax_opts(ax, major_tick_multiple=5, minor_tick_multiple=1, xlim=xlim, ylim=ylim)
            ax.set_ylabel(ylabel, fontsize=cls.fontsize, labelpad=cls.labelpad*2/3)
            ax.set_xlabel('Binding Energy (eV)', fontsize=cls.fontsize, labelpad=cls.labelpad)
            ax.invert_xaxis()

            ax.tick_params(labelsize=cls.fontsize)
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length, which='major')
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length*0.5, which='minor')

            fig.set_size_inches(*dim)
            if savefig:
                    fig.savefig(savefig)
                    
    @classmethod
    def plot_stack(cls, data, color, xlim, ylim, shift=0, savefig=False, hline=True, dim=[4,3], xlabel='Binding Energy (eV)', ylabel='Intensity (a.u.)', subtract_bg=False, 
                   plot_comps=False, comp_color=None, plot_envelope=False, plot_residuals=False, residual_offset=-0.15, major_tick_multiple=5, minor_tick_multiple=1):
            fig, ax = plt.subplots(layout='constrained')

            if subtract_bg:
                bg_suffix = '_no_bg'
            else:
                bg_suffix = ''

            for i in range(len(data)):
                di = data[i]
                if not plot_envelope:
                    ax.plot(di['B.E.'], di['CPS{}_norm'.format(bg_suffix)]+shift*i, linewidth=cls.linewidth, color=color[i], zorder=999)
                if hline:
                    ax.hlines(shift*i, 0, 999, color='gray', linewidth=cls.linewidth*0.75, zorder=1+i+1)
                if plot_envelope:
                    ax.plot(di['B.E.'], di['Envelope CPS{}_norm'.format(bg_suffix)]+shift*i, linewidth=cls.linewidth, color=color[i], zorder=1000)
                    ax.plot(di['B.E.'], di['CPS{}_norm'.format(bg_suffix)]+shift*i, 'k+', linewidth=cls.linewidth, zorder=200, color='#444444',
                            ms=cls.marker_size, mew=cls.marker_edge_width)
                if plot_comps:
                    n_comps = int(len([id for id in di.columns.values if 'p' in id])/5)
                    for j in range(n_comps-1):
                        if comp_color is None:
                            comp_color_j = comp_color
                        else:
                            comp_color_j = comp_color[j]
                        ax.plot(di['B.E.'], di['p{} CPS{}_norm'.format(j, bg_suffix)]+shift*i, linewidth=cls.linewidth*0.75, color=comp_color_j, zorder=100+i+1+j)
                if plot_residuals:
                    ax.plot(di['B.E.'], di['residual_norm']+shift*i+residual_offset, linewidth=cls.linewidth*0.75, color='gray', zorder=100)

            Pes.ax_opts(ax, major_tick_multiple=major_tick_multiple, minor_tick_multiple=minor_tick_multiple, xlim=xlim, ylim=ylim)
            ax.set_ylabel(ylabel, fontsize=cls.fontsize, labelpad=cls.labelpad*2/3)
            ax.set_xlabel(xlabel, fontsize=cls.fontsize, labelpad=cls.labelpad)
            ax.invert_xaxis()

            ax.tick_params(labelsize=cls.fontsize)
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length, which='major')
            ax.xaxis.set_tick_params(width=cls.tick_linewidth, length=cls.tick_length*0.5, which='minor')

            fig.set_size_inches(*dim)
            if savefig:
                    fig.savefig(savefig)
            return fig, ax