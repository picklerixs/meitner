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


class Xps:
    
    def __init__(self, 
                 ds, 
                 be_range=None, 
                 method='area', 
                 **kwargs):
        self.ds = ds
        # trim to be_range
        if Aux.is_list_or_tuple(be_range):
            self.ds = self.ds.sel(be=slice(*be_range))
        if 'bg' not in list(self.ds.data_vars):
            self.fit_background(**kwargs)
        if 'cps_no_bg_norm' not in list(ds.data_vars):
            Processing.normalize(self.ds, method=method)
        
    @classmethod
    def from_vamas(cls, **kwargs):
        return cls(Vms.import_single_vamas(**kwargs))
    
    def fit_background(self,
                       background='shirley',
                       be_range=None,
                       **kwargs):
        if (background == 'shirley') or (background == 's'):
            self.ds['bg'] = ('be', Bg.shirley(self.ds['cps'], **kwargs))
        self.ds['cps_no_bg'] = ('be', (self.ds['cps'] - self.ds['bg']).data)

class Aux:
    '''
    Auxiliary methods.
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
     
class Processing:
    @staticmethod
    def normalize(ds, method='area', norm_range=None, y='cps_no_bg'):
        ymin = ds.min(dim='be')[y]
        ymax = ds.max(dim='be')[y]
        
        if method == 'minmax':
            norm_constant = ymax - ymin
        if method == 'area':
            norm_constant = abs(trapezoid(ds[y] - ymin), x=ds['be'])
            
        for y in ['bg','cps','cps_no_bg']:
            ds['{}_norm'.format(y)] = (ds[y] - ymin)/norm_constant

class Bg:
    '''
    Methods to fit backgrounds to XPS (and other) data.
    '''
    @staticmethod
    def shirley(
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
                            read_phi=False):
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
            
            print('Found ' + str(len(ids)) + ' blocks.')
            print()
            
            # check for and log duplicate block IDs
            duplicate_flag = n != len(set(ids))
            if duplicate_flag:
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
                    
def constrain_doublet_gamma(peak_ids, dict_keys, expr_constraints, spacing_min=None, spacing_max=None):
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