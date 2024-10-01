import pandas as pd
import lmfit
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pathlib
import warnings

from scipy.integrate import trapezoid
from scipy.stats import linregress
from scipy.signal import find_peaks, savgol_filter
from functools import reduce
from matplotlib.ticker import MultipleLocator

class Rixs:        
    def __init__(
        self,
        expt_path,
        calibration_data=None,
        **kwargs
    ):
        '''
        Methods for processing and visualizing RIXS data. Generates an xr.Dataset instance whose dimensions are excitation energy (excitation_energy)
        and CCD pixel (ccd_pixel) and that contains mRIXS and XAS data read from Andor output.
        
        ARGS:
            expt_path (str or pathlib.Path()): Directory containing Andor information (AI) file and Andor/ directory, which
                should contain 1D CCD data in .txt format.
            
        TODO:
            Clean up xr.Dataset variable names
            Allow loading and plotting of XES data
            Read RIXS traces based on Filename in ds
            Implement elastic peak fitting method(s)
            Add support for other endstations (e.g. SALSA)
            Implement interactive mRIXS plots (including interactive (i)PFY selection)
            Clean up plot styling options
            Add methods to combine multiple mRIXS figures together (gridspec?)
        '''
        self.expt_path = pathlib.Path(expt_path)

        # read and clean Andor info (AI) file
        self.ai_path = sorted(self.expt_path.glob('*AI.txt'))
        if len(self.ai_path) > 1:
            warnings.warn('More than one Andor info file found.')
        self.ai_path = self.ai_path[0]
        ai_df = pd.read_table(self.ai_path, skiprows=12)
        self.ds = xr.Dataset(ai_df)
        self.ds = self.ds.assign_coords({'excitation_energy': self.ds['BL 8 Energy']})
        self.ds = self.ds.swap_dims({'dim_0': 'excitation_energy'})
        self.ds = self.ds.drop_vars('dim_0')

        # read RIXS traces
        self.spec_path = sorted(self.expt_path.glob('Andor'))
        if len(self.spec_path) > 1:
            warnings.warn('More than one Andor directory found.')
        self.spec_path = self.spec_path[0]

        self.data_list = []
        self.child_list = sorted(self.spec_path.glob('*-1D.txt'))
        for c in self.child_list:
            self.data_list.append(pd.read_csv(c, skiprows=9, sep=r'\t', engine='python'))
        # concatenate RIXS traces by aligning on X (CCD pixels)
        if len(self.data_list) > 1:
            df = pd.concat([d.set_index('X') for d in self.data_list], axis=1, join='inner').reset_index()
        else:
            df = self.data_list[0]
        df.drop('X', axis=1, inplace=True)
        self.da = xr.DataArray(
            data=df,
            dims=['ccd_pixel', 'excitation_energy'],
            coords={
                'ccd_pixel': list(range(2048)),
                'excitation_energy': self.ds['excitation_energy']
            }
        )
        self.ds = self.ds.merge({'rixs_intensity': self.da})

        if calibration_data:
            self.calibration_data = np.array(calibration_data)
        else:
            self.calibration_data = None
        self.normalize_to_flux()
        print(self.ds)
            
    def calc_pfy(
        self,
        xlim,
        mode='ipfy',
        xmode='emission_energy'
    ):        
        ds = self.ds.where(self.ds[xmode] > min(xlim))
        ds = ds.where(ds[xmode] < max(xlim))
        try:
            if mode == 'ipfy':
                self.ds = self.ds.merge({
                    'norm_iPFY': np.divide(1, ds['norm_rixs_intensity'].sum(dim=xmode))
                })
            else:  
                self.ds = self.ds.merge({
                    'norm_PFY': ds['norm_rixs_intensity'].sum(dim=xmode)
                })
        except:
            warnings.warn('Normalized RIXS intensity not found. (i)PFY calculation aborted.')
        

    def find_elastic_line(
        self,
        distance=None,
        xlim=None,
        ylim=None,
        width=1.25,
        prominence=0.00005,
        **kwargs
    ):
        '''
        Auto-detect elastic line based on search windows in incident energy vs CCD pixel space.

        ARGS:
            distance: Minimum distance between peaks found by scipy.find_peaks(). Defaults to a large value to find the global maximum and nothing else.
            xlim: Emission energy window in which to search for the elastic peak (inclusive).
            ylim: Excitation energy window in which to search for the elastic peak (inclusive).

        Additional kwargs are passed to scipy.find_peaks().
        '''
        da = self.ds["norm_rixs_intensity"]
        ccd_pixel = self.ds["ccd_pixel"]
        idx_arr = []
        peak_arr = []
        ccd_pixel_arr = []
        if isinstance(ylim, list) or isinstance(ylim, tuple):
            self.excitation_energy_filtered = xr.Dataset(coords={'excitation_energy': np.nan})
            for ylim_i in ylim:
                ds = self.ds.where(self.ds['excitation_energy'] > min(ylim_i))
                ds = ds.where(ds['excitation_energy'] < max(ylim_i))
                self.excitation_energy_filtered = xr.concat(
                    [ds, self.excitation_energy_filtered],
                    dim='excitation_energy'
                )
            self.excitation_energy_filtered = self.excitation_energy_filtered.dropna(dim='excitation_energy', how='any')
            self.excitation_energy_filtered = self.excitation_energy_filtered['excitation_energy']
        else:
            self.excitation_energy_filtered = self.ds["excitation_energy"]
        if isinstance(xlim, list) or isinstance(xlim, tuple):
            for xlim_i in xlim:
                da = da.where(da["ccd_pixel"] > min(xlim_i))
                da = da.where(da["ccd_pixel"] < max(xlim_i))
        for e in self.excitation_energy_filtered:
            try:
                rixs_cut = da.sel(excitation_energy=e)
            except:
                warnings.warn('No RIXS cut found at excitation energy = {} eV.'.format(e))
            else:
                idx, _ = find_peaks(
                    rixs_cut, 
                    distance=distance, 
                    width=width,
                    prominence=prominence,
                    **kwargs
                )
                # try:
                #     i = idx[1]
                # except:
                #     i = idx[0]
                try:
                    i = idx[0]
                    # idx_arr.append(i)
                    # peak_arr.append(rixs_cut[i])
                    # ccd_pixel_arr.append(ccd_pixel[i])
                except:
                    pass
            idx_arr.append(i)
            peak_arr.append(rixs_cut[i])
            ccd_pixel_arr.append(ccd_pixel[i])
        idx_arr = np.array(idx_arr)
        peak_arr = np.array(idx_arr)
        self.ccd_pixel_arr = np.array(ccd_pixel_arr)
        return self.ccd_pixel_arr, self.excitation_energy_filtered

    def fit_elastic_line(
        self,
        overwrite=True,
        custom_params=None,
    ):
        '''
        Fit CCD pixel vs excitation energy data

        ARGS:
            ccd_pixel_arr (np.array): Array of CCD pixel values.
            excitation_energy_arr (np.array): Array of excitation energies.
        '''
        if not custom_params:
            custom_params = []
        self.params = lmfit.Parameters()
        self.params.add('intercept', value=-np.average(self.excitation_energy_filtered))
        self.params.add('slope', value=(np.max(self.excitation_energy_filtered)-np.min(self.excitation_energy_filtered))/np.average(self.ccd_pixel_arr), min=0)
        self.params.add_many(*custom_params)
        print(self.params)
        
        print(self.ccd_pixel_arr)
        print(self.excitation_energy_filtered)
        minimizer = lmfit.Minimizer(lambda params, x, y: params['intercept'] + params['slope']*x - y, self.params, fcn_args=(self.ccd_pixel_arr, self.excitation_energy_filtered))
        self.result = minimizer.minimize()
        self.intercept = self.result.params['intercept'].value
        self.slope = self.result.params['slope'].value
        self.fit_energy = self.intercept + self.slope*self.ds['ccd_pixel']

        try:
            self.ds["emission_energy"]
        except:
            self.ds["emission_energy"] = self.fit_energy
        else:
            warnings.warn('Dataset already has emission energy data.')
            if overwrite:
                self.ds["emission_energy"] = self.fit_energy
                warnings.warn('Existing emission energy data overwritten.')
        return self.fit_energy
    
    def normalize_to_flux(
        self,
        i_0='Izero',
        target_vars=['rixs_intensity', 'TFY', 'TEY']
    ):
        '''
        Normalize target variables to x-ray flux.
        
        ARGS:
            i_0 (str or list-like of float): Variable name corresponding to x-ray flux in self.ds or an array of flux values.
        '''
        if isinstance(i_0, str):
            i_0 = self.ds[i_0]
        for var in target_vars:
            try:
                self.ds = self.ds.merge({
                    'norm_{}'.format(var): self.ds[var]/i_0
                })
            except:
                warnings.warn('Target variable {} not found or invalid i_0 specified.'.format(var))
        
        
    def plot_xes(
        self,
        excitation_energy,
        color=None,
        offset=0,
        linewidth=1.8,
        xmode='emission_energy',
        scale_factor=None,
        sum_lim=False,
        sum_dy=0.2,
        # kwargs passed to self.plot_opts()
        dim=[3.25,3.25],
        savefig=False,
        text=None,
        xlabel=None,
        xmajtm=None,
        xmintm=None,
        ylabel='Intensity (a.u.)',
        ymajtm=None,
        ymintm=None,
        xlim=None,
        legend=False
        ):
        '''
        Plot one or more XES traces.

        ARGS:
            excitation_energy (list-like of float or int): Excitation energies to plot. 
                If an exact match does not exist, the closest existing value will be selected.
                
        KWARGS:
            xmode (str): 'emission_energy', 'energy_loss', or 'ccd_pixel'
        '''
        self.fig, self.axs = plt.subplots(
                layout='constrained'
            )
        dy = 0
        i = 0
        self.ei_list = []
        self.i_max_list = []
        self.ccd_max_list = []
        color_ei = color
        for ei in excitation_energy:
            if isinstance(color, list) or isinstance(color, tuple):
                color_ei = color[i]
            self.rixs_cut = self.ds.sel(excitation_energy=ei, method='nearest')   
            if xlim and (xmode == 'ccd_pixel'):
                self.rixs_cut = self.rixs_cut.sel(ccd_pixel=slice(*xlim)) 
            # print('Actual excitation energy: {} eV'.format(self.rixs_cut['excitation_energy'].values))  
            self.ei_list.append(self.rixs_cut['excitation_energy'].values)
            j = self.rixs_cut['norm_rixs_intensity'].argmax()
            self.i_max_list.append(self.rixs_cut['norm_rixs_intensity'].max())
            self.ccd_max_list.append(j)
            k = 0
            if sum_lim:
                y = 0
                for e in np.arange(ei - min(sum_lim), ei + sum_dy + max(sum_lim) + sum_dy, sum_dy):
                    self.rixs_cut = self.ds.sel(excitation_energy=e, method='nearest')
                    if xlim and (xmode == 'ccd_pixel'):
                        self.rixs_cut = self.rixs_cut.sel(ccd_pixel=slice(*xlim))
                    y += self.rixs_cut['norm_rixs_intensity']
                    k += 1
                y = y/k
            else:
                y = self.rixs_cut['norm_rixs_intensity']
            if isinstance(scale_factor, int) or isinstance(scale_factor, float):
                y = y*scale_factor
            elif isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                y = y*scale_factor[i]
            if xmode == 'energy_loss':
                self.rixs_cut['energy_loss'] = self.rixs_cut['emission_energy'] -  self.ei_list[i]
            self.axs.plot(
                self.rixs_cut[xmode],
                y+dy,
                linewidth=linewidth,
                color=color_ei
            )
            dy += offset 
            i += 1
        
        if isinstance(xlabel, str):
            pass
        elif xmode == 'emission_energy':
            xlabel = 'Emission Energy (eV)'
        elif xmode == 'ccd_pixel':
            xlabel = 'CCD Pixel (a.u.)'
        elif xmode == 'energy_loss':
            xlabel = 'Energy Loss (eV)'

        self.plot_opts(
            dim=dim,
            savefig=savefig,
            text=text,
            xlabel=xlabel,
            xmajtm=xmajtm,
            xmintm=xmintm,
            ylabel=ylabel,
            ymajtm=ymajtm,
            ymintm=ymintm,
            xlim=xlim,
            mode='xes'
        )
        
        if legend:
            self.axs.legend(self.ei_list)
            
        self.ei_list = np.array(self.ei_list)
        self.i_max_list = np.array(self.i_max_list)
        if xlim:
            self.ccd_max_list = np.array(self.ccd_max_list) + min(xlim)
        else:
            self.ccd_max_list = np.array(self.ccd_max_list)
            
        self.xy_list = np.column_stack([self.ccd_max_list, self.ei_list])

        return self.fig, self.axs, self.xy_list
    
    
    def plot_xas(
        self,
        plot_tfy=False,
        plot_tey=True,
        plot_ipfy=False,
        offset=0,
        # kwargs passed to self.plot_opts()
        dim=[3.25,3.25],
        savefig=False,
        text=None,
        xlabel='Excitation Energy (eV)',
        xmajtm=None,
        xmintm=None,
        ylabel='Intensity (a.u.)',
        ymajtm=None,
        ymintm=None,
        xlim=None,
        ):
        '''
        Plot one or more XAS traces.

        ARGS:
            excitation_energy (list-like of float or int): Excitation energies to plot. 
                If an exact match does not exist, the closest existing value will be selected.
        '''
        self.fig, self.axs = plt.subplots(layout='constrained')
        
        dy = 0
        if plot_tey:
            self.axs.plot(
                self.ds['excitation_energy'],
                (self.ds['norm_TEY']-self.ds['norm_TEY'].min())/(self.ds['norm_TEY'].max()-self.ds['norm_TEY'].min())+dy
            )
            dy += offset
        if plot_tfy:
            self.axs.plot(
                self.ds['excitation_energy'],
                (self.ds['norm_TFY']-self.ds['norm_TFY'].min())/(self.ds['norm_TFY'].max()-self.ds['norm_TFY'].min())+dy
            )
            dy += offset
        if plot_ipfy:
            self.axs.plot(
                self.ds['excitation_energy'],
                (self.ds['norm_iPFY']-self.ds['norm_iPFY'].min())/(self.ds['norm_iPFY'].max()-self.ds['norm_iPFY'].min())+dy
            )
            
        # for ax in self.axs:
        #     ax.set_ylim([
        #         self.ds['excitation_energy'].min(),
        #         self.ds['excitation_energy'].max()
        #     ])
        
        self.plot_opts(
            dim=dim,
            savefig=savefig,
            text=text,
            xlabel=xlabel,
            xmajtm=xmajtm,
            xmintm=xmintm,
            ylabel=ylabel,
            ymajtm=ymajtm,
            ymintm=ymintm,
            xlim=xlim,
            mode='xes'
        )

        return self.fig, self.axs
        
        
    def plot_mrixs(
        self,
        dim=[3.25,3.25],
        plot_elastic_line=False,
        plot_tfy=False,
        plot_tey=True,
        plot_ipfy=False,
        savefig=False,
        text=None,
        xlabel='Emission Energy (eV)',
        xmajtm=None,
        xmintm=None,
        xlim=None,
        ylabel='Excitation Energy (eV)',
        ymajtm=None,
        ymintm=None,
        ylim=None,
        xas_color='black',
        xas_tick_labels=False,
        xas_linewidth=1.7,
        xmode='emission_energy',
        width_ratios=(1, 0.25),
        # kwargs passed to plt.pcolormesh()
        alpha=1,
        antialiased=True,
        cmap='jet',
        edgecolor='face',
        linewidth=0,
        rasterized=True,
        shading='gouraud',
        **kwargs
    ):
        '''
        Plot mRIXS and XAS data.
        
        KWARGS:
            plot_tfy (bool): Whether to plot TFY data.
            plot_tey (bool): Whether to plot TEY data.
            xlim (list-like): x-limits of mRIXS plot.
            width_ratios (list-like): Relative widths of mRIXS and XAS plots.
            
        Additional kwargs are passed to plt.colormesh().
        '''
        self.fig, self.axs = plt.subplots(
            1,
            2,
            layout='constrained',
            gridspec_kw={'width_ratios': width_ratios}
        )
        
        if xmode == 'emission_energy':
            try:
                self.ds['emission_energy']
            except:
                x = self.ds['ccd_pixel']
            else:
                x = self.ds['emission_energy']
        elif xmode == 'ccd_pixel':
            x = self.ds['ccd_pixel']

        pc = self.axs[0].pcolormesh(
            x,
            self.ds['excitation_energy'], 
            self.ds['norm_rixs_intensity'].transpose(), 
            linewidth=linewidth, 
            antialiased=antialiased, 
            alpha=alpha, 
            edgecolor=edgecolor,
            rasterized=rasterized,
            vmin=self.ds['norm_rixs_intensity'].min(), 
            vmax=self.ds['norm_rixs_intensity'].max(), 
            cmap=cmap,
            shading=shading,
            **kwargs
        )

        if plot_elastic_line and (xmode == 'ccd_pixel'):
            self.axs[0].plot(
                self.ccd_pixel_arr,
                self.excitation_energy_filtered,
                'gx', color='orange'
            )
            self.axs[0].plot(
                self.ds['ccd_pixel'],
                self.ds['emission_energy'],
                'r-'
            )
        # XAS is automatically min-max normalized
        if plot_tfy:
            self.axs[1].plot(
                (self.ds['norm_TFY']-self.ds['norm_TFY'].min())/(self.ds['norm_TFY'].max()-self.ds['norm_TFY'].min()),
                self.ds['excitation_energy'],
                color=xas_color,
                linewidth=xas_linewidth
            )
        if plot_tey:
            self.axs[1].plot(
                (self.ds['norm_TEY']-self.ds['norm_TEY'].min())/(self.ds['norm_TEY'].max()-self.ds['norm_TEY'].min()),
                self.ds['excitation_energy'],
                color=xas_color,
                linewidth=xas_linewidth
            )
        if plot_ipfy:
            self.axs[1].plot(
                (self.ds['norm_iPFY']-self.ds['norm_iPFY'].min())/(self.ds['norm_iPFY'].max()-self.ds['norm_iPFY'].min()),
                self.ds['excitation_energy'],
                color=xas_color,
                linewidth=xas_linewidth
            )
        self.axs[1].set_xlim([-0.1, 1.2])
        self.axs[1].set_xticks([])
        if not xas_tick_labels:
            self.axs[1].set_yticklabels([])
        
            
        for ax in self.axs:
            ax.set_ylim([
                self.ds['excitation_energy'].min(),
                self.ds['excitation_energy'].max()
            ])
        
        self.plot_opts(
            dim=dim,
            savefig=savefig,
            text=text,
            xlabel=xlabel,
            xmajtm=xmajtm,
            xmintm=xmintm,
            ylabel=ylabel,
            ymajtm=ymajtm,
            ymintm=ymintm,
            xlim=xlim,
            ylim=ylim,
            mode='mrixs'
        )
        
        return self.fig, self.axs
        
        
    def plot_opts(
        self,
        dim=[3.25,3.25],
        savefig=False,
        text=None,
        xlabel='Emission Energy (eV)',
        xmajtm=None,
        xmintm=None,
        ylabel='Excitation Energy (eV)',
        ymajtm=None,
        ymintm=None,
        xlim=None,
        ylim=None,
        mode='mrixs'
    ):
        if mode == 'mrixs':
            if xlim:
                self.axs[0].set_xlim(xlim)
            if ylim:
                self.axs[0].set_ylim(ylim)
                self.axs[1].set_ylim(ylim)
            
            if xmajtm:
                self.axs[0].xaxis.set_major_locator(MultipleLocator(xmajtm))
            if xmintm:
                self.axs[0].xaxis.set_minor_locator(MultipleLocator(xmintm))
            if ymajtm:
                self.axs[0].yaxis.set_major_locator(MultipleLocator(ymajtm))
                self.axs[1].yaxis.set_major_locator(MultipleLocator(ymajtm))
            if ymintm:
                self.axs[0].yaxis.set_minor_locator(MultipleLocator(ymintm))
                self.axs[1].yaxis.set_minor_locator(MultipleLocator(ymintm))

            self.axs[0].set_xlabel(xlabel)
            self.axs[0].set_ylabel(ylabel)
            if isinstance(text, str):
                self.axs[0].text(0.05, 0.925, text, horizontalalignment='left',
                                verticalalignment='center', transform=self.axs[0].transAxes, color='white')
        elif mode == 'xes':
            if xlim:
                self.axs.set_xlim(xlim)
            if ylim:
                self.axs.set_ylim(ylim)
            
            if xmajtm:
                self.axs.xaxis.set_major_locator(MultipleLocator(xmajtm))
            if xmintm:
                self.axs.xaxis.set_minor_locator(MultipleLocator(xmintm))
            if ymajtm:
                self.axs.yaxis.set_major_locator(MultipleLocator(ymajtm))
            if ymintm:
                self.axs.yaxis.set_minor_locator(MultipleLocator(ymintm))

            self.axs.set_xlabel(xlabel)
            self.axs.set_ylabel(ylabel)
            if isinstance(text, str):
                self.axs.text(0.05, 0.925, text, horizontalalignment='left',
                                verticalalignment='center', transform=self.axs.transAxes, color='black')
        
        self.fig.set_size_inches(*dim)

        if savefig:
            self.fig.savefig(savefig)
        
        
    def plot_mrixs_legacy(
        self,
        show=False,
        savefig=None,
        plot_pfy=False,
        pfy_color='gray',
        plot_tfy=True,
        plot_tfy_masked=False,
        tfy_color='black',
        plot_tey=False,
        tey_color='blue',
        izero='Izero',
        drop=None,
        ipfy_lim=None,
        plot_ipfy=False,
        ipfy_color='Blue',
        ipfy_box=False,
        dim=[3.25,3.25],
        xlim=[0,2047],
        header=None,
        tfy_skip=None,
        x_minor_tick_multiple=1,
        y_major_tick_multiple=4,
        y_minor_tick_multiple=1,
        vmin=0,
        vmax=1,
        fig=None,
        ax=None,
        cmap='viridis',
        cbar=True
    ):
        fontsize=12
        font_family='Arial'
        axes_linewidth=2.25
        tick_linewidth=axes_linewidth*.9
        tick_length=tick_linewidth*5
        
        # if fig:
        #     self.fig = fig
        # if ax:
        #     self.axs = ax
        
        self.fig, self.axs = plt.subplots(1, 2, layout='constrained', gridspec_kw={'width_ratios': [1, 0.25]})

        if drop is None:
            drop = []

        if self.info_df[izero].min() == 0:
            idx = self.info_df[izero].argmin()
            if idx not in drop:
                drop.append(idx)
            warnings.warn('I0 = 0 detected at row {}'.format(idx))
            
        if len(drop) > 0:
            column_numbers = [x for x in range(self.df.shape[1])]  # list of columns' integer indices
            for i in drop:
                column_numbers.remove(i) #removing column integer index idx
            self.df = self.df.iloc[:, column_numbers] #return all columns except the idx-th column
            self.info_df.drop(drop, inplace=True, axis=0)
            warnings.warn('Deleted row(s) {}.'.format(drop))
            
        I0 = np.array(self.info_df[izero])
        if self.calibration_data is not None:
            x = self.calibration_data
        else:
            x = np.array(self.df['X'])
        y = np.array(self.info_df['BL 8 Energy'])
        Z = np.array(self.df.iloc[:, 1:]).transpose()
        
        # print(I0)
        
        for i in range(len(I0)):
            Z[:,i] = Z[:,i]/I0[i]
            
        # print(Z.argmax())
        idxmin = abs(x - xlim[0]).argmin()
        idxmax = abs(x - xlim[1]).argmin()
        Z = (Z-Z.min())/(Z[:,idxmin:idxmax].max()-Z.min())
        # print(Z.max(keepdims=True))
        
        pc = self.axs[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap)
        if cbar:
            cbar = self.fig.colorbar(pc)

        # print(self.df.columns)

        # temp_df = self.df.drop(labels='X', axis=1)

        I = Z.sum(axis=1)
        ipfy = np.divide(1, Z[:,min(ipfy_lim):max(ipfy_lim)].sum(axis=1))

        tfy = np.array(self.info_df['TFY'])
        tey = np.array(self.info_df['TEY'])
        
        y_masked = []
        tfy_masked = []
        
        for i in range(len(tfy)):
            if tfy[i] != 8413656:
                y_masked.append(y[i])
                tfy_masked.append(tfy[i])
        
        # print(tfy*(tfy != 8413656))

        if plot_pfy:
            self.axs[1].plot((I-min(I))/(max(I)-min(I)), y, color=pfy_color)
        if plot_tfy:
            self.axs[1].plot((tfy-min(tfy))/(max(tfy)-min(tfy)), y, color=tfy_color)
        if plot_tfy_masked:
            self.axs[1].plot((tfy_masked-min(tfy_masked))/(max(tfy_masked)-min(tfy_masked)), y_masked, color=tfy_color)
        if plot_tey:
            self.axs[1].plot((tey-min(tey))/(max(tey)-min(tey)), y, color=tey_color)
        if plot_ipfy:
            self.axs[1].plot((ipfy-min(ipfy))/(max(ipfy)-min(ipfy)), y, color=ipfy_color)

        self.axs[0].set_xlim(xlim)
        self.axs[1].set_xlim([-0.1,1.1])
        self.axs[0].set_ylim([773,787])
        self.axs[1].set_ylim([773,787])


        for a in self.axs:
            # a.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad*2/3)
            # a.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad*2/3)
            a.tick_params(labelsize=fontsize)
            # a.xaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # a.xaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            # a.yaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # a.yaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            
            a.yaxis.set_major_locator(MultipleLocator(y_major_tick_multiple))
            a.yaxis.set_minor_locator(MultipleLocator(y_minor_tick_multiple))
        self.axs[0].set_xlabel('Emission Energy (eV)', fontsize=fontsize)
        self.axs[1].set_xlabel('Norm.\n Intensity', fontsize=fontsize)

        self.axs[0].set_ylabel('Excitation Energy (eV)', fontsize=fontsize)
        
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])
        
        self.axs[0].xaxis.set_minor_locator(MultipleLocator(x_minor_tick_multiple))
        
        self.fig.set_size_inches(*dim)
        if show:
            plt.show()
        
        if isinstance(header, str):
            self.axs[0].text(0.025, 0.95, header,
                        verticalalignment='top',
                        horizontalalignment='left',
                        transform=self.axs[0].transAxes,
                        fontsize=fontsize,
                        color='white')
        
        if plot_tfy_masked:
            self.tfy = (tfy_masked-min(tfy_masked))/(max(tfy_masked)-min(tfy_masked))
            self.y_masked = y_masked
        else:
            self.tfy = (tfy-min(tfy))/(max(tfy)-min(tfy))
        self.ipfy = (ipfy-min(ipfy))/(max(ipfy)-min(ipfy))
        self.y = y
        if savefig:
            plt.savefig(savefig)
            
    def plot_xes_legacy(
        self,
        idx=1,
        fig=None,
        ax=None,
        xlim=None,
        ylim=None,
        color=None,
        offset=0,
        dim=[3.25,3.25],
        savefig=None,
        filter=None,
        filter_args=None,
        filter_kwargs=None,
        norm='minmax',
        norm_kwargs=None,
        # fontsize=12,
        apply_plot_opts=True,
        plot_opts_kwargs={
            'xlabel': 'Emission Energy (eV)'
        }
    ):
        
        if fig is None or (ax is None):
            fig, ax = plt.subplots(layout='constrained')
        if self.calibration_data is not None:
            x = self.calibration_data
        else:
            x = self.df.iloc[:,0]
            
        if filter_args is None:
            filter_args = []
        if filter_kwargs is None:
            filter_kwargs = {}
            
        if filter:
            y = savgol_filter(self.df.iloc[:,idx], *filter_args, **filter_kwargs)
        else:
            y = self.df.iloc[:,idx]
            
        if xlim is not None:
            xidx = [np.argmin(abs(x-xlim[0])), np.argmin(abs(x-xlim[1]))]            
            x = x[xidx[0]:xidx[1]]
            y = y[xidx[0]:xidx[1]]
            
        if norm == 'minmax':
            y = (y-min(y))/(max(y)-min(y))
        elif norm == 'area':
            if norm_kwargs is None:
                norm_kwargs = {}
            _, _, _, area = Util.integrate(
                x,
                y,
                xlim,
                **norm_kwargs
            )
            y = y/area
            print(area)
            
        ax.plot(x, y+offset, color=color)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        # ax.tick_params(labelsize=fontsize)
            
        # ax.set_xlabel('Emission Energy (eV)', fontsize=fontsize)
        # ax.set_ylabel('Intensity (a.u.)', fontsize=fontsize)
        # ax.set_yticks([])
        # if x_minor_tick_multiple:
        #     ax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick_multiple))
        # if x_major_tick_multiple:
        #     ax.xaxis.set_major_locator(MultipleLocator(x_major_tick_multiple))
        
        # fig.set_size_inches(*dim)
        
        if plot_opts_kwargs is None:
            plot_opts_kwargs = {}
        if apply_plot_opts:
            Util.plot_opts(fig, ax, **plot_opts_kwargs)
        
        if savefig:
            fig.savefig(savefig)
        return fig, ax
            
            
class Xas:
    # TODO read csv header to JSON/dict
    def __init__(
        self,
        spec_dir,
        id,
        prefix='SigScan',
        calibration_data=None,
        skiprows=14,
        **kwargs
    ):
        self.spec_dir = pathlib.Path(spec_dir, **kwargs)
        data_list = []
        self.id_list = ['{}{}'.format(prefix, x) for x in id]

        self.child_list = self.spec_dir.glob('{}*.txt'.format(prefix))
        self.child_list = sorted(self.child_list)
        self.child_list = [x for x in self.child_list if x.stem in self.id_list]
        for c in self.child_list:
            data_list.append(pd.read_csv(c, skiprows=skiprows, sep=r'\t', engine='python'))
            
        self.data_list = data_list
        if len(self.data_list) > 1:
            self.df = reduce(lambda df0, df1: df0.combine(df1, lambda x0, x1: np.average([x0, x1], axis=0)), self.data_list)
            # self.df = pd.concat([d.set_index('X') for d in self.data_list], axis=1, join='inner').reset_index()
        else:
            self.df = self.data_list[0]
        # self.info_df = pd.read_csv(pathlib.Path(info_file, **kwargs), skiprows=12, sep=r'\t', engine='python')
        
    def plot(
        self,
        fig=None,
        ax=None,
        x='Mono Energy',
        y='Counter 2',
        i0='Counter 0',
        apply_plot_opts=True,
        norm='minmax',
        plot_opts_kwargs=None,
        offset=0,
        **kwargs
    ):
        if y == 'TEY':
            y = 'Counter 1'
        elif y == 'TFY':
            y = 'Counter 2'
        if not ax:
            fig, ax = plt.subplots(layout='constrained')
        y_arr = self.df[y]/self.df[i0]
        if norm:
            y_arr = (y_arr-min(y_arr))/(max(y_arr)-min(y_arr))
        y_arr += offset
        ax.plot(self.df['Mono Energy'], y_arr, **kwargs)
        
        if not plot_opts_kwargs:
            plot_opts_kwargs = {}
        
        if apply_plot_opts:
            Util.plot_opts(fig, ax, **plot_opts_kwargs)
            
        self.y_arr = y_arr
        
        return fig, ax
            
        
class Util:
    @staticmethod
    def bulk_data_read(
        dir,
        data_dir="Andor",
        info_file_suffix="AI.txt"
    ):
        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        
        dir_list = []
        info_file_list = []
        
        for c in sorted([x for x in dir.iterdir() if x.is_dir()]):
            dir_list.append(c/data_dir)
            
        for c in sorted(dir.glob('*/*{}'.format(info_file_suffix))):
            info_file_list.append(c)
            
        return dir_list, info_file_list
    
    @staticmethod
    def replace_entries(
        rixs0,
        rixs1,
        energy_tol = 0.015,
        energy_col = 'BL 8 Energy'
    ):
        '''
        Replaces data points in Rixs() object rixs0 with corresponding points from rixs1 based on excitation energy.
        
        ARGS:
            rixs0 (pyrixs.Rixs())
            rixs1 (pyrixs.Rixs())
            
        KWARGS:
            energy_tol: Tolerance for matching excitation energies.
        '''
        target_energies = rixs1.info_df[energy_col]
        target_indices = []
        for e in target_energies:
            target_indices.append(Util.find_closest_indices(rixs0.info_df[energy_col], e, energy_tol))

        for i in range(len(target_indices)):
            idx = target_indices[i]
            rixs0.info_df.iloc[idx,:] = rixs1.info_df.iloc[i,:]
            rixs0.info_df["Frame #"].iloc[idx] = float(idx+1)
            rixs0.df.iloc[:,idx+1] = rixs1.df.iloc[:,i+1]
        
    @staticmethod
    def find_closest_indices(
        arr,
        val,
        tol
    ):
        idxs = [ idx for idx,el in enumerate(arr) if (np.abs(el - val) < tol)]
        return idxs[0] if len(idxs) != 0 else np.nan
    
    @staticmethod
    def plot_opts(
        fig,
        ax,
        dim=[3.25,3.25],
        fontsize=12,
        xmintm=1,
        xmajtm=5,
        xlabel='Excitation Energy (eV)',
        ylabel='Intensity (a.u.)',
        xticks=None,
        yticks=[],
        axis_linewidth=None,
        linewidth=None
    ):
        fig.set_size_inches(*dim)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if xmintm:
            ax.xaxis.set_minor_locator(MultipleLocator(xmintm))
        if xmajtm:
            ax.xaxis.set_major_locator(MultipleLocator(xmajtm))
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(labelsize=fontsize)
        
        if axis_linewidth:
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(axis_linewidth)
            tick_linewidth = axis_linewidth*0.9
            tick_length = tick_linewidth*5
            ax.tick_params(width=tick_linewidth, which='both')
            ax.xaxis.set_tick_params(length=tick_length, which='major')
            ax.xaxis.set_tick_params(length=tick_length*0.5, which='minor')
            
        if linewidth:
            for line in ax.lines:
                line.set_linewidth(linewidth)
        
    # TODO add logic to input idxmin, idxmax directly instead of searching for xlim if desired
    @staticmethod
    def integrate(
        x,
        y,
        xlim,
        background='linear',
        background_points=[4,4]
    ):
        idxmin = abs(x - xlim[0]).argmin()
        idxmax = abs(x - xlim[1]).argmin()
        
        x = np.array(x)
        y = np.array(y)
        
        x = x[idxmin:idxmax]
        y = y[idxmin:idxmax]
        
        # x = x.iloc[idxmin:idxmax]
        # y = y.iloc[idxmin:idxmax]
        
        if background == 'linear':
            background_res = linregress(
                np.concatenate([x[:background_points[0]], x[-background_points[1]:]]),
                np.concatenate([y[:background_points[0]], y[-background_points[1]:]])
            )
            y_background = background_res.slope*x + background_res.intercept
            y_no_background = y - y_background
        else:
            y_no_background = y
            
        area = trapezoid(y_no_background, x=x)
            
        return x, y, y_background, area
